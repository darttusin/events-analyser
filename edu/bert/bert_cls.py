import torch
import torch.nn as nn
from torch.optim import AdamW
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score
from transformers import BertModel, BertConfig, get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

import pyarrow.parquet as pq
import pyarrow as pa

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True, nb_workers=30)

df_train = pd.read_parquet('train.parquet')[:100_000]
df_test = pd.read_parquet('test.parquet')


class EventSequenceDataset(Dataset):
    def __init__(self, df, labels, max_len, tokenizer=None):
        self.df = df
        self.labels = labels
        self.max_len = max_len

        if tokenizer is None:
            self.tokenizer = self.build_tokenizer(self.df)
            import json

            with open("tokenizer.json", "w") as f:
                json.dump(self.tokenizer, f, indent=4)
        else:
            self.tokenizer = tokenizer

        self.df["tokenized"] = self.df["domain"].apply(self.tokenize_sequence)

    def build_tokenizer(self, df):
        """
        Build a simple tokenizer from the event sequences.
        """
        unique_events = set([event for seq in df["domain"] for event in seq])
        tokenizer = {event: idx for idx, event in enumerate(unique_events)}
        tokenizer["[CLS]"] = len(unique_events)
        return tokenizer

    def tokenize_sequence(self, seq):
        sequence = [
            self.tokenizer.get(event, 0)
            for event in seq
            # if event not in ["OFFER_ACCEPTED", "OFFER_DECLINED"]
        ] 
        return sequence

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sequence = self.df.iloc[idx]["tokenized"][-self.max_len + 1 :]
        label = self.labels.iloc[idx]

        sequence = [self.tokenizer.get("[CLS]", 0)] + sequence


        if len(sequence) < self.max_len:
            input_ids = sequence + [0] * (
                self.max_len - len(sequence)
            )  
            attention_mask = [1] * len(sequence) + [0] * (
                self.max_len - len(sequence)
            )  
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(label, dtype=torch.float),
            }
        else:
            input_ids = sequence
            attention_mask = [
                1
            ] * self.max_len 

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(label, dtype=torch.float),
            }


class BertEventClassifier(pl.LightningModule):
    def __init__(self, config, lr, num_warmup_steps, class_weights=None):
        super().__init__()
        self.bert = BertModel(config)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self.lr = lr
        self.preds_epoch = []
        self.labels_epoch = []
        self.train_preds_epoch = []
        self.train_labels_epoch = []
        self.num_warmup_steps = num_warmup_steps

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

        hidden_states = outputs.last_hidden_state  
        cls_token_hidden_state = hidden_states[:, 0, :]  

        logits = self.classifier(cls_token_hidden_state)

        loss = (
            self.loss_fn(logits.float().view(-1), labels.float().view(-1))
            if labels is not None
            else None
        )
        return {"loss": loss, "logits": logits}

    def training_step(self, batch, batch_idx):
        out = self(**batch)
        preds = torch.sigmoid(out["logits"]).detach().cpu()
        labels = batch["labels"].detach().cpu()
        self.log("train_loss", out["loss"], prog_bar=True, on_epoch=True)
        self.train_preds_epoch.append(preds)
        self.train_labels_epoch.append(labels)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self(**batch)
        preds = torch.sigmoid(out["logits"]).detach().cpu()
        labels = batch["labels"].detach().cpu()

        self.log("val_loss", out["loss"], prog_bar=True)
        self.preds_epoch.append(preds)
        self.labels_epoch.append(labels)
        return out["loss"]

    def on_validation_epoch_end(self):
        preds = torch.cat(self.preds_epoch)
        labels = torch.cat(self.labels_epoch)
        auc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0
        self.log("val_roc_auc", auc, prog_bar=True)



        self.preds_epoch.clear()
        self.labels_epoch.clear()

    def on_train_epoch_end(self):
        preds = torch.cat(self.train_preds_epoch).numpy()
        labels = torch.cat(self.train_labels_epoch).numpy()
        auc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0
        self.log("train_roc_auc", auc, prog_bar=True)

        self.train_preds_epoch.clear()
        self.train_labels_epoch.clear()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            threshold_mode="abs",
            factor=0.2,
            patience=1,
            min_lr=1e-5,
            verbose=True,
            threshold=1e-3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


def calculate_class_weights(train_labels):
    class_weights = compute_class_weight(
        "balanced", classes=np.array([0, 1]), y=train_labels
    )
    return torch.tensor([class_weights[1] / class_weights[0]]).float()


train_dataset = EventSequenceDataset(
    df_train.drop("target", axis=1), df_train["target"], max_len=200
)
val_dataset = EventSequenceDataset(
    df_test.drop("target", axis=1),
    df_test["target"],
    max_len=200,
    tokenizer=train_dataset.tokenizer,
)
class_weights = calculate_class_weights(df_train["target"])

del df_train, df_test

train_loader = DataLoader(
    train_dataset, batch_size=256, shuffle=True, drop_last=True, num_workers=30
)
val_loader = DataLoader(val_dataset, batch_size=256, num_workers=30)



num_warmup_steps = int(0.1 * len(train_loader)) 

config = BertConfig(
    hidden_size=64,
    num_hidden_layers=3,
    num_attention_heads=2,
    intermediate_size=256,
    max_position_embeddings=200,
    num_labels=1,
    pad_token_id=0,
)


model = BertEventClassifier(
    config=config,
    lr=1e-4,
    num_warmup_steps=num_warmup_steps,
)

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

early_stopping_callback = EarlyStopping(
    monitor="val_loss", patience=2, mode="min", min_delta=1e-5
)
lr_monitor = LearningRateMonitor(logging_interval="epoch")
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/final",
    mode="max",
    monitor="val_roc_auc",
    filename="bert_ds8kk_len_200_classweights_final_final_64_256_{epoch}-{val_roc_auc:.3f}-{val_loss:.3f}-{train_loss_epoch:.3f}",
    save_top_k=1,
)

trainer = pl.Trainer(
    max_epochs=2, callbacks=[early_stopping_callback, lr_monitor, checkpoint_callback], accelerator='gpu',
    devices=[1]
)

trainer.fit(
    model,
    train_loader,
    val_loader,
)

# Сохраняем модель (state_dict)
torch.save(model.state_dict(), "bert_event_classifier.pt")

# Сохраняем tokenizer отдельно
import json
with open("tokenizer.json", "w") as f:
    json.dump(train_dataset.tokenizer, f, indent=4)