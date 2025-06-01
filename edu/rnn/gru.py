import torch
import torch.nn as nn
from torch.optim import AdamW
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pyarrow.parquet as pq

def read_parquet_skip_broken_rows(
    parquet_file, columns, batch_size=400_000, rows=2_000_000
):
    pf = pq.ParquetFile(parquet_file)
    batches = []
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        try:
            df = batch.to_pandas()
            batches.append(df)
        except Exception as e:
            print(f"Skipping a batch due to error: {e}")
        if len(batches) * batch_size > rows:
            break
    return pd.concat(batches, ignore_index=True) if batches else pd.DataFrame()

df = read_parquet_skip_broken_rows(
    "final_data.parquet", 
    columns=["target", "_seq_len", "domain", "user_id"]
)

users = df["user_id"].unique()
train_ids, test_ids = train_test_split(users, test_size=0.1, random_state=42)
df_train = df[df["user_id"].isin(train_ids)]
df_test = df[df["user_id"].isin(test_ids)]
del df

class EventSequenceDataset(Dataset):
    def __init__(self, df, labels, max_len, tokenizer=None):
        self.df = df
        self.labels = labels
        self.max_len = max_len
        self.tokenizer = tokenizer if tokenizer else self.build_tokenizer(df)
        self.df["tokenized"] = self.df["domain"].apply(self.tokenize_sequence)

    def build_tokenizer(self, df):
        unique_events = set([event for seq in df["domain"] for event in seq])
        tokenizer = {event: idx + 1 for idx, event in enumerate(unique_events)}
        tokenizer["[PAD]"] = 0
        tokenizer["[CLS]"] = len(tokenizer)
        return tokenizer

    def tokenize_sequence(self, seq):
        return [
            self.tokenizer.get(event, 0)
            for event in seq
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sequence = self.df.iloc[idx]["tokenized"]
        label = self.labels.iloc[idx]

        if len(sequence) < self.max_len:
            pad_len = self.max_len - len(sequence)
            input_ids = [0] * pad_len + sequence  # Pre-padding
            attention_mask = [0] * pad_len + [1] * len(sequence)
        else:
            input_ids = sequence[-self.max_len :]  # Truncate from the start
            attention_mask = [1] * self.max_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.float),
        }

class GRUEventClassifier(pl.LightningModule):
    def __init__(
        self, vocab_size, embed_dim, hidden_size, lr, num_layers=1, dropout=0.2
    ):
        super().__init__()
        self.save_hyperparameters()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(  # Changed from LSTM to GRU
            embed_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.preds_epoch = []
        self.labels_epoch = []
        self.lr = lr

    def forward(self, input_ids, attention_mask, labels=None):
        x = self.embedding(input_ids)
        output, h_n = self.gru(x)  # GRU returns (output, hidden)
        last_hidden_state = h_n[-1]  # Use last layer hidden state
        logits = self.classifier(last_hidden_state)
        loss = (
            self.loss_fn(logits.view(-1), labels.view(-1))
            if labels is not None
            else None
        )
        return {"loss": loss, "logits": logits}

    def training_step(self, batch, batch_idx):
        out = self(**batch)
        self.log("train_loss", out["loss"], prog_bar=True, on_epoch=True)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self(**batch)
        preds = torch.sigmoid(out["logits"]).detach().cpu()
        labels = batch["labels"].detach().cpu()
        self.preds_epoch.append(preds)
        self.labels_epoch.append(labels)
        self.log("val_loss", out["loss"], prog_bar=True)
        return out["loss"]

    def on_validation_epoch_end(self):
        preds = torch.cat(self.preds_epoch).numpy()
        labels = torch.cat(self.labels_epoch).numpy()
        
        auc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0
        self.log("val_roc_auc", auc, prog_bar=True)
        
        if len(np.unique(labels)) > 1:
            binary_preds = (preds > 0.5).astype(int)
            self.log("val_fscore_macro", f1_score(labels, binary_preds, average="macro"), prog_bar=True)
            self.log("val_fscore_micro", f1_score(labels, binary_preds, average="micro"), prog_bar=True)
            self.log("val_fscore", f1_score(labels, binary_preds), prog_bar=True)

        self.preds_epoch.clear()
        self.labels_epoch.clear()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.2,
            patience=1,
            min_lr=1e-5,
            threshold=1e-3,
            verbose=True,
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

SEQ_LEN = 70 
EMBED_DIM = 64
HIDDEN_SIZE = 128
LR = 1e-4
NUM_LAYERS = 1

train_dataset = EventSequenceDataset(
    df_train.drop("target", axis=1), df_train["target"], max_len=SEQ_LEN
)
val_dataset = EventSequenceDataset(
    df_test.drop("target", axis=1),
    df_test["target"],
    max_len=SEQ_LEN,
    tokenizer=train_dataset.tokenizer,
)

train_loader = DataLoader(
    train_dataset, batch_size=512, shuffle=True, drop_last=True, num_workers=30
)
val_loader = DataLoader(val_dataset, batch_size=512, num_workers=30)

vocab_size = len(train_dataset.tokenizer)
model = GRUEventClassifier(
    vocab_size=vocab_size, 
    embed_dim=EMBED_DIM, 
    hidden_size=HIDDEN_SIZE, 
    lr=LR, 
    num_layers=NUM_LAYERS
)

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

early_stopping_callback = EarlyStopping(
    monitor="val_loss", patience=5, mode="min", min_delta=1e-5
)
lr_monitor = LearningRateMonitor(logging_interval="epoch")
checkpoint_callback = ModelCheckpoint(
    dirpath="edu/checkpoints",
    mode="max",
    monitor="val_roc_auc",
    filename="gru_ds2kk_len70_64_128_{epoch}-{val_roc_auc:.3f}-{val_fscore:.3f}",
    save_top_k=1,
)

trainer = pl.Trainer(
    max_epochs=25, 
    callbacks=[early_stopping_callback, lr_monitor, checkpoint_callback],
    accelerator='gpu',
    devices=[1]
)

trainer.fit(model, train_loader, val_loader)