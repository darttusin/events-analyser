from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput

from transformers import BertConfig, BertModel

import json

import torch
import torch.nn as nn
from torch.optim import AdamW

import pytorch_lightning as pl

from sklearn.metrics import roc_auc_score

import numpy as np


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


class CustomModel(MLModel):
    async def load(self):
        config_dict = self._settings.parameters.extra["config"]
        tokenizer_path = self._settings.parameters.extra["tokenizer_path"]
        max_len = self._settings.parameters.extra["max_len"]

        self.tokenizer = json.load(open(tokenizer_path))
        config = BertConfig(**config_dict)
        self.model = BertEventClassifier(config=config, lr=1e-4, num_warmup_steps=0)
        self.model.load_state_dict(torch.load("data/model.pt", map_location="cpu"))
        self.model.eval()
        self.max_len = max_len

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        input_data = payload.inputs[0].data  # Expecting list of dicts [{"domain": [...], "timestamp": [...]}]
        batch = [self._preprocess(d) for d in input_data]
        input_ids = torch.tensor([b["input_ids"] for b in batch])
        attention_mask = torch.tensor([b["attention_mask"] for b in batch])

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs["logits"]).view(-1).tolist()

        return InferenceResponse(
            model_name=self.name,
            outputs=[
                ResponseOutput(name="probabilities", shape=[len(probs)], datatype="FP32", data=probs)
            ],
        )

    def _preprocess(self, d):
        seq = [self.tokenizer.get(s, 0) for s in d["domain"]]
        seq = [self.tokenizer.get("[CLS]", 0)] + seq[-self.max_len + 1:]
        attention = [1] * len(seq)

        if len(seq) < self.max_len:
            pad_len = self.max_len - len(seq)
            seq += [0] * pad_len
            attention += [0] * pad_len

        return {"input_ids": seq, "attention_mask": attention}
