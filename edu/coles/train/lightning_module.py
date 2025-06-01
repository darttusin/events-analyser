import torch
import pytorch_lightning as pl
from tqdm import tqdm

from .losses import classical_contrastive_loss, nt_xent_loss
from .model import CoLESEncoder


class CoLESLitModule(pl.LightningModule):
    def __init__(
        self,
        num_domains,
        domain_embed_dim=16,
        time_embed_dim=8,
        event_hidden_dim=32,
        seq_hidden_dim=64,
        learning_rate=1e-3,
        margin=1.0,
        use_classical_loss=True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["num_domains"])

        self.total_train_loss = 0.0
        self.num_train_batches = 0

        self.encoder = CoLESEncoder(
            num_domains,
            domain_embed_dim,
            time_embed_dim,
            event_hidden_dim,
            seq_hidden_dim,
        )

    def forward(self, time_diff, domain):
        return self.encoder(time_diff, domain)

    def training_step(self, batch, batch_idx):
        view1 = batch["view1"]
        view2 = batch["view2"]
        time1 = view1["time_diff"]
        domain1 = view1["domain"]
        time2 = view2["time_diff"]
        domain2 = view2["domain"]

        z1 = self(time1, domain1)
        z2 = self(time2, domain2) 

        if self.hparams.use_classical_loss:
            loss = classical_contrastive_loss(
                z1, z2, batch["is_same_pair"], margin=self.hparams.margin
            )
        else:
            loss = nt_xent_loss(z1, z2, temperature=0.5)

        self.total_train_loss += loss.item()
        self.num_train_batches += 1

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def epoch_end(self, epoch, epoch_idx):
        avg_epoch_loss = (
            self.total_train_loss / self.num_train_batches
            if self.num_train_batches > 0
            else 0.0
        )

        self.log("epoch_loss", avg_epoch_loss, prog_bar=True)

        self.total_train_loss = 0.0
        self.num_train_batches = 0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=1,
                threshold=1e-3,
            ),
            "monitor": "train_loss",
        }
        return [optimizer], [scheduler]

    def compute_embeddings(self, dataloader):
        self.encoder.eval()
        embeddings = {}
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader)):
                seq_ids = batch["sequence_id"]
                time_input = batch["view1"]["time_diff"].to(self.device)
                domain_input = batch["view1"]["domain"].to(self.device)
                z = self.encoder(time_input, domain_input)
                z = z.cpu().numpy()
                for sid, emb in zip(seq_ids, z):
                    embeddings[sid] = emb
        return embeddings