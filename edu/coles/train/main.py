from .datamodule import EventSequenceDataModule
from .lightning_module import CoLESLitModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint



def train(
    parquet_file: str = "data/train.parquet",
    seq_sample_len_min: int = 5,
    seq_sample_len_max: int = 401,
    min_seq_len: int = 0,
    batch_size: int = 512,
    epochs: int = 40,
    learning_rate: float = 1e-3,
    margin: float= 0.5,
    use_classical_loss: bool = True
):
    # Initialize DataModule
    data_module = EventSequenceDataModule(parquet_file=parquet_file, 
                                            seq_sample_len_min=seq_sample_len_min,
                                            seq_sample_len_max=seq_sample_len_max,
                                            batch_size=batch_size,
                                            min_seq_len=min_seq_len, num_workers=10)
    data_module.setup()

    # Initialize LightningModule (model)
    model = CoLESLitModule(num_domains=data_module.dataset.num_domains,
                            domain_embed_dim=128,
                            time_embed_dim=64,
                            event_hidden_dim=256,
                            seq_hidden_dim=256,
                            learning_rate=learning_rate,
                            margin=margin,
                            use_classical_loss=use_classical_loss)

    # Train with PyTorch Lightning Trainer

    # Define callbacks
    early_stopping_callback = EarlyStopping(monitor="train_loss", patience=4, mode="min", min_delta=5e-6)
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename='coles_{epoch}-{train_loss:.7f}',
        save_top_k=1
    )
    # Instantiate Trainer with callbacks
    trainer = pl.Trainer(
        max_epochs=epochs,
        # gradient_clip_val=1.0,
        accelerator='gpu',
        devices=[1],
        callbacks=[early_stopping_callback, checkpoint_callback]
    )
    trainer.fit(model, datamodule=data_module)