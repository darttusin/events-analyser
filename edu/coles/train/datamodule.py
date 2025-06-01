from .dataset import EventSequenceDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class EventSequenceDataModule(pl.LightningDataModule):
    def __init__(self, parquet_file, seq_sample_len_min=50,seq_sample_len_max=50, batch_size=64, num_workers=2, min_seq_len = 10):
        super().__init__()
        self.parquet_file = parquet_file
        self.seq_sample_len_min = seq_sample_len_min
        self.seq_sample_len_max = seq_sample_len_max
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.min_seq_len = min_seq_len

    def setup(self, stage=None):
        self.dataset = EventSequenceDataset(self.parquet_file,
                                             seq_sample_len_min=self.seq_sample_len_min,
                                             seq_sample_len_max=self.seq_sample_len_max,
                                               min_seq_len=self.min_seq_len)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers)