import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class EventSequenceDataset(Dataset):
    def __init__(
        self,
        parquet_file,
        seq_sample_len_min=50,
        seq_sample_len_max=50,
        min_seq_len=10,
        domain_vocab=None,
    ):

        self.df = pd.read_parquet(parquet_file)
        self.df = self.df[self.df["_seq_len"] >= min_seq_len].reset_index(drop=True)
        self.seq_sample_len_min = seq_sample_len_min
        self.seq_sample_len_max = seq_sample_len_max
        self.inference_mode = False
        if domain_vocab is None:
            all_domains = set()
            for domains in self.df["domain"].tolist():
                all_domains.update(domains)
            self.domain_vocab = {
                dom: idx for idx, dom in enumerate(sorted(all_domains))
            }
        else:
            self.domain_vocab = domain_vocab
        self.num_domains = len(self.domain_vocab)

    def to_inference_mode(self):
        self.inference_mode = True
        return self

    def scale_time_series(self, time_arr):
        time_series = np.copy(time_arr)

        time_series = (time_series - time_series.min()) / (
            time_series.max() - time_series.min()
        )

        return torch.tensor(time_series)

    def random_slice(self, arr, time_arr):
        L = len(arr)
        # if L >= self.seq_sample_len:

        rand_length = random.randint(self.seq_sample_len_min, self.seq_sample_len_max)

        start = random.randint(0, max(0, L - self.seq_sample_len_min))
        sliced = arr[start : start + rand_length]

        sliced_time = time_arr[start : start + rand_length]
        # else:
        #     padded = np.zeros(self.seq_sample_len, dtype=arr.dtype)
        #     padded[:L] = arr
        #     sliced = padded
        return sliced, self.scale_time_series(sliced_time)

    def process_domains(self, domains):
        return torch.tensor([self.domain_vocab.get(d, 0) for d in domains])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # time_diff_arr = row['time_diff'].tolist()  # Convert to list for manipulation
        domain_arr = row["domain"]
        domain_indices = self.process_domains(domain_arr)[-self.seq_sample_len_max :]

        if self.inference_mode:
            return {
                "sequence_id": row["sequence_id"],
                "target": row["target"],
                "view1": {
                    # "time_diff": torch.nn.functional.pad(
                    #     row["time_diff"],
                    #     (self.seq_sample_len_max - len(domain_indices), 0),
                    #     value=0,
                    # ),
                    "time_diff": torch.tensor(0),
                    "domain": torch.nn.functional.pad(
                        domain_indices,
                        (self.seq_sample_len_max - len(domain_indices), 0),
                        value=0,
                    ),
                },
            }

        domain_view1, domain_time1 = self.random_slice(domain_indices, row["time_diff"])

        same_row_p = 0.1

        if np.random.random() > same_row_p:
            idx2 = random.randint(0, len(self) - 1)
            domain_indices2 = self.process_domains(self.df.iloc[idx2]["domain"])
            domain_view2, domain_time2 = self.random_slice(
                domain_indices2, self.df.iloc[idx2]["time_diff"]
            )

            is_same = 0
        else:
            domain_view2, domain_time2 = self.random_slice(
                domain_indices, row["time_diff"]
            )
            is_same = 1

        sample = {
            "sequence_id": row["sequence_id"],
            "target": row["target"],
            "view1": {
                # "time_diff": torch.nn.functional.pad(
                #     domain_time1,
                #     (self.seq_sample_len_max - len(domain_view1), 0),
                #     value=-1,
                # ),
                "time_diff": torch.tensor(0),
                "domain": torch.nn.functional.pad(
                    domain_view1,
                    (self.seq_sample_len_max - len(domain_view1), 0),
                    value=0,
                ),
            },
            "view2": {
                # "time_diff": torch.nn.functional.pad(
                #     domain_time2,
                #     (self.seq_sample_len_max - len(domain_view2), 0),
                #     value=-1,
                # ),
                "time_diff": torch.tensor(0),
                "domain": torch.nn.functional.pad(
                    domain_view2,
                    (self.seq_sample_len_max - len(domain_view2), 0),
                    value=0,
                ),
            },
            "is_same_pair": is_same,
        }
        return sample
