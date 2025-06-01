import torch
import torch.nn as nn


class CoLESEncoder(nn.Module):
    def __init__(
        self,
        num_domains,
        domain_embed_dim=16,
        time_embed_dim=8,
        event_hidden_dim=32,
        seq_hidden_dim=64,
    ):
        super(CoLESEncoder, self).__init__()
        self.domain_embedding = nn.Embedding(
            num_domains, domain_embed_dim, padding_idx=0
        )
        self.time_linear = nn.Linear(1, time_embed_dim)
        combined_dim = domain_embed_dim  # + time_embed_dim
        self.event_fc = nn.Linear(combined_dim, event_hidden_dim)
        self.relu = nn.ReLU()

        self.gru = nn.LSTM(
            input_size=event_hidden_dim,
            hidden_size=seq_hidden_dim,
            batch_first=True,
        )
        # self.gru = nn.GRU(
        #     event_hidden_dim,
        #     seq_hidden_dim,
        #     batch_first=True,
        # )

        self.mlp = nn.Sequential(
            nn.Linear(seq_hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, time_diff, domain):
        domain_emb = self.domain_embedding(domain) 
        # time_input = time_diff.unsqueeze(-1)         
        # time_emb = self.time_linear(time_input)    
        event_feat = torch.cat([domain_emb], dim=-1)  
        event_emb = self.relu(self.event_fc(event_feat)) 
        _, h_n = self.gru(event_emb)  
        seq_embedding = h_n[0].squeeze(0) 
        # seq_embedding = F.normalize(seq_embedding, dim=-1)
        return self.mlp(seq_embedding)