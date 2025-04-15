import torch
from torch import nn
from config import *

class TransformerForClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.encoder2 = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.linear1 = nn.Linear(num_features, embed_dim)
        self.linear2 = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

    def forward(self, x):
        # x: (batch_size, input_seq_length, num_features)
        # since embed_dim need to be divisible by nheads
        # and our num_features is unfortunately 19
        # need to resize first
        x = self.linear1(x)
        # src: (batch_size, input_seq_length, embed_dim),
        # need to apply transformer(src)
        x = self.encoder1(x) 
        x = self.encoder2(x)
        # x:  (batch_size, input_seq_length, embed_dim)
        x = x[:, -1, :]
        # x: (batch_size, embed_dim)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x