import math
import numpy as np
import torch
from torch import nn
from config_custom.config_transformer import *
import torch.nn.functional as F

class TransformerForClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, batch_first=True)
        self.linear1 = nn.Linear(num_features, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout()
        self.loss = nn.BCELoss()

    def forward(self, x):
        src = x
        tgt = x[:, -1:, :]
        out = self.transformer(src, tgt)
        out = out[:, -1:, :]
        out = self.dropout(self.activation(self.linear1(out)))
        out = self.dropout(self.activation(self.linear2(out)))
        out = self.sigmoid(self.linear3(out))
        return out