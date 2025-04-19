import math
import numpy as np
import torch
from torch import nn
from config_custom.config_transformer import *
import torch.nn.functional as F


# Define a self-attention layer implementation
class SelfAttentionLayer(nn.Module):
    def __init__(self):
        super(SelfAttentionLayer, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x): # (n, s, e)
        batch_size = x.size(0)
        query = self.query(x).view(batch_size, -1, embed_dim) # (n, s, e)
        key = self.key(x).view(batch_size, -1, embed_dim) # (n, s, e)
        value = self.value(x).view(batch_size, -1, embed_dim) # (n, s, e)

        product = torch.bmm(query, key.transpose(1, 2))/(embed_dim**0.5) # (n, s, s)
        attention_weights = F.softmax(product, dim = 2) # (n, s, s)
        out = torch.bmm(attention_weights, value) # (n, s, e)
        return out
    
class EncoderLayer(nn.Module):
    def __init__(self, dropout = 0.1, layer_norm_eps = 1e-5, activation = F.relu):
        super(EncoderLayer, self).__init__()
        self.attn = SelfAttentionLayer()

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout) 
        self.activation = activation

    def _ff_block(self, x):
        x = self.linear2(self.dropout1(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, x): # (n, s, e)
        x = self.norm1(x + self.dropout1(self.attn(x)))
        x = self.norm2(x + self._ff_block(x))

        return x

class TransformerForClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = EncoderLayer()
        self.encoder2 = EncoderLayer()
        self.encoder3 = EncoderLayer()
        self.linear1 = nn.Linear(num_features, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout()
        self.loss = nn.BCELoss()

    def forward(self, x):
        x = self.encoder1(x)
        x = self.encoder2(x)
        memory = self.encoder3(x)
        out = memory[:, -1:, :]
        out = out[:, -1:, :]
        out = self.dropout(self.activation(self.linear1(out)))
        out = self.dropout(self.activation(self.linear2(out)))
        out = self.sigmoid(self.linear3(out))
        return out