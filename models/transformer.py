import math
import numpy as np
import torch
from torch import nn
from config import *
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

        product = torch.bmm(query, query.transpose(1, 2))/(embed_dim**0.5) # (n, s, s)
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
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first: bool = False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # [d_model/2]
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]`` or ``[batch_size, seq_len, embedding_dim]`` if batch_first=True
        """
        if self.batch_first:
            seq_len = x.size(1)
            x = x + self.pe[:seq_len].transpose(0, 1)  # [1, seq_len, d_model]
        else:
            seq_len = x.size(0)
            x = x + self.pe[:seq_len]  # [seq_len, 1, d_model]
        return self.dropout(x)
    
# Copied from transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding with Marian->Informer
class InformerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int) -> None:
        super().__init__(num_positions, embedding_dim)

    def _init_weight(self):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = self.weight.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out = torch.empty(n_pos, dim, dtype=self.weight.dtype, requires_grad=False, device=device)
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]), device=device)
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]), device=device)
        self.weight = nn.Parameter(out, requires_grad=False)

    @torch.no_grad()
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=device
        )
        return super().forward(positions)

class TransformerForClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = nn.TransformerEncoderLayer(embed_dim, nhead, dim_feedforward, batch_first=True) # EncoderLayer(dropout=0.5)
        self.encoder2 = nn.TransformerEncoderLayer(embed_dim, nhead, dim_feedforward, batch_first=True) # EncoderLayer(dropout=0.5)
        self.encoder3 = nn.TransformerEncoderLayer(embed_dim, nhead, dim_feedforward, batch_first=True) # EncoderLayer(dropout=0.5)
        self.encoder4 = nn.TransformerEncoderLayer(embed_dim, nhead, dim_feedforward, batch_first=True) # EncoderLayer(dropout=0.5)
        self.encoder5 = nn.TransformerEncoderLayer(embed_dim, nhead, dim_feedforward, batch_first=True) # EncoderLayer(dropout=0.5)
        self.encoder6 = nn.TransformerEncoderLayer(embed_dim, nhead, dim_feedforward, batch_first=True) # EncoderLayer(dropout=0.5)
        self.decoder1 = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=0.5)
        self.decoder2 = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=0.5)
        self.decoder3 = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=0.5)
        self.decoder4 = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=0.5)
        self.decoder5 = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=0.5)
        self.decoder6 = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=0.5)
        # self.position = InformerSinusoidalPositionalEmbedding(seq_length, embed_dim)
        self.linear1 = nn.Linear(num_features, embed_dim)
        self.linear2 = nn.Linear(embed_dim, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, 1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout()
        self.loss = nn.BCELoss()

    def forward(self, x):
        # x: (batch_size, input_seq_length, num_features)
        # since embed_dim need to be divisible by nheads
        # and our num_features is unfortunately 19
        # need to resize first
        x = self.dropout(self.linear1(x))

        # src: (batch_size, input_seq_length, embed_dim),
        # need to apply transformer(src)
        memory = self.encoder1(x)
        memory = self.encoder2(memory) # (batch_size, input_seq_length, embed_dim)
        # memory = self.encoder3(memory) # (batch_size, input_seq_length, embed_dim)
        # memory = self.encoder4(memory) # (batch_size, input_seq_length, embed_dim)
        # memory = self.encoder5(memory) # (batch_size, input_seq_length, embed_dim)
        # memory = self.encoder6(memory) # (batch_size, input_seq_length, embed_dim)

        out_embeddings = torch.empty(x.size()[0], 0, embed_dim).to(device)
        # tgt = x[:, -1:, :] # (batch_size, 1, embed_dim)
        # for t in range(target_seq_length):
        #     tgt = self.decoder1(tgt, memory) # (batch_size, 1, embed_dim)
        #     tgt = self.decoder2(tgt, memory) # (batch_size, 1, embed_dim)
        #     # tgt = self.decoder3(tgt, memory) # (batch_size, 1, embed_dim)
        #     # tgt = self.decoder4(tgt, memory) # (batch_size, 1, embed_dim)
        #     # tgt = self.decoder5(tgt, memory) # (batch_size, 1, embed_dim)
        #     # tgt = self.decoder6(tgt, memory) # (batch_size, 1, embed_dim)
        #     out_embeddings = torch.cat((out_embeddings, tgt), dim=1)

        out_embeddings = memory[:, -1:, :]
        out = self.linear2(out_embeddings) # (batch_size, 1, 1)
        out = self.activation(out)
        out = self.linear3(out)
        out = self.activation(out)
        out = self.linear4(out)
        out = self.sigmoid(out)
        return out