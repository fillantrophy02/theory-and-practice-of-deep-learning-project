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
        self.decoder1 = nn.TransformerDecoderLayer(
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
        memory = self.encoder1(x) 
        memory = self.encoder2(memory) #(batch_size, input_seq_length, embed_dim)

        out_embeddings = torch.empty(x.size()[0], 0, embed_dim).to(device)
        tgt = x[:, -1:, :] # (batch_size, 1, embed_dim)
        for t in range(target_seq_length):
            decoded = self.decoder1(tgt, memory) # (batch_size, 1, embed_dim)
            out_embeddings = torch.cat((out_embeddings, decoded), dim=1)
            tgt = decoded # this becomes decoder input for next time step

        out = self.linear2(out_embeddings) # (batch_size, target_seq_length, 1)
        out = self.sigmoid(out)
        return out