import torch.nn as nn

class EncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          dropout=dropout if num_layers > 1 else 0.0,
                          batch_first=True)

    def forward(self, input_seq):
        _, hidden = self.gru(input_seq)
        return hidden