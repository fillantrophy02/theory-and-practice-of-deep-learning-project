import torch.nn as nn

class DecoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          dropout=dropout if num_layers > 1 else 0.0,
                          batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden):
        output, hidden = self.gru(input_seq, hidden)
        output = self.linear(output)
        return output, hidden