import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.0, num_layers=1):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, hidden=None):
        # inputs: (batch_size, seq_len, input_size)
        # hidden: (num_layers, batch_size, hidden_size) or None
        out, hidden = self.gru(inputs, hidden)  # out: (batch_size, seq_len, hidden_size)
        out = self.fc(out)  # Apply final layer to each time step → (batch_size, seq_len, output_size)
        out = out.squeeze(-1)  # Remove last dim if output_size is 1 → (batch_size, seq_len)
        return hidden[-1], out  # Return final hidden state and predictions
