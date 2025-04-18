import torch

# Define the model parameters
# Input Size : Number of features (22)
# Hidden_Size : 64
# Output_Size: 1

class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Initializing Weight Parameters
        #Input: input_size + hidden_size
        #Output: hidden_size

        # Forget gate
        self.Wf = torch.nn.Linear(input_size + hidden_size, hidden_size)

        # Input gate
        self.Wi = torch.nn.Linear(input_size + hidden_size, hidden_size)

        # Cell gate
        self.Wc = torch.nn.Linear(input_size + hidden_size, hidden_size)
        
        # Output gate
        self.Wo = torch.nn.Linear(input_size + hidden_size, hidden_size)

        # Prediction
        self.V = torch.nn.Linear(hidden_size, output_size)

    def forward(self, inputs, cell_state=None, hidden_state=None):

        # inputs: shape (batch_size, seq_len, input_size)
        # hidden state: (batch_size, hidden_state)
        
        # Shape: (batch_size, input_size + hidden_size)
        # Adding horizontally

        batch_size, seq_len, _ = inputs.size()

        if hidden_state is None:
            hidden_state = torch.zeros(inputs.size(0), self.hidden_size).to(inputs.device)
        
        if cell_state is None:
            cell_state = torch.zeros(inputs.size(0), self.hidden_size).to(inputs.device)

        outputs = []

        for t in range(seq_len):

            x_t = inputs[:, t, :]  # shape: (batch_size, input_size)

            combined = torch.cat((x_t, hidden_state), dim=1)  # (batch_size, input_size + hidden_size)

            forget_gate = torch.sigmoid(self.Wf(combined))
            input_gate = torch.sigmoid(self.Wi(combined))
            output_gate = torch.sigmoid(self.Wo(combined))
            candidate_cell = torch.tanh(self.Wc(combined))

            cell_state = forget_gate * cell_state + input_gate * candidate_cell
            hidden_state = output_gate * torch.tanh(cell_state)

            output = self.V(hidden_state)  # shape: (batch_size, output_size)
            outputs.append(output.unsqueeze(1))  # (batch_size, 1, output_size)

        outputs = torch.cat(outputs, dim=1)  # (batch_size, seq_len, output_size)

        return cell_state, hidden_state, outputs[:, -1, :]