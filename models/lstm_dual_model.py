import torch

class DualMemoryCellLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DualMemoryCellLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # MODIFICATION: Primary memory cell gates
        self.Wf1 = torch.nn.Linear(input_size + hidden_size, hidden_size)  # Forget gate 1
        self.Wi1 = torch.nn.Linear(input_size + hidden_size, hidden_size)  # Input gate 1
        self.Wc1 = torch.nn.Linear(input_size + hidden_size, hidden_size)  # Cell gate 1
        self.Wo1 = torch.nn.Linear(input_size + hidden_size, hidden_size)  # Output gate 1
        
        # MODIFICATION: Secondary memory cell gates
        self.Wf2 = torch.nn.Linear(input_size + hidden_size, hidden_size)  # Forget gate 2
        self.Wi2 = torch.nn.Linear(input_size + hidden_size, hidden_size)  # Input gate 2
        self.Wc2 = torch.nn.Linear(input_size + hidden_size, hidden_size)  # Cell gate 2
        self.Wo2 = torch.nn.Linear(input_size + hidden_size, hidden_size)  # Output gate 2
        
        # MODIFICATION: Integration Gate
        self.Wg = torch.nn.Linear(input_size + hidden_size, hidden_size)
        
        # Prediction layer
        self.V = torch.nn.Linear(hidden_size, output_size)

    def forward(self, inputs, cell_states=None, hidden_state=None):
        batch_size, seq_len, _ = inputs.size()

        if hidden_state is None:
            hidden_state = torch.zeros(inputs.size(0), self.hidden_size).to(inputs.device)
        
        if cell_states is None:
            cell_state1 = torch.zeros(inputs.size(0), self.hidden_size).to(inputs.device)
            cell_state2 = torch.zeros(inputs.size(0), self.hidden_size).to(inputs.device)
        else:
            cell_state1, cell_state2 = cell_states

        outputs = []

        for t in range(seq_len):
            x_t = inputs[:, t, :]  # shape: (batch_size, input_size)
            combined = torch.cat((x_t, hidden_state), dim=1)  # (batch_size, input_size + hidden_size)

            # Primary memory cell
            forget_gate1 = torch.sigmoid(self.Wf1(combined))
            input_gate1 = torch.sigmoid(self.Wi1(combined))
            output_gate1 = torch.sigmoid(self.Wo1(combined))
            candidate_cell1 = torch.tanh(self.Wc1(combined))
            cell_state1 = forget_gate1 * cell_state1 + input_gate1 * candidate_cell1
            
            # Secondary memory cell
            forget_gate2 = torch.sigmoid(self.Wf2(combined))
            input_gate2 = torch.sigmoid(self.Wi2(combined))
            output_gate2 = torch.sigmoid(self.Wo2(combined))
            candidate_cell2 = torch.tanh(self.Wc2(combined))
            cell_state2 = forget_gate2 * cell_state2 + input_gate2 * candidate_cell2
            
            # Integration gate - determines how to mix information from both cells
            # A value close to 1 favors cell 1, close to 0 favors cell 2
            integration_gate = torch.sigmoid(self.Wg(combined))
            
            # Compute hidden state using both cell states
            hidden_cell1 = output_gate1 * torch.tanh(cell_state1)
            hidden_cell2 = output_gate2 * torch.tanh(cell_state2)
            
            # Weighted combination of the two cell outputs
            hidden_state = integration_gate * hidden_cell1 + (1 - integration_gate) * hidden_cell2

            output = self.V(hidden_state)  # shape: (batch_size, output_size)
            outputs.append(output.unsqueeze(1))  # (batch_size, 1, output_size)

        outputs = torch.cat(outputs, dim=1)  # (batch_size, seq_len, output_size)

        return (cell_state1, cell_state2), hidden_state, outputs[:, -1, :]