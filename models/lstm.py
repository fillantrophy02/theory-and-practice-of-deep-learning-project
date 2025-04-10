import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd 

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

        # inputs : (batch_size, input_size)
        # hidden state: (batch_size, hidden_state)
        
        # Shape: (batch_size, input_size + hidden_size)
        # Adding horizontally

        # ENSURE SIZE for hidden state and cell state: (batch_size, hidden_size)
        if hidden_state is None:
            hidden_state = torch.zeros(inputs.size(0), self.hidden_size).to(inputs.device)
        
        if cell_state is None:
            cell_state = torch.zeros(inputs.size(0), self.hidden_size).to(inputs.device)
        
        combined = torch.cat((inputs, hidden_state), dim=1)

        forget_gate = torch.sigmoid(self.Wf(combined)) # (batch_size, hidden_size)
        input_gate = torch.sigmoid(self.Wi(combined))
        output_gate = torch.sigmoid(self.Wo(combined))
        candidate_cell = torch.tanh(self.Wc(combined))

        cell_state = forget_gate * cell_state + input_gate * candidate_cell

        hidden_state = output_gate * torch.tanh(cell_state)
        
        # Compute the output
        output = self.V(hidden_state)

        return cell_state, hidden_state, output



