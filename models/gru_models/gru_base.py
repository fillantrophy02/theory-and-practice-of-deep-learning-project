import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd 

# Define the model parameters
# Input Size : Number of features (22)
# Hidden_Size : 64
# Output_Size: 1

class GRU(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Parameters for the reset gate
        self.Wr = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.Ur = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.br = torch.nn.Parameter(torch.zeros(hidden_size))
        # Parameters for the update gate
        self.Wz = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.Uz = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bz = torch.nn.Parameter(torch.zeros(hidden_size))
        # Parameters for the candidate hidden state
        self.W = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.U = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bc = torch.nn.Parameter(torch.zeros(hidden_size))

        # Parameters for the output prediction
        self.V = torch.nn.Parameter(torch.randn(hidden_size, 1))
        self.bo = torch.nn.Parameter(torch.randn(1))

    def forward(self, inputs, hidden):
        # Compute the reset gate
        reset_gate = torch.sigmoid(torch.matmul(inputs, self.Wr) + torch.matmul(hidden, self.Ur) + self.br)
        # Compute the update gate
        update_gate = torch.sigmoid(torch.matmul(inputs, self.Wz) + torch.matmul(hidden, self.Uz) + self.bz)

        # Compute the candidate hidden state
        candidate_hidden = torch.tanh(torch.matmul(inputs, self.W) + torch.matmul(reset_gate * hidden, self.U) + self.bc)
        # Compute the updated hidden state
        new_hidden = (1 - update_gate) * hidden + update_gate * candidate_hidden

        # Compute the output
        output = torch.matmul(new_hidden, self.V).squeeze(1) + self.bo  # [batch_size]


        return new_hidden, output



