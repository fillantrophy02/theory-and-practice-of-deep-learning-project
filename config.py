import torch

debug_mode = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 256
num_epochs = 15
no_of_days = 10
num_features = 19

embed_dim = 16 # embedding dim
seq_length = no_of_days
target_seq_length = 3 # today, tmr, and day after tmr
nhead = 4 # no. of heads
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 512