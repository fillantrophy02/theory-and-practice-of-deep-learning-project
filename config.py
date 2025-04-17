import torch

debug_mode = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_split = 0.8
val_split = 0.2
batch_size = 512
num_epochs = 50
no_of_days = 21
num_features = 19

embed_dim = 16 # embedding dim
d_model = embed_dim
seq_length = no_of_days # input seq length
target_seq_length = 1 # output seq length, today, tmr, and day after tmr
nhead = 4 # no. of heads
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 512