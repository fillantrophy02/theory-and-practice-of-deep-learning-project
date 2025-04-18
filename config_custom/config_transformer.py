import torch

debug_mode = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_split = 0.8
val_split = 0.2
batch_size = 512
num_epochs = 50
no_of_days = 7
# excluded_features = ["WindDir3pm_cos", "WindDir3pm_sin", "WindDir9am_cos", "WindDir9am_sin", "WindGustDir_cos", "WindGustDir_sin"]
excluded_features = []
num_features = 20 - len(excluded_features)

embed_dim = num_features # embedding dim
d_model = embed_dim
seq_length = no_of_days # input seq length
target_seq_length = 1 # output seq length, today, tmr, and day after tmr
nhead = 4 # no. of heads
num_encoder_layers = 2
num_decoder_layers = 2
dim_feedforward = 512
