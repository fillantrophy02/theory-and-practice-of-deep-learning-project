import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 256
num_epochs = 1
no_of_days = 10
patch_length = 2