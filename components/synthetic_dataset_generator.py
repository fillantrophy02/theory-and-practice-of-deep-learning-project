
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from config import *

class SyntheticDataset(Dataset):
    def __init__(self, split: str = "train"):
        size = 43948 if split == "train" else 41809
        self.split = split
        self.x = np.random.rand(size, no_of_days, num_features)
        self.y = np.random.binomial(n=1, p=0.5, size=(size, 1))
        
    def __len__(self):
        return len(self.y)
    
    def report(self):
        num_label_0 = (self.y == 0).sum()
        num_label_1 = (self.y == 1).sum()
        print(f"\n------ Stats for {self.split} --------")
        print(f"Number of samples with label 0: {num_label_0}")
        print(f"Number of samples with label 1: {num_label_1}")
            
    def __getitem__(self, index):
        xs = torch.tensor(self.x[index], dtype=torch.float32)
        y = torch.tensor(self.y[index], dtype=torch.float32)

        return xs, y