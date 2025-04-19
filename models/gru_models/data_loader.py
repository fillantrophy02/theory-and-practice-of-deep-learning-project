import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from config_custom.config_gru import CONFIG
from torch.utils.data import WeightedRandomSampler

seq_length = CONFIG['sequence_length']

class DataframeLoader():
    def __init__(self, split):
        self.df = pd.read_csv(f"data/processed-data/{split}.csv")
        self.df = self.df.sort_values(by=['Location', 'Date'])

        # Include RainToday, exclude label and metadata
        self.feature_cols = [
            col for col in self.df.columns
            if col not in ['Date', 'Location', 'RainTomorrow']
        ]

        # Drop rows with NaNs in selected columns
        self.df = self.df.dropna(subset=self.feature_cols + ['RainTomorrow'])

    def split_df_into_sequences_with_labels(self):
        all_x, all_y = [], []

        for _, group in self.df.groupby('Location'):
            features = group[self.feature_cols].to_numpy()
            labels = group['RainTomorrow'].to_numpy().reshape(-1, 1)

            total_len = seq_length + 1  # we need 1 label after each sequence
            if len(features) < total_len:
                continue

            for i in range(len(features) - seq_length):
                x = features[i : i + seq_length]
                y = labels[i + seq_length]  # single label
                all_x.append(x)
                all_y.append(y)

        print(" Using features:", self.feature_cols)
        print(" Final x shape:", np.array(all_x).shape)
        return np.array(all_x), np.array(all_y)
    
    def split_df_into_seq2seq(self, future_steps=5):
        all_x, all_y = [], []

        for _, group in self.df.groupby('Location'):
            features = group[self.feature_cols].to_numpy()
            labels = group['RainTomorrow'].to_numpy()

            total_len = CONFIG['sequence_length'] + future_steps
            if len(features) < total_len:
                continue

            for i in range(len(features) - CONFIG['sequence_length'] - future_steps + 1):
                x_seq = features[i:i + CONFIG['sequence_length']]
                y_seq = labels[i + CONFIG['sequence_length']:i + CONFIG['sequence_length'] + future_steps]
                all_x.append(x_seq)
                all_y.append(y_seq)

        print(" Using features:", self.feature_cols)
        print(" Final x shape:", np.array(all_x).shape)
        return np.array(all_x), np.array(all_y)

class RainDataset(Dataset):
    def __init__(self, x, y, normalize=True, scaler=None):
        self.x = x
        self.y = y
        self.scaler = scaler

        if normalize:
            self._normalize()

    def _normalize(self):
        n, s, f = self.x.shape
        reshaped = self.x.reshape(-1, f)

        if self.scaler:
            scaled = self.scaler.transform(reshaped)
        else:
            self.scaler = MinMaxScaler()
            scaled = self.scaler.fit_transform(reshaped)

        self.x = scaled.reshape(n, s, f)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )



def get_balanced_sampler(y):
    """Returns a WeightedRandomSampler for binary label imbalance."""
    y_flat = y.flatten()
    class_counts = np.bincount(y_flat)
    weights = 1. / class_counts
    sample_weights = weights[y_flat]

    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float32)
    return WeightedRandomSampler(sample_weights_tensor, len(sample_weights_tensor))