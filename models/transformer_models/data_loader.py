import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from config_custom.config_gru import CONFIG

seq_length = CONFIG['sequence_length']
target_seq_length = CONFIG['future_steps']
val_split = 0.2
batch_size = CONFIG['batch_size']

class DataframeLoader():
    def __init__(self, split):
        self.df = self._load_df(split)

    def _load_df(self, split):
        return pd.read_csv(f"data/processed-data/{split}.csv")

    def split_df_into_sequences_with_labels(self):
        self.df = self.df.sort_values(by=['Location', 'Date'])
        feature_cols = [col for col in self.df.columns if col not in ['Date', 'RainTomorrow', 'RainToday', 'Location']]
        all_x, all_y = [], []

        for _, group in self.df.groupby('Location'):
            values = group[feature_cols + ['RainTomorrow']].to_numpy()

            if len(values) >= (seq_length + target_seq_length - 1):
                windows = np.lib.stride_tricks.sliding_window_view(values, seq_length + target_seq_length - 1, axis=0)
                for w in windows:
                    x = w[:seq_length, :-1]
                    y = w[seq_length-1:, -1:]
                    all_x.append(x)
                    all_y.append(y)

        return np.array(all_x), np.array(all_y)

class RainDataset(Dataset):
    def __init__(self, x, y=None, normalize=True, undersample=False, scaler=None, split="train"):
        self.x = x
        self.y = y
        self.split = split
        self.scaler = scaler

        if normalize:
            self._normalize()

        if undersample and y is not None:
            self._undersample()

    def _normalize(self):
        n, s, f = self.x.shape
        reshaped = self.x.reshape(-1, f)

        if self.scaler:
            scaled = self.scaler.transform(reshaped)
        else:
            self.scaler = MinMaxScaler()
            scaled = self.scaler.fit_transform(reshaped)

        self.x = scaled.reshape(n, s, f)

    def _undersample(self):
        idx_0 = np.where(self.y[:, 0, 0] == 0)[0]
        idx_1 = np.where(self.y[:, 0, 0] == 1)[0]
        sample_0 = np.random.choice(idx_0, size=len(idx_1), replace=False)
        balanced_idx = np.concatenate([sample_0, idx_1])
        self.x = self.x[balanced_idx]
        self.y = self.y[balanced_idx]

    def get_sample_weights(self):
        flat_y = self.y[:, 0, 0].astype(int)
        class_counts = np.bincount(flat_y)
        weights = 1.0 / class_counts
        return torch.tensor([weights[y] for y in flat_y], dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_tensor = torch.tensor(self.x[idx], dtype=torch.float32)
        if self.y is not None:
            y_tensor = torch.tensor(self.y[idx], dtype=torch.float32)
            return x_tensor, y_tensor
        return x_tensor
