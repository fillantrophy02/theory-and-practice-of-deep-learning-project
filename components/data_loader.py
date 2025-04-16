from enum import Enum
import os

import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import torch
from config import *
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class SplitDataset():
    def __init__(self):
        pass

    def _load_df(self, split):
        fp = f"data/processed-data/{split}.csv"
        df = pd.read_csv(fp)
        return df

    def get_test_df(self) -> pd.DataFrame:
        return self._load_df('test')
    
    def get_train_df(self) -> pd.DataFrame:
        return self._load_df('train')


class RainDataset(Dataset):
    def __init__(self, split: str = "train", activate_undersampling: bool = True, enable_normalization: bool = True, scaler: MinMaxScaler = None):
        self.split_sleep_ds = SplitDataset()
        self.split = split
        self.scaler = scaler
        
        df = self._load_df_split(split)
        self.x, self.y = self._split_df_into_sequences(df)

        if enable_normalization:
            #self._standardize()
            self._normalize()

        if split == 'train' and activate_undersampling:
            self._undersample()

    def _standardize(self):
        """Applies Z-score standardization to self.x while maintaining its original shape."""
        num_samples, seq_len, num_features = self.x.shape  # Extract dimensions
        reshaped_x = self.x.reshape(-1, num_features)  # Flatten sequence dimension

        if isinstance(self.scaler, StandardScaler) and hasattr(self.scaler, "mean_"):
            standardized_x = self.scaler.transform(reshaped_x)  # Transform test data
        else:
            self.scaler = StandardScaler()
            standardized_x = self.scaler.fit_transform(reshaped_x)  # Fit and transform training data

        self.x = standardized_x.reshape(num_samples, seq_len, num_features)  # Reshape back to original form

    def _normalize(self):
        # min-max scaling
        num_samples, seq_len, num_features = self.x.shape  # Get shape of feature matrix
        reshaped_x = self.x.reshape(-1, num_features)

        if self.scaler:
            scaled_x = self.scaler.transform(reshaped_x)  # Only transform test data
        else:
            self.scaler = MinMaxScaler()
            scaled_x = self.scaler.fit_transform(reshaped_x)  # Fit and transform on train data
        self.x = scaled_x.reshape(num_samples, seq_len, num_features)
        
    def _load_df_split(self, split) -> pd.DataFrame:
        if split == "train":
            return self.split_sleep_ds.get_train_df()
        elif split == "test":
            return self.split_sleep_ds.get_test_df()
        
    def _split_df_into_sequences(self, df) -> tuple:
        df = df.sort_values(by=['Location', 'Date'])
        feature_cols = [col for col in df.columns if col not in ['Date', 'RainToday']]        
        num_features = len(feature_cols)
        all_x = np.empty((0, seq_length, num_features))
        all_y = np.empty((0, target_seq_length, 1))

        for _, group in df.groupby('Location'):
            group = group.set_index('Date')  # Set date as index
            values = group[feature_cols + ['RainToday']].to_numpy()

            if len(values) >= (seq_length + target_seq_length):
                all_data = np.lib.stride_tricks.sliding_window_view(values, seq_length + target_seq_length, axis=0)  # (N, num_features, seq_length+target_seq-1)
                all_data = np.transpose(all_data, (0, 2, 1)) # (N, seq_length+target_seq-1, num_features)
                x_data = all_data[:, 0:seq_length, :-1] # (N, seq_length, num_features)
                y_data = all_data[:, seq_length:, -1:] # (N, target_seq_length, 1)

                all_x = np.concatenate((all_x, x_data), axis=0)
                all_y = np.concatenate((all_y, y_data), axis=0)

        return all_x, all_y
    
    def _undersample(self):
        """Reduce majority class (label 0) to match the count of the minority class (label 1)."""
        indices_label_0 = np.where(self.y[:, 0, :] == 0)[0]
        indices_label_1 = np.where(self.y[:, 0, :] == 1)[0]

        num_label_1 = len(indices_label_1)
        sampled_indices_0 = np.random.choice(indices_label_0, size=num_label_1, replace=False)
        balanced_indices = np.concatenate([sampled_indices_0, indices_label_1])
        
        self.x = self.x[balanced_indices]
        self.y = self.y[balanced_indices]

    
    def report(self):
        num_label_0 = (self.y[:, 0, :] == 0).sum() # only consider 'today' for label
        num_label_1 = (self.y[:, 0, :] == 1).sum()
        print(f"\n------ Stats for {self.split} --------")
        print(f"Number of samples with label 0: {num_label_0}")
        print(f"Number of samples with label 1: {num_label_1}")
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        xs = torch.tensor(self.x[index], dtype=torch.float32)
        y = torch.tensor(self.y[index], dtype=torch.float32)

        return xs, y
    
train_ds = RainDataset("train", activate_undersampling=True, scaler=None)
test_ds = RainDataset("test", scaler=train_ds.scaler)
print(f"Train: {len(train_ds)} samples, Test: {len(test_ds)} samples.")

torch.manual_seed(24)
train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

sample_train_features_batch, sample_train_labels_batch = next(iter(train_dataloader))
feature_batch_size = sample_train_features_batch.size()
label_batch_size = sample_train_labels_batch.size()
print(f"Feature batch shape: {feature_batch_size}") # (batch_size, seq_length, num_features)
print(f"Labels batch shape: {label_batch_size}") # (batch_size, target_seq_length, 1)

train_ds.report()