from enum import Enum
import os

import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import torch
from config_custom.config_transformer import *
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split


class DataframeLoader():
    def __init__(self, split):
        self.split = split
        if self.split == "train_and_val":
            self.df = self._load_df('train')
        elif self.split == "test":
            self.df = self._load_df('test')

    def _load_df(self, split):
        fp = f"data/processed-data/{split}.csv"
        df = pd.read_csv(fp)
        return df
    
    def split_df_into_sequences_with_labels(self) -> tuple:
        self.df = self.df.sort_values(by=['Location', 'Date'])
        feature_cols = [col for col in self.df.columns if col not in (['Date', 'RainTomorrow'] + excluded_features)]        
        num_features = len(feature_cols)
        all_x = np.empty((0, seq_length, num_features))
        all_y = np.empty((0, target_seq_length, 1))

        for _, group in self.df.groupby('Location'):
            group = group.set_index('Date')  # Set date as index
            values = group[feature_cols + ['RainTomorrow']].to_numpy()

            if len(values) >= (seq_length):
                all_data = np.lib.stride_tricks.sliding_window_view(values, seq_length, axis=0)  # (N, num_features, seq_length+target_seq-1)
                all_data = np.transpose(all_data, (0, 2, 1)) # (N, seq_length+target_seq-1, num_features)
                x_data = all_data[:, 0:seq_length, :-1] # (N, seq_length, num_features)
                y_data = all_data[:, seq_length-1:, -1:] # (N, target_seq_length, 1)

                all_x = np.concatenate((all_x, x_data), axis=0)
                all_y = np.concatenate((all_y, y_data), axis=0)

        return all_x, all_y
    
    def split_df_into_sequences_without_labels(self) -> pd.DataFrame:
        self.df = self.df.sort_values(by=['Location', 'Date'])
        feature_cols = [col for col in self.df.columns if col not in ['Date', 'RainTomorrow', 'RainToday']]        
        num_features = len(feature_cols)
        all_x = np.empty((0, seq_length, num_features))

        for _, group in self.df.groupby('Location'):
            group = group.set_index('Date')  # Set date as index
            values = group[feature_cols].to_numpy()

            if len(values) >= (seq_length):
                all_data = np.lib.stride_tricks.sliding_window_view(values, seq_length, axis=0)  # (N, num_features, seq_length+target_seq-1)
                all_data = np.transpose(all_data, (0, 2, 1)) # (N, seq_length+target_seq-1, num_features)
                x_data = all_data[:, 0:seq_length, :] # (N, seq_length, num_features)
                all_x = np.concatenate((all_x, x_data), axis=0)

        return all_x
    
class RainDataset(Dataset):
    def __init__(self, x, y = None, split = "train", activate_undersampling: bool = True, enable_normalization: bool = True, scaler: MinMaxScaler = None):
        self.split = split
        self.scaler = scaler
        self.x = x
        self.y = y

        if enable_normalization:
            #self._standardize()
            self._normalize()

        if activate_undersampling:
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

    def _undersample(self):
        """Reduce majority class (label 0) to match the count of the minority class (label 1)."""
        indices_label_0 = np.where(self.y[:, 0, :] == 0)[0]
        indices_label_1 = np.where(self.y[:, 0, :] == 1)[0]

        num_label_1 = len(indices_label_1)
        sampled_indices_0 = np.random.choice(indices_label_0, size=num_label_1, replace=False)
        balanced_indices = np.concatenate([sampled_indices_0, indices_label_1])
        
        self.x = self.x[balanced_indices]
        self.y = self.y[balanced_indices]

    def get_samples_weight(self):
        y_flat = self.y.flatten().astype(int)
        num_label_0 = (y_flat == 0).sum() # only consider 'today' for label
        num_label_1 = (y_flat == 1).sum()
        class_sample_count = np.array([num_label_0, num_label_1])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in y_flat])
        samples_weight = torch.from_numpy(samples_weight)
        return samples_weight

    def report(self):
        num_label_0 = (self.y[:, 0, :] == 0).sum() # only consider 'today' for label
        num_label_1 = (self.y[:, 0, :] == 1).sum()
        print(f"\n------ Stats for {self.split} --------")
        print(f"Number of samples with label 0: {num_label_0}")
        print(f"Number of samples with label 1: {num_label_1}")

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        xs = torch.tensor(self.x[index], dtype=torch.float32)

        if self.y is None:
            return xs
        
        y = torch.tensor(self.y[index], dtype=torch.long)
        return xs, y


# Train and val
x, y = DataframeLoader("train_and_val").split_df_into_sequences_with_labels()
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_split, random_state=42, shuffle=True)
train_ds = RainDataset(x_train, y_train, "train", activate_undersampling=False, scaler=None)
val_ds = RainDataset(x_val, y_val, "val", activate_undersampling=False, scaler=train_ds.scaler)
print(f"Train: {len(train_ds)} samples")
print(f"Val: {len(val_ds)} samples")

# Test (no labels)
# x = DataframeLoader("test").split_df_into_sequences_without_labels()
# test_ds = RainDataset(x, split = "test", activate_undersampling=False, scaler=train_ds.scaler)
# print(f"Test: {len(test_ds)} samples.")

# Ensure each batch has balanced representation of classes
torch.manual_seed(24)
samples_weight = train_ds.get_samples_weight()
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

train_dataloader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
# test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

sample_train_features_batch, sample_train_labels_batch = next(iter(train_dataloader))
feature_batch_size = sample_train_features_batch.size()
label_batch_size = sample_train_labels_batch.size()
print(f"Feature batch shape: {feature_batch_size}") # (batch_size, seq_length, num_features)
print(f"Labels batch shape: {label_batch_size}") # (batch_size, target_seq_length, 1)

train_ds.report()
val_ds.report()