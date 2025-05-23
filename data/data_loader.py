from enum import Enum
import os, sys

import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import torch

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from config_custom.config_lstm import CONFIG

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

    def __init__(self, split: str = "train", activate_undersampling: bool = True, 
                 enable_normalization: bool = True, scaler: MinMaxScaler = None,
                 seq_length: int = 7, target_seq_length: int = 1):
        self.split_sleep_ds = SplitDataset()
        self.split = split
        self.scaler = scaler
        self.seq_length = seq_length
        self.target_seq_length = target_seq_length
        
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
        target_col = 'RainTomorrow'
        
        if target_col in feature_cols:
            feature_cols.remove(target_col)
            
        num_features = len(feature_cols)
        sequences = []
        targets = []

        for _, group in df.groupby('Location'):
            group = group.set_index('Date')
            features = group[feature_cols].values
            target_values = group[target_col].values
            
            if len(features) >= self.seq_length:
                # Pre-allocate arrays for this location
                num_sequences = len(features) - self.seq_length - self.target_seq_length + 1
                if num_sequences > 0:
                    loc_sequences = np.zeros((num_sequences, self.seq_length, num_features))
                    loc_targets = np.zeros((num_sequences, self.target_seq_length, 1))
                    
                    for i in range(num_sequences):
                        loc_sequences[i] = features[i:i+self.seq_length]
                        loc_targets[i, 0, 0] = target_values[i+self.seq_length-1]
                    
                    sequences.append(loc_sequences)
                    targets.append(loc_targets)
        
        if sequences:
            all_x = np.vstack(sequences)
            all_y = np.vstack(targets)
            return all_x, all_y
        else:
            return np.empty((0, self.seq_length, num_features)), np.empty((0, self.target_seq_length, 1))
        
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
    
    def show_examples(self, num_examples=3):

        for i in range(min(num_examples, len(self.x))):
            print(f"\n======= Example {i+1} =======")
            print(f"Input sequence shape: {self.x[i].shape}")
            
            # Show features for each time step
            for t in range(self.seq_length):
                print(f"  Time t{t}: {self.x[i, t, :5]}... (showing first 5 features)")
            
            # Show target
            print(f"Target (RainTomorrow at t{self.seq_length-1}): {self.y[i, 0, 0]}")
    
train_ds = RainDataset("train", activate_undersampling=True, scaler=None, 
                       seq_length=CONFIG["seq_length"], target_seq_length=CONFIG["target_seq_length"])
test_ds = RainDataset("test", scaler=train_ds.scaler,
                      seq_length=CONFIG["seq_length"], target_seq_length=CONFIG["target_seq_length"])

train_dataloader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
test_dataloader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False)  # Usually no shuffle for test

# sample_train_features_batch, sample_train_labels_batch = next(iter(train_dataloader))
# feature_batch_size = sample_train_features_batch.size()
# label_batch_size = sample_train_labels_batch.size()
# print(f"Feature batch shape: {feature_batch_size}") # (batch_size, seq_length, num_features)
# print(f"Labels batch shape: {label_batch_size}") # (batch_size, target_seq_length, 1)

# train_ds.report()

