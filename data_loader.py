from enum import Enum
import os

import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from config import *
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class SplitSleepDataset():
    def __init__(self):
        df = self._load_df()
        self.df = self._interpolate_missing_dates(df)

    def visualize_interpolated_data(self):
        patient_id = data_visualization.patient_id
        patient_email = f'nia+{patient_id}@rowan.kr'
        group = self.df[self.df['EMAIL'] == patient_email]
        output_fp = f'visualizations/time_series_patient_{patient_id}_interpolated.png'
        plot_and_save_time_series(group, output_fp)

    def _interpolate_missing_dates(self, df):
        """Fills missing dates per user and applies interpolation."""
        df = df.sort_values(by=['EMAIL', 'date'])
        feature_cols = [col for col in df.columns if col not in ['EMAIL', 'date', 'label']]

        interpolated_dfs = []
        for _, group in df.groupby('EMAIL'):
            group = group.set_index('date')  # Set date as index
            full_date_range = pd.date_range(start=group.index.min(), end=group.index.max(), freq='D')

            # Reindex to fill missing dates, then interpolate
            group = group.reindex(full_date_range)
            group[feature_cols] = group[feature_cols].interpolate(method='slinear', limit_direction='both')
            group['label'] = group['label'].ffill().bfill()  # Fill label using forward and backward fill

            group = group.reset_index().rename(columns={'index': 'date'})
            group['EMAIL'] = _  # Re-add EMAIL column

            interpolated_dfs.append(group)

        return pd.concat(interpolated_dfs, ignore_index=True)

    def _load_df(self):
        fp = "data/processed-data/dataset.csv"
        df = pd.read_csv(fp)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def get_test_df(self) -> pd.DataFrame:
        '''Extracted from isolating the final week of data from each subject'''
        latest_dates = self.df.groupby('EMAIL')['date'].max()
        cutoff_dates = latest_dates - pd.Timedelta(days=7)
        df_test = self.df[self.df.apply(lambda row: row['date'] > cutoff_dates[row['EMAIL']], axis=1)]
        return df_test
    
    def get_train_df(self) -> pd.DataFrame:
        latest_dates = self.df.groupby('EMAIL')['date'].max()
        cutoff_dates = latest_dates - pd.Timedelta(days=7)
        df_train = self.df[self.df.apply(lambda row: row['date'] <= cutoff_dates[row['EMAIL']], axis=1)]
        return df_train


class SleepDataset(Dataset):
    def __init__(self, split: str = "train", no_of_days: int = 5, activate_undersampling: bool = True, enable_normalization: bool = True, scaler: MinMaxScaler = None):
        self.split_sleep_ds = SplitSleepDataset()
        self.split_sleep_ds.visualize_interpolated_data()

        self.seq_length = no_of_days
        self.split = split
        self.scaler = scaler
        
        df = self._load_df_split(split)
        self.x, self.y = self._split_df_into_sequences(df)

        if enable_normalization:
            self._standardize()
            #self._normalize()

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
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by=['EMAIL', 'date'])
        feature_cols = [col for col in df.columns if col not in ['EMAIL', 'date', 'label']]        
        num_features = len(feature_cols)
        all_x = np.empty((0, self.seq_length, num_features))
        all_y = np.empty((0, 1))

        for _, group in df.groupby('EMAIL'):
            group = group.set_index('date')  # Set date as index
            values = group[feature_cols + ['label']].to_numpy()

            if len(values) >= self.seq_length:
                x_data = np.lib.stride_tricks.sliding_window_view(values[:, :-1], self.seq_length, axis=0)  # (N, seq_length, num_features)
                x_data = np.transpose(x_data, (0, 2, 1))  # (N, num_features, seq_length)
                y_data = np.expand_dims(values[self.seq_length - 1:, -1], axis=1)

                all_x = np.concatenate((all_x, x_data), axis=0)
                all_y = np.concatenate((all_y, y_data), axis=0)

        return all_x, all_y
    
    def _undersample(self):
        """Reduce majority class (label 0) to match the count of the minority class (label 1)."""
        indices_label_0 = np.where(self.y == 0)[0]
        indices_label_1 = np.where(self.y == 1)[0]

        num_label_1 = len(indices_label_1)
        sampled_indices_0 = np.random.choice(indices_label_0, size=num_label_1, replace=False)
        balanced_indices = np.concatenate([sampled_indices_0, indices_label_1])
        
        self.x = self.x[balanced_indices]
        self.y = self.y[balanced_indices]

    
    def report(self):
        num_label_0 = (self.y == 0).sum()
        num_label_1 = (self.y == 1).sum()
        print(f"\n------ Stats for {self.split} --------")
        print(f"Number of samples with label 0: {num_label_0}")
        print(f"Number of samples with label 1: {num_label_1}")
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        xs = torch.tensor(self.x[index], dtype=torch.float32)
        y = torch.tensor(self.y[index], dtype=torch.float32)

        return xs, y
    
train_ds = SleepDataset("train", no_of_days=5, activate_undersampling=True, scaler=None)
test_ds = SleepDataset("test", no_of_days=5, scaler=train_ds.scaler)
print(f"Train: {len(train_ds)} samples, Test: {len(test_ds)} samples.")

torch.manual_seed(24)
train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

sample_train_features_batch, sample_train_labels_batch = next(iter(train_dataloader))
feature_batch_size = sample_train_features_batch.size()
label_batch_size = sample_train_labels_batch.size()
print(f"Feature batch shape: {feature_batch_size}") # (32, 5, 32)
print(f"Labels batch shape: {label_batch_size}") # (32, 1)

input_size = feature_batch_size[-1] # aka no. of features

# train_ds.report()