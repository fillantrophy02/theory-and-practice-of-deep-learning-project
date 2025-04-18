import pandas as pd
import torch
import torch

class DataProcessingPipeline():
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def clean(self, type):
        #self._drop_rows_with_na_labels()
        self._rain_today_na(type) #Train
        self._drop_rows_with_na_labels(type)
    def clean(self, type):
        #self._drop_rows_with_na_labels()
        self._rain_today_na(type) #Train
        self._drop_rows_with_na_labels(type)
        self._transform_categorical_features_to_numerical()
        self._extract_time_series_feature_for_city()
        self._transform_into_multi_index()
        self._drop_columns_with_too_many_missing_values()
        # self._drop_unnecessary_columns()
        # self._drop_unnecessary_columns()
        self._interpolate_missing_values()
        self._transform_label_to_binary()
        self._test_rain_tomorrow(type) #Test
        self._test_rain_tomorrow(type) #Test

    def get(self) -> pd.DataFrame:
        return self.df
    
    def export_to_csv(self, filepath):
        self.df.to_csv(filepath)

    def report(self):
        num_columns = self.df.shape[1]
        num_rows = self.df.shape[0]
        if 'Location' in self.df.columns:
            unique_locations = self.df['Location'].nunique()
        else:
            unique_locations = self.df.index.get_level_values('Location').nunique()
        avg_rows_per_location = num_rows / unique_locations if unique_locations > 0 else 0
        num_empty_cells = self.df.isnull().sum().sum()
        
        report_str = (
            f"Number of columns: {num_columns}\n"
            f"Number of rows: {num_rows}\n"
            f"Number of missing values: {num_empty_cells}\n"
            f"Number of unique locations: {unique_locations}\n"
            f"Average number of rows per location: {avg_rows_per_location:.0f}\n"
        )
        
        print(report_str)
        # print(self.df.head())
        print(self.df.iloc[:20])

    def _transform_into_multi_index(self):
        self.df.sort_values(by=['row ID'], inplace=True)
        self.df.drop(columns=['row ID'], inplace=True)
        self.df.set_index(['Location', 'Date'], inplace=True)
        self.df.sort_index(inplace=True)

    def _drop_columns_with_too_many_missing_values(self, threshold=0.3):
        missing_fraction = self.df.isnull().mean() 
        columns_to_drop = missing_fraction[missing_fraction > threshold].index  
        self.df.drop(columns=columns_to_drop, inplace=True)

    def _drop_unnecessary_columns(self):
        columns_to_drop = ["RainTomorrow"]
        for col in columns_to_drop:
            if col in self.df.columns:
                self.df.drop(columns=columns_to_drop, inplace=True)

    def _drop_rows_with_na_labels(self, type):
        if type == 'Test':
            self.df.dropna(subset=['RainToday'], inplace=True)
        else:
            pass
    def _drop_rows_with_na_labels(self, type):
        if type == 'Test':
            self.df.dropna(subset=['RainToday'], inplace=True)
        else:
            pass

    def _interpolate_missing_values(self):
        self.df.interpolate(inplace=True, limit_direction='both')

    def _extract_time_series_feature_for_city(self):
        self.df['Date'] = self.df.groupby('Location').cumcount() 
        first_two_columns = ['Location', 'Date']    
        self.df = self.df[first_two_columns + [col for col in self.df.columns if col not in first_two_columns]]

    def _transform_categorical_features_to_numerical(self):

        if 'RainToday' in self.df.columns:
            self.df['RainToday'] = self.df['RainToday'].map(lambda x: 1 if x == 'Yes' else (0 if x == 'No' else x))

        cat_columns = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']

        if 'RainToday' in self.df.columns:
            self.df['RainToday'] = self.df['RainToday'].map(lambda x: 1 if x == 'Yes' else (0 if x == 'No' else x))

        cat_columns = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
        existing_columns = [col for col in cat_columns if col in self.df.columns]

        for col in existing_columns:
            self.df[col] = self.df[col].astype('category').cat.codes

        for col in existing_columns:
            self.df[col] = self.df[col].astype('category').cat.codes


    def _transform_label_to_binary(self):
        pass

    def _rain_today_na(self, type):
        if type == "Train":
            for i in range(1, len(self.df)):
                if pd.isna(self.df.loc[i, "RainToday"]):
                    self.df.loc[i, "RainToday"] = self.df.loc[i - 1, "RainTomorrow"]
        else:
            pass

    def prepare_tensor_data(self):

        features_columns = self.df.columns[:-1]
        target_column = self.df.columns[-1]

        X = self.df[features_columns].values
        Y = self.df[target_column].values

        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)

        return X_tensor, Y_tensor
    
    def create_sequences(self, X, Y, seq_len):

        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        X_seq = []
        Y_seq = []
        
        for i in range(n_samples - seq_len):
            X_seq.append(X[i:i+seq_len])
            Y_seq.append(Y[i:i+seq_len])
        
        X_seq = torch.stack(X_seq)
        Y_seq = torch.stack(Y_seq)
        
        return X_seq, Y_seq
    
    def na_values(self):
        na_columns = self.df.columns[self.df.isna().any()].tolist()
        
        if na_columns:
            print("Columns with NA values:")
            for col in na_columns:
                na_count = self.df[col].isna().sum()
                print(f"- {col}: {na_count} missing")
        else:
            print("No columns have NA vsalues.")

    def _test_rain_tomorrow(self, type):
        # This assumes that in the last data, it will not rain the next day!
        if type == "Test":
            self.df['RainTomorrow'] = self.df['RainToday'].shift(-1).fillna(0).astype(int)
        else:
            pass

if __name__ == '__main__':
    type = "Train"
    df = pd.read_csv('data/raw-data/train.csv')
    pipeline = DataProcessingPipeline(df)
    pipeline.report()
    pipeline.clean(type)
    print("\nAfter cleaning ----------------------------------")
    pipeline.report()
    pipeline.export_to_csv('data/processed-data/train.csv')

    type = "Test"
    df = pd.read_csv('data/raw-data/test.csv')
    pipeline = DataProcessingPipeline(df)
    pipeline.report()
    pipeline.clean(type)
    print("\nAfter cleaning ----------------------------------")
    pipeline.report()
    pipeline.export_to_csv('data/processed-data/test.csv')