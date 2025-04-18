import csv
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

class DataProcessingPipeline():
    def __init__(self, df: pd.DataFrame, city_code_mappings=None):
        self.df = df
        if city_code_mappings is not None:
            self.city_code_mappings = city_code_mappings
        else:
            self.city_code_mappings = self._get_city_code_mappings()

    def clean(self):
        self._generate_rain_tomorrow_from_today()
        self._transform_cities_to_codes()
        self._transform_categorical_features_to_numerical()
        self._encode_wind_direction_sin_cos()
        self._extract_time_series_feature_for_city()
        self._transform_into_multi_index()
        self._drop_columns_with_too_many_missing_values()
        self._drop_unnecessary_columns()
        self._interpolate_missing_values()
        self._transform_label_to_binary()

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
        if 'row ID' in self.df.columns:
            self.df.sort_values(by=['row ID'], inplace=True)
            self.df.drop(columns=['row ID'], inplace=True)

        self.df.set_index(['Location', 'Date'], inplace=True)
        self.df.sort_index(inplace=True)

    def _drop_columns_with_too_many_missing_values(self, threshold=0.3):
        missing_fraction = self.df.isnull().mean() 
        columns_to_drop = missing_fraction[missing_fraction > threshold].index  
        self.df.drop(columns=columns_to_drop, inplace=True)

    def _drop_unnecessary_columns(self):
        columns_to_drop = []
        for col in columns_to_drop:
            if col in self.df.columns:
                self.df.drop(columns=columns_to_drop, inplace=True)

    def _drop_rows_with_na_labels(self):
        self.df.dropna(subset=['RainToday'], inplace=True)

    def _interpolate_missing_values(self):
        self.df.interpolate(inplace=True, limit_direction='both')

    def _extract_time_series_feature_for_city(self):
        self.df['Date'] = self.df.groupby('Location').cumcount() 
        first_two_columns = ['Location', 'Date']    
        self.df = self.df[first_two_columns + [col for col in self.df.columns if col not in first_two_columns]]

    def _get_city_code_mappings(self):
        with open('data/city_codes.csv') as f:
            reader = csv.reader(f)
            next(reader)
            return dict([[row[0], int(row[1])] for row in reader])
        
    def _transform_cities_to_codes(self):
        self.df['Location'] = self.df['Location'].map(lambda x: self.city_code_mappings[x])

    def _transform_categorical_features_to_numerical(self):
        cat_columns = ['RainToday', 'RainTomorrow']
        existing_columns = [col for col in cat_columns if col in self.df.columns]

        if existing_columns:
            for col in existing_columns:
                self.df[col] = self.df[col].astype('category').cat.codes
                self.df[col] = self.df[col].replace(-1, self.df[col].max() + 1)  # Replace -1 with the next available number

        
    def _encode_wind_direction_sin_cos(self):
        direction_map = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }

        wind_columns = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
        
        for col in wind_columns:
            if col in self.df.columns:
                radians = self.df[col].map(direction_map).apply(np.deg2rad)
                self.df[f'{col}_sin'] = np.sin(radians)
                self.df[f'{col}_cos'] = np.cos(radians)
                self.df.drop(columns=[col], inplace=True)

    def _transform_label_to_binary(self):
        pass

    def _generate_rain_tomorrow_from_today(self):
        if 'RainToday' in self.df.columns and 'RainTomorrow' not in self.df.columns:
            self.df['RainTomorrow'] = (
                self.df.groupby('Location')['RainToday'].shift(-1)
            )



    

df = pd.read_csv('data/raw-data/train.csv')
pipeline = DataProcessingPipeline(df)
pipeline.report()
pipeline.clean()
print("\nAfter cleaning ----------------------------------")
pipeline.report()
pipeline.export_to_csv('data/processed-data/train_pro.csv')

df = pd.read_csv('data/raw-data/test.csv')
pipeline = DataProcessingPipeline(df)
pipeline.report()
pipeline.clean()
print("\nAfter cleaning ----------------------------------")
pipeline.report()
pipeline.export_to_csv('data/processed-data/test_pro.csv')
