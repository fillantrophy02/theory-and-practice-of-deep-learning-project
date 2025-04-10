import pandas as pd

from data.data_processing import DataProcessingPipeline
from models.lstm import LSTM

from training.trainer import train_model
from config.config import CONFIG

def main():
    
    # Step 1: Load and preprocess data
    df = pd.read_csv('data/processed-data/train.csv')
    data_processor = DataProcessingPipeline(df)
    X_train, Y_train = data_processor.prepare_tensor_data()

    # Step 2: Initialize model, criterion, and optimizer
    model = LSTM(input_size=CONFIG['input_size'], hidden_size=CONFIG['hidden_size'], output_size=CONFIG['output_size'])

    # Step 3: Train the model
    train_model(model, X_train, Y_train, num_epochs=CONFIG['num_epochs'], batch_size=CONFIG['batch_size'])

if __name__ == "__main__":
    main()

