import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from data.data_processing_rachel import DataProcessingPipeline
from models.lstm_onestep import LSTM_OneStep
from models.lstm_multistep import LSTM_MultiStep

from training.trainer import train_model
from training.trainer2 import train_model2

from training.evaluator import evaluate_model
from training.evaluator2 import evaluate_model2

from config.config import CONFIG

def main():
    
    # Step 1: Load and preprocess data
    df = pd.read_csv('data/processed-data/train.csv')
    data_processed = DataProcessingPipeline(df)

    # Step 3: Split before preparing tensors
    features = df.iloc[:, :-1]  # all columns except the last
    labels = df.iloc[:, -1]     # last column is the target
    X_train_df, X_test_df, Y_train_df, Y_test_df = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Step 4: Preparing tensors
    X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
    Y_train = torch.tensor(Y_train_df.values, dtype=torch.float32) #(batch_size, 1)
    Y_train = Y_train.unsqueeze(-1)

    X_test = torch.tensor(X_test_df.values, dtype=torch.float32)
    Y_test = torch.tensor(Y_test_df.values, dtype=torch.float32) #(batch_size, 1)
    Y_test = Y_test.unsqueeze(-1) #(batch_size, seq_len, 1)

    seq_len = CONFIG['seq_lens']
    X_train_seq, Y_train_seq = data_processed.create_sequences(X_train, Y_train, seq_len)
    X_test_seq, Y_test_seq = data_processed.create_sequences(X_test, Y_test, seq_len) 

    # Step 5: Model Initialization 
    # model1 = LSTM_OneStep(input_size=CONFIG['input_size'], hidden_size=CONFIG['hidden_size'], output_size=CONFIG['output_size'])
    # train_model(model1, X_train, Y_train, num_epochs=CONFIG['num_epochs'], batch_size=CONFIG['batch_size'])
    # evaluate_model(model1, X_test, Y_test, num_epochs=CONFIG['num_epochs'], batch_size=CONFIG['batch_size'])

    model2 = LSTM_MultiStep(input_size=CONFIG['input_size'], hidden_size=CONFIG['hidden_size'], output_size=CONFIG['output_size'])
    #train_model2(model2, X_train_seq, Y_train_seq, num_epochs=CONFIG['num_epochs'], batch_size=CONFIG['batch_size'])
    model2.load_state_dict(torch.load("model2_weights.pth", weights_only=True))
    evaluate_model2(model2, X_test_seq, Y_test_seq)

if __name__ == "__main__":
    main()

