# Configurations for model training
CONFIG = {
    'batch_size': 64,

    'learning_rate': 0.001,
    'num_epochs': 50,
    'hidden_size': 64,
    'output_size': 1,
    'input_size': 20, #Number of features minus the target

    'seq_length': 21,
    'target_seq_length': 1,
    
        # MODEL CHOICES: 
        # 1: Normal LSTM
        # 2: Modulation Gate LSTM
        # 3: Dual Memory LSTM
    'model': 1
}
