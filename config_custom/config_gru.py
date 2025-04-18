# Configurations for model training
CONFIG = {
    # Shared
    'input_size': 19,
    'hidden_size': 64,
    'output_size': 1,
    'batch_size': 32,
    'num_epochs': 20,
    'learning_rate': 0.001,

    # Seq2Seq only
    'sequence_length': 10,
    'future_steps': 5,
    'num_layers': 2,
    'dropout': 0.1
}
