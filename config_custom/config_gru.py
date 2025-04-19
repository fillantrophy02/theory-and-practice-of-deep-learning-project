# Configurations for model training
CONFIG = {
    # Shared
    'input_size': 20,
    'hidden_size': 128,
    'output_size': 1,
    'batch_size': 128,
    'num_epochs': 50,
    'learning_rate': 0.001,

    # Seq2Seq only
    'sequence_length': 10,
    'future_steps': 5,
    'num_layers': 1,
    'dropout': 0.2
}
