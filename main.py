from config import model, use_existing_weights
from models.lstm import run_lstm
from models.transformer import run_transformer
from models.gru import run_gru

if model == "LSTM":
    run_lstm(use_existing_weights)
elif model == "GRU":
    run_gru(use_existing_weights)
elif model == "Transformer":
    run_transformer(use_existing_weights)
