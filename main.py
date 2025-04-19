from config import *
from models.lstm import run_lstm
from models.transformer import run_transformer

if model == "LSTM":
    run_lstm()
elif model == "GRU":
    pass
elif model == "Transformer":
    run_transformer()
