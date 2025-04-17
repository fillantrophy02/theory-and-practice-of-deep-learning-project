from config import *
from models.lstm import run_lstm

if model == "LSTM":
    run_lstm()
elif model == "GRU":
    pass
elif model == "Transformer":
    pass