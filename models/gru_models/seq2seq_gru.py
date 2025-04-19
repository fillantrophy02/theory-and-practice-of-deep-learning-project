from models.gru_models.encoder_gru_seq2seq import EncoderGRU
from models.gru_models.decoder_gru_seq2seq import DecoderGRU
import torch.nn as nn

class Seq2SeqGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.encoder = EncoderGRU(input_size, hidden_size, num_layers, dropout)
        self.decoder = DecoderGRU(input_size, hidden_size, output_size, num_layers, dropout)

    def forward(self, input_seq, target_seq_length):
        encoder_hidden = self.encoder(input_seq)
        decoder_input = input_seq[:, -1].unsqueeze(1).repeat(1, target_seq_length, 1)
        decoder_output, _ = self.decoder(decoder_input, encoder_hidden)
        return decoder_output
