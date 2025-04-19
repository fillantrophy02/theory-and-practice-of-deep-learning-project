from models.gru_models.encoder_gru_seq2seq import EncoderGRU
from models.gru_models.decoder_gru_seq2seq import DecoderGRU
import torch.nn as nn
import torch
from config_custom.config_gru import CONFIG

class Seq2SeqGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.encoder = EncoderGRU(input_size, hidden_size, num_layers, dropout)
        self.decoder = DecoderGRU(input_size, hidden_size, output_size, num_layers, dropout)
        self.output_to_input = nn.Linear(output_size, input_size)  # 🔁 For auto-regressive feedback

    def forward(self, input_seq, target_seq=None, teacher_forcing_ratio=0.5):
        batch_size = input_seq.size(0)
        input_dim = input_seq.size(2)
        target_len = target_seq.size(1) if target_seq is not None else CONFIG['future_steps']

        encoder_hidden = self.encoder(input_seq)
        decoder_input = input_seq[:, -1].unsqueeze(1)  # [B, 1, input_size]
        hidden = encoder_hidden

        outputs = []

        for t in range(target_len):
            output, hidden = self.decoder(decoder_input, hidden)  # [B, 1, output_size]
            outputs.append(output)

            if self.training and target_seq is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target_seq[:, t].unsqueeze(1)  # [B, 1, input_size]
            else:
                projected_input = self.output_to_input(output)  # [B, 1, input_size]
                decoder_input = projected_input

        return torch.cat(outputs, dim=1)  # [B, T, output_size]
