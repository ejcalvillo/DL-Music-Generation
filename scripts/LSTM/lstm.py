import torch
import torch.nn as nn
import torch.nn.functional as F


class MusicLSTM(nn.Module):
    def __init__(self, pitch_embed_dim=16, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()

        # Pitch is categorical (128 MIDI values), not a continuous number.
        # An embedding lets the model learn which pitches are musically related
        # (octaves, harmonics) rather than treating MIDI 60 as "halfway to 127".
        self.pitch_embed = nn.Embedding(128, pitch_embed_dim)

        # LSTM input = learned pitch vector + raw step + raw duration
        lstm_input_size = pitch_embed_dim + 2

        self.lstm = nn.LSTM(
            lstm_input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # LayerNorm stabilizes the hidden state before the output heads,
        # preventing one head from being overwhelmed by large activations.
        self.norm    = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.pitch_head = nn.Linear(hidden_size, 128)
        self.step_head  = nn.Linear(hidden_size, 1)
        self.dur_head   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, 3)  —  [pitch_normalized, step, duration]
        pitch_int = (x[:, :, 0] * 127).long().clamp(0, 127)
        pitch_emb = self.pitch_embed(pitch_int)          # (batch, seq_len, embed_dim)
        x_in      = torch.cat([pitch_emb, x[:, :, 1:]], dim=-1)  # + step, dur

        lstm_out, _ = self.lstm(x_in)
        last_out    = self.dropout(self.norm(lstm_out[:, -1, :]))

        pitch_logits = self.pitch_head(last_out)
        step_pred    = F.softplus(self.step_head(last_out))
        dur_pred     = F.softplus(self.dur_head(last_out))

        return pitch_logits, step_pred, dur_pred
