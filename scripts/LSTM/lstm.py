import torch.nn as nn


class MusicLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=256, num_layers=2):
        super(MusicLSTM, self).__init__()
        
        # The "Brain": Learns sequences
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # The "Heads": Each predicts one feature of the next note
        self.pitch_head = nn.Linear(hidden_size, 128) # 128 MIDI pitches
        self.step_head = nn.Linear(hidden_size, 1)    # Continuous time
        self.dur_head = nn.Linear(hidden_size, 1)     # Continuous duration

    def forward(self, x):
        # x shape: (batch, seq_len, 3)
        lstm_out, _ = self.lstm(x)
        
        # We only care about the very last note's output
        last_out = lstm_out[:, -1, :]
        
        pitch_logits = self.pitch_head(last_out)
        step_pred = self.step_head(last_out)
        dur_pred = self.dur_head(last_out)
        
        return pitch_logits, step_pred, dur_pred


