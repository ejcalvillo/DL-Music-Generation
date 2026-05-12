import os
import numpy as np
import pandas as pd
import pretty_midi
import torch
from torch.utils.data import Dataset, DataLoader


MAESTRO_DIR = os.path.join(os.path.dirname(__file__), '..', 'maestro')

# Data processing and loading
def process_maestro_subset(base_dir, year=None, num_files=None, seq_length=50):
    csv_path = os.path.join(base_dir, 'maestro-v3.0.0.csv')
    metadata = pd.read_csv(csv_path)

    if year is not None:
        metadata = metadata[metadata['year'] == year].reset_index(drop=True)
    if num_files is not None:
        metadata = metadata.iloc[:min(num_files, len(metadata))]

    all_sequences, all_targets = [], []

    for i in range(len(metadata)):
        file_path = os.path.join(base_dir, metadata.iloc[i]['midi_filename'])
        print(f"Processing: {file_path}")

        pm    = pretty_midi.PrettyMIDI(file_path)
        notes = sorted(pm.instruments[0].notes, key=lambda n: n.start)

        prev, note_data = notes[0].start, []
        for n in notes:
            note_data.append([
                n.pitch / 127.0,
                np.log1p(n.start - prev),
                np.log1p(n.end - n.start),
            ])
            prev = n.start

        note_data = np.array(note_data)
        for j in range(len(note_data) - seq_length):
            all_sequences.append(note_data[j : j + seq_length])
            all_targets.append(note_data[j + seq_length])

    return np.array(all_sequences), np.array(all_targets)

# Dataset and DataLoader
class MaestroDataset(Dataset):
    def __init__(self, X, y):
        self.X       = torch.tensor(X, dtype=torch.float32)
        self.y_pitch = torch.tensor(y[:, 0] * 127, dtype=torch.long)
        self.y_step  = torch.tensor(y[:, 1], dtype=torch.float32).unsqueeze(1)
        self.y_dur   = torch.tensor(y[:, 2], dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_pitch[idx], self.y_step[idx], self.y_dur[idx]


def load_data(year=None, num_files=None, seq_length=50, batch_size=128, val_split=0.1):
    X, y = process_maestro_subset(MAESTRO_DIR, year=year, num_files=num_files, seq_length=seq_length)

    split = int(len(X) * (1 - val_split))
    X_train, y_train = X[:split], y[:split]
    X_val,   y_val   = X[split:], y[split:]

    def make_loader(Xd, yd, shuffle):
        return DataLoader(MaestroDataset(Xd, yd), batch_size=batch_size,
                          shuffle=shuffle, num_workers=0, pin_memory=False)

    return make_loader(X_train, y_train, shuffle=True), make_loader(X_val, y_val, shuffle=False)
