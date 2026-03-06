import pandas as pd
import numpy as np
import pretty_midi
import torch
from torch.utils.data import Dataset, DataLoader
import os


# Get the correct path to maestro directory relative to this file
MAESTRO_DIR = os.path.join(os.path.dirname(__file__), '..', 'maestro')


def process_maestro_subset(base_dir, num_files=10, seq_length=50):
    # 1. Read the CSV map
    csv_path = os.path.join(base_dir, 'maestro-v3.0.0.csv')
    metadata = pd.read_csv(csv_path)
    
    all_sequences = []
    all_targets = []
    
    # 2. Loop through a small subset of files
    for i in range(num_files):
        file_path = os.path.join(base_dir, metadata.iloc[i]['midi_filename'])
        print(f"Processing: {file_path}")
        
        # Extract notes
        pm = pretty_midi.PrettyMIDI(file_path)
        instr = pm.instruments[0]
        notes = sorted(instr.notes, key=lambda x: x.start)
        
        # Convert to numeric features: [Pitch, Step, Duration]
        prev_start = notes[0].start
        note_data = []
        for n in notes:
            note_data.append([
                n.pitch / 127.0,          # Normalized Pitch
                n.start - prev_start,      # Step
                n.end - n.start            # Duration
            ])
            prev_start = n.start
        
        # 3. Create sliding windows
        note_data = np.array(note_data)
        for j in range(len(note_data) - seq_length):
            all_sequences.append(note_data[j : j + seq_length])
            all_targets.append(note_data[j + seq_length])
            
    return np.array(all_sequences), np.array(all_targets)

X, y = process_maestro_subset(MAESTRO_DIR)


class MaestroDataset(Dataset):
    def __init__(self, X, y):
        # Ensure data is float32 for PyTorch
        self.X = torch.tensor(X, dtype=torch.float32)
        # Pitch is used for CrossEntropy, so it must be Long (integer)
        self.y_pitch = torch.tensor(y[:, 0] * 127, dtype=torch.long) 
        self.y_step = torch.tensor(y[:, 1], dtype=torch.float32).unsqueeze(1)
        self.y_dur = torch.tensor(y[:, 2], dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_pitch[idx], self.y_step[idx], self.y_dur[idx]

# Initialize with your arrays
dataset = MaestroDataset(X, y)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
