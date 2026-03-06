import sys
import os

# Get the path of the directory one level up
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add it to the system path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# NOW you can import
from data_import import train_loader
import torch
from lstm import MusicLSTM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MusicLSTM().to(device)

# Check if a trained model already exists
script_dir = os.path.dirname(__file__)
models_dir = os.path.join(script_dir, 'models')
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, 'music_lstm_v1.pth')

if os.path.exists(model_path):
    print(f"Loading existing model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("\n\nResuming training from saved checkpoint.")
else:
    print("No existing model found. Starting training from scratch.")

# Loss functions
criterion_pitch = torch.nn.CrossEntropyLoss()
criterion_mse = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, b_pitch, b_step, b_dur in train_loader:
            # Move to GPU/CPU
            batch_x, b_pitch = batch_x.to(device), b_pitch.to(device)
            b_step, b_dur = b_step.to(device), b_dur.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            p_logits, s_pred, d_pred = model(batch_x)
            
            # Calculate combined loss
            loss_p = criterion_pitch(p_logits, b_pitch)
            loss_s = criterion_mse(s_pred, b_step)
            loss_d = criterion_mse(d_pred, b_dur)
            
            # Sum the losses (you can weight these if one is too high)
            loss = loss_p + loss_s + loss_d
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(train_loader):.4f}")

train(epochs=5)

# %%
# Save model to the models directory
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")