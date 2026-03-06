import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pretty_midi

def generate_music(model, seed_sequence, num_to_generate=100):
    model.eval()
    generated_notes = []
    
    # Start with a real sequence from your data
    current_sequence = torch.tensor(seed_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(num_to_generate):
            p_logits, s_pred, d_pred = model(current_sequence)
            
            # 1. Get Pitch (highest probability)
            pitch = torch.argmax(p_logits, dim=1).item() / 127.0
            # 2. Get Step and Duration
            step = s_pred.item()
            dur = d_pred.item()
            
            # Store the new note
            new_note = [pitch, step, dur]
            generated_notes.append(new_note)
            
            # Update sequence: Remove oldest note, add the new one
            new_note_tensor = torch.tensor(new_note).reshape(1, 1, 3).to(device)
            current_sequence = torch.cat((current_sequence[:, 1:, :], new_note_tensor), dim=1)
            
    return generated_notes



def notes_to_midi(notes, out_file='output.mid'):
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0) # 0 is Grand Piano
    
    current_time = 0
    for pitch, step, dur in notes:
        # Denormalize pitch
        p = int(pitch * 127)
        # Ensure values are positive
        start = current_time + max(0, step)
        end = start + max(0.01, dur)
        
        note = pretty_midi.Note(velocity=100, pitch=p, start=start, end=end)
        piano.notes.append(note)
        current_time = start # Move time forward
        
    pm.instruments.append(piano)
    pm.write(out_file)