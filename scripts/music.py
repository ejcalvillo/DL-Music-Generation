import math
import os
import random
import torch
import torch.nn.functional as F
import pretty_midi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_music(model, seed_sequence, num_to_generate=100, temperature=1.0, min_step=0.0, min_dur=0.0, timing_sigma=0.15):
    """
    Generate notes autoregressively from a seed sequence.

    temperature : pitch randomness. < 1.0 = repetitive, > 1.0 = experimental.
    min_step    : minimum seconds between note onsets. Raise to space notes out.
    min_dur     : minimum note duration in seconds.
    """
    model.eval()
    generated_notes = []

    current_sequence = torch.tensor(seed_sequence, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(num_to_generate):
            p_logits, s_pred, d_pred = model(current_sequence)

            scaled_logits = p_logits / temperature
            probs     = F.softmax(scaled_logits, dim=-1)
            pitch_idx = torch.multinomial(probs, num_samples=1).item()
            pitch     = pitch_idx / 127.0

            # Clamp in log1p-space so the feedback stays in the model's training range,
            # then invert to real seconds only for MIDI output.
            # Gaussian noise on the log-scale prevents the timing heads from
            # locking into a deterministic repeating pattern over long sequences.
            step_log = max(math.log1p(min_step), s_pred.item() + random.gauss(0, timing_sigma))
            dur_log  = max(math.log1p(min_dur),  d_pred.item() + random.gauss(0, timing_sigma * 0.5))

            generated_notes.append([pitch, math.expm1(step_log), math.expm1(dur_log)])

            # Feed log1p values back — matches the format the model was trained on
            feedback = torch.tensor([pitch, step_log, dur_log], dtype=torch.float32).reshape(1, 1, 3).to(device)
            current_sequence = torch.cat((current_sequence[:, 1:, :], feedback), dim=1)

    return generated_notes


def prepend_seed_to_midi(seed_midi_path, generated_midi_path, out_file=None, seq_length=None):
    """
    Prepend the first seq_length notes of seed_midi_path to generated_midi_path
    so you can hear the seed and generated music back-to-back.

    seed_midi_path      : path to the original seed .mid file
    generated_midi_path : path to any generated .mid file
    out_file            : output path; defaults to <generated>_with_seed.mid
    seq_length          : how many seed notes to include; None = all notes
    """
    seed_pm = pretty_midi.PrettyMIDI(seed_midi_path)
    gen_pm  = pretty_midi.PrettyMIDI(generated_midi_path)

    seed_notes = sorted(seed_pm.instruments[0].notes, key=lambda n: n.start)
    if seq_length is not None:
        seed_notes = seed_notes[:seq_length]

    gen_notes = sorted(gen_pm.instruments[0].notes, key=lambda n: n.start)

    # Shift generated notes to start immediately after the last seed note ends
    seed_end  = max(n.end for n in seed_notes) if seed_notes else 0.0
    gen_start = min(n.start for n in gen_notes) if gen_notes else 0.0
    offset    = seed_end - gen_start

    out_pm = pretty_midi.PrettyMIDI()
    piano  = pretty_midi.Instrument(program=0)

    for n in seed_notes:
        piano.notes.append(pretty_midi.Note(
            velocity=n.velocity, pitch=n.pitch, start=n.start, end=n.end
        ))
    for n in gen_notes:
        piano.notes.append(pretty_midi.Note(
            velocity=n.velocity, pitch=n.pitch,
            start=n.start + offset, end=n.end + offset
        ))

    out_pm.instruments.append(piano)

    if out_file is None:
        base, ext = os.path.splitext(generated_midi_path)
        out_file  = f"{base}_with_seed{ext}"

    out_pm.write(out_file)
    return out_file


def notes_to_midi(notes, out_file='output.mid'):
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)

    current_time = 0
    for pitch, step, dur in notes:
        p     = int(pitch * 127)
        start = current_time + max(0, step)
        end   = start + max(0.01, dur)

        note = pretty_midi.Note(velocity=100, pitch=p, start=start, end=end)
        piano.notes.append(note)
        current_time = start

    pm.instruments.append(piano)
    pm.write(out_file)
