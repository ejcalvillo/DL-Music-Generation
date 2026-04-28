import os

# ── Data ──────────────────────────────────────────────────────────────────────
YEAR       = 2011   # filter MAESTRO dataset to this year; None = all years
NUM_FILES  = None   # cap number of files; None = all files for the year
SEQ_LENGTH = 32     # context window in notes fed to the LSTM each step
BATCH_SIZE = 128    # samples per gradient update
VAL_SPLIT  = 0.1   # fraction of sequences held out for validation

# ── Architecture ──────────────────────────────────────────────────────────────
PITCH_EMBED_DIM = 32    # size of the learned pitch embedding (see lstm.py)
HIDDEN_SIZE     = 512   # LSTM hidden units per layer; try 512 for more capacity
NUM_LAYERS      = 2     # stacked LSTM layers; 3 adds depth but trains slower
DROPOUT         = 0.3   # applied between LSTM layers and before output heads

# ── Training ──────────────────────────────────────────────────────────────────
LEARNING_RATE = 3e-4  # Adam lr; 1e-3 converges faster but overshoots more
EPOCHS        = 40    # total passes over the training data
GRAD_CLIP     = 1.0   # max gradient norm; prevents LSTM gradient explosions
WEIGHT_DECAY  = 1e-4  # L2 regularization on weights; helps prevent overfitting
W_PITCH       = 0.5   # weight on pitch cross-entropy loss
W_STEP        = 1.0   # weight on step MSE loss
W_DUR         = 1.0   # weight on duration MSE loss

# ── Generation ────────────────────────────────────────────────────────────────
NUM_TO_GENERATE = 200   # notes to generate after the seed
TEMPERATURE     = 0.5   # < 1.0 = safer/repetitive, > 1.0 = more experimental
MIN_STEP        = 0.2   # minimum seconds between note onsets (~1/8 note at 100 BPM); raise to space notes out more
MIN_DUR         = 0.15  # minimum note duration in seconds
TIMING_SIGMA    = 0.12  # standard deviation of Gaussian noise added to timing predictions

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(__file__)

CHECKPOINT_EVERY = 5   # save a full resume checkpoint every N epochs

MODEL_PATH    = os.path.join(_HERE, 'models', 'music_lstm_v1.pth')
RESUME_CKPT   = os.path.join(_HERE, 'models', 'resume_checkpoint.pth')
LOSS_LOG      = os.path.join(_HERE, 'models', 'loss_log.json')

# The seed file every model uses for a fair comparison
SEED_MIDI = os.path.join(
    _HERE, '..', '..', 'maestro', '2013',
    'ORIG-MIDI_01_7_6_13_Group__MID--AUDIO_01_R1_2013_wav--1.midi'
)

OUTPUT_MIDI = os.path.join(_HERE, 'generated_music', 'lstm_output2.mid')
