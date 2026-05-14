# DL-Music-Generation

A comparative music generation project using three architectures:

- `LSTM` (PyTorch)
- `Transformer Encoder` (TensorFlow/Keras)
- `Transformer Decoder` (PyTorch)

The repository contains a working LSTM pipeline and transformer notebooks for encoder and decoder experiments.

## Project overview

- `scripts/data_import.py`: loads MAESTRO MIDI files, converts notes into training sequences, and builds PyTorch data loaders.
- `scripts/music.py`: helper functions for MIDI conversion, generated MIDI export, and seed prepending.
- `scripts/LSTM/lstm.py`: defines the LSTM model and note generation utilities.
- `scripts/LSTM/lstm_train.py`: LSTM training loop with validation, checkpointing, and loss logging.
- `scripts/LSTM/lstm_test.ipynb`: notebook for loading the trained LSTM, plotting loss curves, generating MIDI, and saving outputs.
- `scripts/Transformer-Encoder/transformer_model.ipynb`: notebook for Transformer Encoder preprocessing, training, and generation.
- `scripts/Transformer-Decoder/dl-music-generation.ipynb`: notebook for Transformer Decoder preprocessing, training, and sample generation.
- `scripts/Transformer-Decoder/Decoder_Transformer.ipynb`: alternate decoder-transformer notebook for exploration.
- `maestro/`: local MAESTRO dataset and metadata used for training.

## Models

This repository compares three model approaches:

- **LSTM**: recurrent architecture predicting pitch, step timing, and duration at each note.
- **Transformer Encoder**: attention-based model operating on numeric note features.
- **Transformer Decoder**: token-based autoregressive model for MIDI event sequences.

## Requirements

- Python 3.8+
- PyTorch
- TensorFlow
- NumPy
- pandas
- pretty_midi
- matplotlib

Install the main dependencies with:

```bash
pip install numpy pandas pretty_midi matplotlib torch tensorflow
```

If you have a GPU, install the appropriate PyTorch and TensorFlow packages for your platform.

## Dataset

This project uses the MAESTRO dataset: https://magenta.withgoogle.com/datasets/maestro.

The repository expects the dataset under the `maestro/` folder, including:

- `maestro/maestro-v3.0.0.csv`
- `maestro/maestro-v3.0.0.json`
- `maestro/<year>/*.midi`

## Configuration

For the LSTM pipeline, edit `scripts/LSTM/config.py` to change:

- year selection and number of files
- sequence length, batch size, validation split
- model hyperparameters and architecture
- generation settings such as temperature, minimum step/duration, and note count
- input/output paths for model weights, seed MIDI, and generated MIDI

Transformer notebooks may also contain environment-specific file paths; update paths before running them locally.

## LSTM training

From the repository root, run:

```bash
python scripts/LSTM/lstm_train.py
```

This trains the LSTM, saves the best checkpoint to `scripts/LSTM/models/music_lstm_v1.pth`, and writes loss history to `scripts/LSTM/models/loss_log.json`.

## LSTM music generation

Open `scripts/LSTM/lstm_test.ipynb` and run the notebook cells to:

1. load the trained LSTM model
2. plot training and validation loss
3. generate MIDI from a seed file
4. prepend the seed notes to the generated output

The default generated output is saved to `scripts/LSTM/generated_music/lstm_output2.mid`.

## Transformer Encoder

Open `scripts/Transformer-Encoder/transformer_model.ipynb` to:

- preprocess MAESTRO MIDI data
- build and train a Transformer Encoder model in TensorFlow/Keras
- save the trained model to `scripts/Transformer-Encoder/models/maestro_transformer_model.keras`
- generate MIDI and save outputs to `scripts/Transformer-Encoder/generated_music/`

## Transformer Decoder

Open `scripts/Transformer-Decoder/dl-music-generation.ipynb` to:

- convert MIDI into symbolic event tokens
- build and train a Transformer Decoder model in PyTorch
- sample autoregressive MIDI event sequences
- convert generated tokens back into MIDI

The `scripts/Transformer-Decoder/Decoder_Transformer.ipynb` notebook is also available for additional decoder-transformer experimentation.

## Notes

- The LSTM model uses normalized pitch and log-scaled step/duration features.
- The Transformer Encoder notebook uses TensorFlow/Keras and saves models as `.keras` files.
- The Transformer Decoder notebook uses a token vocabulary of `NOTE_ON`, `NOTE_OFF`, and `TIME_SHIFT` events.
- Generated MIDI outputs are stored under each model folder's `generated_music/` directory.

## License

This repository is provided as an experimental project for research and learning.
