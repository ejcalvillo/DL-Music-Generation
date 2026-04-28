import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config as C
import torch
from data_import import load_data
from lstm import MusicLSTM


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading data...")
train_loader, val_loader = load_data(
    year=C.YEAR, num_files=C.NUM_FILES,
    seq_length=C.SEQ_LENGTH, batch_size=C.BATCH_SIZE,
    val_split=C.VAL_SPLIT,
)
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

# ── Model ─────────────────────────────────────────────────────────────────────
model = MusicLSTM(
    pitch_embed_dim=C.PITCH_EMBED_DIM,
    hidden_size=C.HIDDEN_SIZE,
    num_layers=C.NUM_LAYERS,
    dropout=C.DROPOUT,
).to(device)

os.makedirs(os.path.dirname(C.MODEL_PATH), exist_ok=True)

criterion_pitch = torch.nn.CrossEntropyLoss()
criterion_mse   = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=C.LEARNING_RATE, weight_decay=C.WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# ── Resume from checkpoint if available ───────────────────────────────────────
start_epoch   = 0
best_val_loss = float("inf")
loss_log      = {"train": [], "val": []}

if os.path.exists(C.RESUME_CKPT):
    ckpt = torch.load(C.RESUME_CKPT, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch   = ckpt["epoch"] + 1
    best_val_loss = ckpt["best_val_loss"]
    loss_log      = ckpt["loss_log"]
    print(f"Resumed from epoch {start_epoch} (best val: {best_val_loss:.4f})")
elif os.path.exists(C.MODEL_PATH):
    # Best-model checkpoint only — restore weights but not optimizer state
    model.load_state_dict(torch.load(C.MODEL_PATH, map_location=device))
    print(f"Loaded best model weights from {C.MODEL_PATH} (optimizer state not restored)")
else:
    print("Starting from scratch.")

# ── Training loop ─────────────────────────────────────────────────────────────
for epoch in range(start_epoch, C.EPOCHS):
    # ── train ──
    model.train()
    total = 0
    for batch_x, b_pitch, b_step, b_dur in train_loader:
        batch_x = batch_x.to(device)
        b_pitch = b_pitch.to(device)
        b_step  = b_step.to(device)
        b_dur   = b_dur.to(device)

        optimizer.zero_grad()
        p_logits, s_pred, d_pred = model(batch_x)
        loss = (
            C.W_PITCH * criterion_pitch(p_logits, b_pitch) +
            C.W_STEP  * criterion_mse(s_pred, b_step) +
            C.W_DUR   * criterion_mse(d_pred, b_dur)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
        optimizer.step()
        total += loss.item()

    train_loss = total / len(train_loader)

    # ── validate ──
    model.eval()
    total = 0
    with torch.no_grad():
        for batch_x, b_pitch, b_step, b_dur in val_loader:
            batch_x = batch_x.to(device)
            b_pitch = b_pitch.to(device)
            b_step  = b_step.to(device)
            b_dur   = b_dur.to(device)

            p_logits, s_pred, d_pred = model(batch_x)
            loss = (
                C.W_PITCH * criterion_pitch(p_logits, b_pitch) +
                C.W_STEP  * criterion_mse(s_pred, b_step) +
                C.W_DUR   * criterion_mse(d_pred, b_dur)
            )
            total += loss.item()

    val_loss = total / len(val_loader)

    loss_log["train"].append(train_loss)
    loss_log["val"].append(val_loss)

    # Save best inference model when val loss improves
    improved = val_loss < best_val_loss
    if improved:
        best_val_loss = val_loss
        torch.save(model.state_dict(), C.MODEL_PATH)

    # Save full resume checkpoint every N epochs so a crash loses at most N epochs
    if (epoch + 1) % C.CHECKPOINT_EVERY == 0:
        torch.save({
            "epoch":         epoch,
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "scheduler":     scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "loss_log":      loss_log,
        }, C.RESUME_CKPT)

    print(f"Epoch {epoch+1:>3}/{C.EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}{' *' if improved else ''}")
    scheduler.step(val_loss)

# ── Done ──────────────────────────────────────────────────────────────────────
with open(C.LOSS_LOG, "w") as f:
    json.dump(loss_log, f)

# Resume checkpoint no longer needed once training completes cleanly
if os.path.exists(C.RESUME_CKPT):
    os.remove(C.RESUME_CKPT)

print(f"Best val loss: {best_val_loss:.4f}")
print(f"Model saved to {C.MODEL_PATH}")
print(f"Loss log saved to {C.LOSS_LOG}")