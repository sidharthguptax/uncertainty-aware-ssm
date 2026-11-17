# Training follows the official code (src/DSSSMCode.py, main.py), with these deviations from the paper:
# 6. KL annealing stays at 0.01 for the first half of training, then rises to about 0.51 (paper: 0.01 -> 1).
# Simplifications relative to the official code:
# - No early stopping (theirs averages all past validation losses and rarely triggers).
# - No retraining loop that repeats training until both regimes are used.

import copy
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ds3m.model import DS3M
from ds3m.data import load_toy

torch.manual_seed(3)

splits, (mean, std), y_test, d_test = load_toy()
X_train, Y_train = splits["train"]
X_valid, Y_valid = splits["valid"]

model = DS3M(x_dim=1, y_dim=1, h_dim=10, z_dim=2, K=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)

N_EPOCHS = 100
BATCH = 64
best_valid = float("inf")
best_state = None

for epoch in range(1, N_EPOCHS + 1):
    if epoch < N_EPOCHS / 2:
        anneal = 0.01
    else:
        anneal = min(1.0, 0.01 + epoch / N_EPOCHS / 2)

    model.train()
    for start in range(0, len(X_train), BATCH):
        xb = X_train[start:start + BATCH]
        yb = Y_train[start:start + BATCH]
        nll, kl_z, kl_d = model(xb, yb)
        n = xb.shape[0] * xb.shape[1]
        loss = (nll + anneal * (kl_z + kl_d)) / n
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        nll, kl_z, kl_d = model(X_valid, Y_valid)
        valid_loss = (nll + kl_z + kl_d).item()

    scheduler.step(valid_loss)
    if valid_loss < best_valid:
        best_valid = valid_loss
        best_state = copy.deepcopy(model.state_dict())

    if epoch % 10 == 0:
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d} | anneal {anneal:.2f} | valid loss {valid_loss:.1f} | lr {lr:.0e}")

torch.save(best_state, "experiments/toy/ds3m_toy.pt")
model.load_state_dict(best_state)
print("Best validation loss:", best_valid)
print("Learned Gamma:\n", model.transition().detach())