# Training follows the official code (src/DSSSMCode.py, main.py), with these deviations from the paper:
# 6. KL annealing stays at 0.01 for the first half of training, then rises to about 0.51 (paper: 0.01 -> 1).
# Simplifications relative to the official code:
# - No early stopping (theirs averages all past validation losses and rarely triggers).
# - No retraining loop that repeats training until both regimes are used.
# Optional stabilisation (off by default, so the official recipe is reproduced exactly):
# - shuffle=True: shuffled mini-batches each epoch (official code goes through batches in date order).
# - clip=10: gradient-norm clipping at 10 (official main.py defines clip=10 but never applies it).

import copy
import json
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ds3m.model import DS3M


def fit(splits, seed, path, n_epochs=100, batch=64, h_dim=10, z_dim=2, K=2,
        tag="", shuffle=False, clip=None):
    torch.manual_seed(seed)
    X_train, Y_train = splits["train"]
    X_valid, Y_valid = splits["valid"]
    y_dim = X_train.shape[-1]

    model = DS3M(x_dim=y_dim, y_dim=y_dim, h_dim=h_dim, z_dim=z_dim, K=K)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)

    best_valid = float("inf")
    best_state = None

    for epoch in range(1, n_epochs + 1):
        if epoch < n_epochs / 2:
            anneal = 0.01
        else:
            anneal = min(1.0, 0.01 + epoch / n_epochs / 2)

        model.train()
        order = torch.randperm(len(X_train)) if shuffle else torch.arange(len(X_train))
        train_loss = 0.0
        for start in range(0, len(X_train), batch):
            idx = order[start:start + batch]
            xb, yb = X_train[idx], Y_train[idx]
            nll, kl_z, kl_d = model(xb, yb)
            n = xb.shape[0] * xb.shape[1]
            loss = (nll + anneal * (kl_z + kl_d)) / n
            optimizer.zero_grad()
            loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            train_loss += loss.item() * len(idx)
        train_loss /= len(X_train)

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
            print(f"{tag} | epoch {epoch:3d} | anneal {anneal:.2f} | train loss {train_loss:.3f} "
                  f"| valid loss {valid_loss:.1f} | lr {lr:.0e}")

    torch.save(best_state, path)
    model.load_state_dict(best_state)
    gamma = model.transition().detach()
    print(f"{tag} | learned Gamma:\n", gamma)
    with open(path.replace(".pt", "_gamma.json"), "w") as f:
        json.dump(gamma.tolist(), f)
    return best_valid