from ds3m.data import load_toy
from ds3m.train import fit


def train(seed, n_epochs=100, batch=64):
    splits, _, _, _ = load_toy()
    path = f"experiments/toy/ds3m_toy_seed{seed}.pt"
    best_valid = fit(splits, seed, path, n_epochs, batch, tag=f"Seed {seed}")
    return path, best_valid


if __name__ == "__main__":
    path, best_valid = train(seed=3)
    print("Saved:", path, "| best validation loss:", best_valid)