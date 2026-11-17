from ds3m.data import load_spy
from ds3m.train import fit

if __name__ == "__main__":
    splits, _, _, _ = load_spy()
    path = "experiments/spy/ds3m_spy_seed3.pt"
    best_valid = fit(splits, seed=3, path=path, tag="SPY seed 3")
    print("Saved:", path, "| best validation loss:", best_valid)