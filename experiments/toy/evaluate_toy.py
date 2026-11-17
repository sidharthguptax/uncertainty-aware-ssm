import numpy as np
import torch

from ds3m.model import DS3M
from ds3m.data import load_toy


def mean_duration(states):
    runs, length = [], 1
    for t in range(1, len(states)):
        if states[t] == states[t - 1]:
            length += 1
        else:
            runs.append(length)
            length = 1
    runs.append(length)
    return np.mean(runs)


torch.manual_seed(3)
splits, (mean, std), y_test, d_test = load_toy()
X_test, Y_test = splits["test"]

model = DS3M(x_dim=1, y_dim=1, h_dim=10, z_dim=2, K=2)
model.load_state_dict(torch.load("experiments/toy/ds3m_toy.pt"))
model.eval()

y_samples, d_samples = model.forecast(X_test, Y_test, S=200)
y_samples = y_samples[..., 0].numpy() * std + mean

y_true = y_test[:, 0]
y_mean = y_samples.mean(axis=1)
rmse = np.sqrt(np.mean((y_mean - y_true) ** 2))
lo, hi = np.quantile(y_samples, [0.05, 0.95], axis=1)
coverage = np.mean((y_true >= lo) & (y_true <= hi))

d_pred = (d_samples.float().mean(dim=1) > 0.5).long().numpy()
acc = max(np.mean(d_pred == d_test), np.mean(d_pred != d_test))

print("RMSE:", rmse)
print("90% interval coverage:", coverage)
print("Regime accuracy:", acc)
print("Mean duration, predicted:", mean_duration(d_pred), "| true:", mean_duration(d_test))