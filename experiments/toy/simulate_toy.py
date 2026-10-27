import numpy as np

#Assumptions:
# 1. N(0, s) in eq. (23) is read as standard deviation s. Confirmed against the
#    official Toy CSV: regime-0 std of y is about 20, matching this reading.
# 2. Y_0 = 0, so that X_1 = Y_0 is defined.

SEED = 42
T = 2000

GAMMA = np.array([[0.95, 0.05],
                  [0.05, 0.95]])

STD_W = {0: 10.0, 1: 1.0} # transition noise std, w_t^(d)
STD_V = {0: 5.0, 1: 0.5} # emission noise std, v_t^(d)

rng = np.random.default_rng(SEED)

d = np.zeros(T + 1, dtype=int)
d[0] = rng.binomial(1, 0.5)

for t in range(1, T+1):
    d[t] = rng.choice(2, p=GAMMA[d[t-1]])

print("Fraction of time in regime 0:", np.mean(d[1:] == 0))
print("Number of switches:", np.sum(d[2:] != d[1:-1]))

z = np.zeros(T+1)
y = np.zeros(T+1)
x = np.zeros(T+1)

for t in range(1,T+1):
    x[t] = y[t-1]
    w = rng.normal(0, STD_W[d[t]])
    v = rng.normal(0, STD_V[d[t]])

    if d[t] == 0:
        z[t] = 0.6 * z[t-1] + 0.4 * np.tanh(x[t] + z[t-1]) + w
        y[t] = 1.5 * z[t] + np.tanh(z[t]) + v

    else:
        z[t] = 0.1 * z[t-1] + 0.2 * np.sin(x[t] + z[t-1]) + w
        y[t] = 0.5 * z[t] + np.sin(z[t]) + v

print("Std of y in regime 0:", np.std(y[1:][d[1:]==0]))
print("Std of y in regime 1:", np.std(y[1:][d[1:]==1]))

counts = np.zeros((2,2))
for t in range(2, T+1):
    counts[d[t-1], d[t]] += 1

gamma_hat = counts / counts.sum(axis=1, keepdims=True)
print("Estimated Gamma:\n", gamma_hat)

runs = {0: [], 1: []}
length = 1
for t in range(2, T+1):
    if d[t] == d[t-1]:
        length += 1
    else:
        runs[d[t-1]].append(length)
        length = 1
runs[d[T]].append(length)
print("Mean duration, regime 0:", np.mean(runs[0]))
print("Mean duration, regime 1:", np.mean(runs[1]))