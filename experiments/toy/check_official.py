import numpy as np

OFFICIAL = "../Deep-Switching-State-Space-Model/data/Toy/"

y_off = np.loadtxt(OFFICIAL + "simulation_data_nonlinear_y.csv", delimiter=",")
d_off = np.loadtxt(OFFICIAL + "simulation_data_nonlinear_d.csv", delimiter=",")

print("Length of y:", len(y_off), "| Length of d:", len(d_off))
print("Std of their y in regime 0:", np.std(y_off[d_off[1:] == 0]))
print("Std of their y in regime 1:", np.std(y_off[d_off[1:] == 1]))

print("Alt alignment, regime 0:", np.std(y_off[d_off[:-1] == 0]))
print("Alt alignment, regime 1:", np.std(y_off[d_off[:-1] == 1]))