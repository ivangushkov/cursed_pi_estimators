from ekf_pi_approx import estimate_pi
import numpy as np

sigma_z = 0.03
dt = 1e-2
T = 4

M = 1000
pies = np.zeros((M,))

for i in range(M):
    pi = estimate_pi(dt, sigma_z, T)
    pies[i] = pi


pi_exact = np.average(pies)
print(f"The exact value of pi is: {pi_exact}")
