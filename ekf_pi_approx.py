import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

"""
x = [x, xdot]
xdot = [xdot, xddot]
"""
def predict_step(x_prev, P_prev, Ad):
    x_pred = Ad @ x_prev
    P_pred = Ad @ P_prev @ Ad.T
    return x_pred, P_pred

def update_step(x_pred, P_pred, z, H, R):
    n = 2
    z_pred = H @ x_pred
    innov = z - z_pred
    S = H @ P_pred @ H.T + R
    W = P_pred @ H.T @ np.linalg.inv(S)

    x_upd = x_pred + W @ innov
    P_upd = (np.eye(n) - W @ H) @ P_pred
    return x_upd, P_upd

def kalman_step(x_prev, P_prev, z, dt, Ad, H, R):
    x_pred, P_pred = predict_step(x_prev, P_prev, Ad)
    x_upd, P_upd = update_step(x_pred, P_pred, z, H, R)
    return x_upd, P_upd

def estimate_pi(dt, sigma_z, T):

    x0 = [1, 0]
    N = int(T/dt)


    k = 4
    d = 0
    m = 1

    A = np.array([
        [0, 1], [-k/m, -d/m]
    ])
    H = np.array([
        [1, 0], [0, 1]
    ])
    sigma_z = 0.3
    sigma_a = 0.1

    Q = sigma_a**2 * np.eye(2)
    R = np.array([[sigma_z**2, 0], [0, sigma_z**2]]) 

    eigs = np.linalg.eigvals(A)

    Ad = expm(A*dt)
    eigs = np.linalg.eigvals(Ad)

    all_x = np.zeros((N, len(x0)))

    for i in range(N):
        all_x[i,:] = x0
    
        x0 = Ad @ x0
    all_x_noisy = all_x + np.random.normal(0, sigma_z, (N,2))

    x_kf = np.zeros_like(all_x)
    x_kf0 = np.array([0, 0])
    P_kf0 = np.array([[1,0], [0,1]])

    for i in range(N):
        if i == 0:
            x_prev = x_kf0
            P_prev = P_kf0
        z = all_x_noisy[i,:]
        x_n, P_n = kalman_step(x_prev, P_prev, z, dt, Ad, H, R)
        x_kf[i, :] = x_n
        x_prev, P_prev = x_n, P_n

    x_pos = x_kf[:,0]
    x_pos_noise = all_x_noisy[:,0]

    start_i = int(2 / dt)
    ind = start_i
    diff = 2
    while diff > 0:
        diff = x_kf[ind+1, 0] - x_kf[ind, 0]
        ind += 1

    pi = 0
    for i in range(ind):
        pi += np.sqrt( (x_pos[ind+1]-x_pos[ind])**2 + dt**2 )
    t = np.linspace(0, T, N)
    print(pi)
    return pi

#plt.plot(t,x_kf[:,0])
#plt.plot(t,all_x_noisy[:,0])
#plt.show()