import dtw
import numpy as np

from ..data_tools import simulation, structures
from ..math_tools import gaussian_tools


def compute_rmse(fun_handle, Xi_ref, Xi_dot_ref):
    Xi_dot_pred = fun_handle(Xi_ref)
    predict_diff = (Xi_dot_pred - Xi_dot_ref) ** 2
    trajectory_RMSE = np.sqrt(np.mean(predict_diff, axis=0))
    rmse = np.mean(trajectory_RMSE)
    print('LPV-DS got prediction RMSE on training set: {}'.format(rmse))
    return rmse


def compute_e_dot(fun_handle, Xi_ref, Xi_dot_ref):
    Xi_dot_pred = fun_handle(Xi_ref).T
    Xi_dot_ref = Xi_dot_ref.T
    e_cum = 0
    M = Xi_ref.shape[1]
    for i in np.arange(M):
        norm_term = np.linalg.norm(Xi_dot_pred[i]) * np.linalg.norm(Xi_dot_ref[i])
        if norm_term < 10 ** (-10):
            e_cum += 0
        else:
            e_cum += np.abs(1 - (Xi_dot_pred[i] @ Xi_dot_ref[i].reshape(-1, 1)) / norm_term)

    print('LPV-DS got prediction e_dot on training set: {}'.format(e_cum/M))
    return e_cum / M


def compute_dtwd(fun_handle, Xi_ref, demo_size, x0_all):
    opt_sim = structures.Opt_Sim()
    opt_sim.dt = 0.005
    opt_sim.i_max = 10000
    opt_sim.tol = 0.001
    opt_sim.plot = 0
    num_traj = x0_all.shape[1]
    x_sim = simulation.simulation(x0_all, fun_handle, opt_sim)
    trajs = np.array(x_sim)
    start_idx = 0
    dist = []
    
    for i in np.arange(num_traj):
        cur_traj = trajs[:, :, i]
        end_index = start_idx + demo_size[i]
        cur_ref_traj = Xi_ref[:, start_idx:end_index].T
        dist.append(dtw.dtw(cur_traj, cur_ref_traj).distance)
        start_idx = end_index

    mean_dist = np.mean(dist)
    cov_dist = np.std(dist)
    print('LPV-DS got prediction dwtd on training set: {} +- {}'.format(mean_dist, cov_dist))
    return dist


def reproduction_metrics(Data, A_k, b_k, traj_length, x0_all, ds_gmm):
    dim = int( len(Data) / 2)
    Xi_ref = Data[:dim]
    Xi_dot_ref = Data[dim:]
    ds_handle = lambda x_velo: lpv_ds(x_velo, ds_gmm, A_k, b_k)
    rmse = compute_rmse(ds_handle, Xi_ref, Xi_dot_ref)
    e_dot = compute_e_dot(ds_handle, Xi_ref, Xi_dot_ref)
    dtwd = compute_dtwd(ds_handle, Xi_ref, traj_length, x0_all)
    return rmse, e_dot, dtwd



def lpv_ds(x, ds_gmm, A_g, b_g):
    N, M = x.shape  # N for dim and M for data point number
    K = len(ds_gmm.Priors)

    # Posterior Probabilities per local DS
    beta_k_x = gaussian_tools.posterior_probs_gmm(x, ds_gmm, 'norm')  # K*M matrix

    x_dot = np.zeros((N, M))
    for i in np.arange(M):
        f_g = np.zeros((N, K))  # dim * cluster p（k）* （A_k @ x + b_k)
        if b_g.shape[1] > 1:
            for k in np.arange(K):
                f_g[:, k] = beta_k_x[k][i] * (A_g[k] @ x[:, i] + b_g[:, k])
            f_g = np.sum(f_g, axis=1)

        else:
            # Estimate the Global Dynamics component as Linear DS
            f_g = (A_g @ x[:, i] + b_g.reshape(-1))

        x_dot[:, i] = f_g

    return x_dot