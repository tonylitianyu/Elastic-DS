import numpy as np
from lpv_ds_a.math_tool.gaussain.posterior_probs_gmm import posterior_probs_gmm


def lpv_ds(x, ds_gmm, A_g, b_g):
    N, M = x.shape  # N for dim and M for data point number
    K = len(ds_gmm.Priors)

    # Posterior Probabilities per local DS
    beta_k_x = posterior_probs_gmm(x, ds_gmm, 'norm')  # K*M matrix

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
