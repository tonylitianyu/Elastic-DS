import numpy as np


def adjust_covariances(Priors, Sigma, tot_scale_fact, rel_scale_fact):
    # Check Relative Covariance Matrix Eigenvalues
    est_K = len(Sigma)
    dim = np.shape(Sigma)[1]
    Vs = np.zeros((est_K, dim, dim))
    Ls = np.zeros((est_K, dim, dim))
    p1_eig = []
    p2_eig = []
    p3_eig = []

    baseline_prior = (0.5 / len(Priors))

    for k in np.arange(est_K):
        w, v = np.linalg.eig(Sigma[k])
        Ls[k] = np.diag(w)
        Vs[k] = v.copy()
        if not all(sorted(w) == w):
            ids = w.argsort()
            L_ = np.sort(w)
            Ls[k] = np.diag(L_)
            Vs[k] = Vs[k][:, ids]

        if Priors[k] > baseline_prior:
            Ls[k] = tot_scale_fact * Ls[k]

        # 提取最大的两个特征值
        lambda_1 = Ls[k][0][0]
        lambda_2 = Ls[k][1][1]
        p1_eig.append(lambda_1)
        p2_eig.append(lambda_2)
        if dim == 3:
            lambda_3 = Ls[k][2][2]
            p3_eig.append(lambda_3)
        Sigma[k] = Vs[k] @ Ls[k] @ Vs[k].T

    p1_eig = np.array(p1_eig)
    p2_eig = np.array(p2_eig)
    p3_eig = np.array(p3_eig)

    if dim == 2:
        cov_ratios = np.array(p1_eig / p2_eig)
        for k in np.arange(0, est_K):
            if cov_ratios[k] < rel_scale_fact:
                lambda_1 = p1_eig[k]
                lambda_2 = p2_eig[k]
                lambda_1_ = lambda_1 + lambda_2 * (rel_scale_fact - cov_ratios[k])
                Sigma[k] = Vs[k] @ np.diag([lambda_1_, lambda_2]) @ Vs[k].T
    elif dim == 3:
        cov_ratios = np.array(p2_eig / p3_eig)
        for k in np.arange(0, est_K):
            if cov_ratios[k] < rel_scale_fact:
                lambda_1 = p1_eig[k]
                lambda_2 = p2_eig[k]
                lambda_3 = p3_eig[k]
                lambda_2_ = lambda_2 + lambda_3 * (rel_scale_fact - cov_ratios[k])
                lambda_1_ = lambda_1 + lambda_3 * (rel_scale_fact - cov_ratios[k])
                Sigma[k] = Vs[k] @ np.diag([lambda_1_, lambda_2_, lambda_3]) @ Vs[k].T

    return Sigma





