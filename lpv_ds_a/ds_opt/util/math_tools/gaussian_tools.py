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





def my_gaussPDF(X, Mu, Sigma):
    """
    %MY_GAUSSPDF computes the Probability Density Function (PDF) of a
    % multivariate Gaussian represented by a mean and covariance matrix.
    %
    % Inputs -----------------------------------------------------------------
    %       o X     : (N x M), a data set with M samples each being of dimension N.
    %                          each column corresponds to a datapoint
    %       o Mu    : (N x 1), an Nx1 vector corresponding to the mean of the
    %							Gaussian function
    %       o Sigma : (N x N), an NxN matrix representing the covariance matrix
    %						   of the Gaussian function
    % Outputs ----------------------------------------------------------------
    %       o prob  : (1 x M),  a 1xM vector representing the probabilities for each
    %                           M datapoints given Mu and Sigma
    %%
    This function has passed the check ( 08/11/22 )
    """
    # Auxiliary Variables
    N = len(X)
    M = len(X[0])

    # Output Variable
    prob = np.zeros(M)

    # Demean Data
    X = X - Mu

    # Compute Probabilities
    A = X.T @ np.linalg.inv(Sigma)
    B = A * X.T
    # 1) The exponential term is the inner products of the zero-mean data
    exp_term = np.sum(B, axis=1)

    # 2) Compute Equation (2) there is a real-min here but I did'nt add it
    prob = np.exp(-0.5 * exp_term) / np.sqrt((2*np.pi)**N * (np.abs(np.linalg.det(Sigma))))

    return prob


def posterior_probs_gmm(x, gmm, type):
    N = len(x)
    M = len(x[0])

    # Unpack gmm
    Mu = gmm.Mu
    Priors = gmm.Priors
    Sigma = gmm.Sigma
    K = len(Priors)
    # Compute mixing weights for multiple dynamics
    Px_k = np.zeros((K, M))

    # Compute probabilities p(x^i|k)
    for k in np.arange(K):
        Px_k[k, :] = my_gaussPDF(x, Mu[:, k].reshape(N, 1), Sigma[k, :, :])

    # Compute posterior probabilities p(k|x) -- FAST WAY --- %%%
    alpha_Px_k = np.repeat(Priors.reshape(len(Priors),1), M, axis=1) * Px_k

    if type == 'norm':
        Pk_x = alpha_Px_k / np.repeat(np.sum(alpha_Px_k, axis=0, keepdims=True), K, axis=0)
    else:
        Pk_x = alpha_Px_k

    return Pk_x

# Note:
# 这个函数首先计算了一个K*M的矩阵，这个矩阵的列是当前点属于各个cluster的概率（基于point）然后alpha_Px_k是将prior扩张成K*M，乘上Pk_x
# 就能够得到 每个点属于每个cluster的概率,这里的norm就是 P（z = k) / (sum from i = 1 to K) P(z = i)
# 该函数最终返回一个K * M的矩阵，表示每个点属于每个类的概率

# np.sum([[0, 1], [0, 5]], axis=0)
# array([0, 6])
# np.sum([[0, 1], [0, 5]], axis=1)
# array([1, 5])
# sum 沿着什么axis就是沿着哪里加，比如0轴是列，axis=0就是沿着列相加