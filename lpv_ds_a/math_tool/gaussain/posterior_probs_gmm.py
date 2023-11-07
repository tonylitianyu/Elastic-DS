import numpy as np
from lpv_ds_a.math_tool.gaussain.my_gaussPDF import my_gaussPDF
# havent been tested yet


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




