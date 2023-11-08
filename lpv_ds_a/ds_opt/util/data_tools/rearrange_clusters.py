import numpy as np
from ..math_tools import gaussian_tools
from ..data_tools import structures



def knn_search(Mu, att, size):
    distances = np.zeros(len(Mu))
    index = 0
    for mu in Mu:
        distances[index] = np.linalg.norm(mu - att)
        index = index + 1

    order = []
    for i in np.arange(0, len(Mu)):
        cur_value = distances[i]
        cur_index = 0
        for j in np.arange(0, len(Mu)):
            if distances[j] < cur_value:
                cur_index += 1
        order.append(cur_index)

    return order



# Plug in Priors (1xK) Mu(K x dim) Sigma (K x dim x dim) attractor
def rearrange_clusters(Priors, Mu, Sigma, att):
    Mu = Mu.T
    dim = len(Mu)
    # rearrange the probability arrangement
    idx = knn_search(Mu.T, att.reshape(len(att)), len(Mu[0]))
    Priors_old = Priors.copy()
    Mu_old = Mu.copy()
    Sigma_old = Sigma.copy()
    for i in np.arange(len(idx)):
        Priors[idx[i]] = Priors_old[i]
        Mu[:, idx[i]] = Mu_old[:, i]
        Sigma[idx[i]] = Sigma_old[i]
    # Make the closest Gaussian isotropic and place it at the attractor location
    #Sigma[0] = 1 * np.max(np.diag(Sigma[0])) * np.eye(dim)
    #Mu[:, 0] = att.reshape(len(att))
    # gmm = GMM(len(Mu[0]), Priors, Mu.T, Sigma)  # checked 10/22/2022

    # This is recommended to get smoother streamlines/global dynamics
    # This is used to expand the covariance
    ds_gmm = structures.ds_gmms()
    ds_gmm.Mu = Mu
    ds_gmm.Sigma = Sigma
    ds_gmm.Priors = Priors
    adjusts_C = 1
    if adjusts_C == 1:
        if dim == 2:
            tot_dilation_factor = 1
            rel_dilation_fact = 0.25
        else:
            # this is for dim == 3
            tot_dilation_factor = 1
            rel_dilation_fact = 0.75
        Sigma_ = gaussian_tools.adjust_covariances(ds_gmm.Priors, ds_gmm.Sigma, tot_dilation_factor, rel_dilation_fact)
        ds_gmm.Sigma = Sigma_

    return ds_gmm
