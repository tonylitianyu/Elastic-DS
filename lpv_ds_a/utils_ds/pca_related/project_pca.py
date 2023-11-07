import numpy as np


def project_pca(X, Mu, V, p):

    A_p = V[:, :p].T

    N, M = X.shape # N for dim M for num

    X = X - Mu.reshape(N, 1)

    Y = A_p @ X

    return A_p, Y