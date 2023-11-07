import numpy as np


def my_pca(X_input):
    # Auxiliary variables
    X = np.copy(X_input)
    N = len(X)
    M = len(X[0])

    # Output variables
    V = np.zeros((N, N))
    L = np.zeros((N, N))
    Mu = np.zeros((N, 1))

    if M > 1:
        Mu = np.mean(X, axis=1, keepdims=True)
    else:
        Mu = np.mean(X, axis=0, keepdims=True)

    X = X - Mu

    if M > 1:
        C = (1 / (M - 1)) * X @ X.T
    else:
        C = (1 / M) * X @ X.T

    L, V = np.linalg.eig(C) 
    ids = np.argsort(-L)
    L = np.diag(L[ids])
    V = V[:, ids]

    return V, L, Mu.reshape(-1)
