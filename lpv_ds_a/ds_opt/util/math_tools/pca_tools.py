import numpy as np


def my_minmax(x):
    x_sorted = np.sort(x)
    mm = np.array([x_sorted[0], x_sorted[-1]])
    return mm


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


def project_pca(X, Mu, V, p):

    A_p = V[:, :p].T

    N, M = X.shape # N for dim M for num

    X = X - Mu.reshape(N, 1) 

    Y = A_p @ X

    return A_p, Y

def reconstruct_pca(Y, A_p, Mu):
    # We need the Mu be the shape M
    X_hat = np.linalg.pinv(A_p) @ Y + Mu.reshape(len(Mu), 1)
    return X_hat