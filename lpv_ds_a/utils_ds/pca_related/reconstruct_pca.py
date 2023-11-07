import numpy as np

def reconstruct_pca(Y, A_p, Mu):
    # We need the Mu be the shape M
    X_hat = np.linalg.pinv(A_p) @ Y + Mu.reshape(len(Mu), 1)
    return X_hat