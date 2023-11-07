import numpy as np


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
