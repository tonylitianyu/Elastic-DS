import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plot_sample = 200

def multivariate_normal(x, d, mean, covariance):
    """pdf of the multivariate normal distribution."""
    x_m = x - mean
    return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(covariance))) * 
            np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))

def generate_surface(mean, covariance, d):
    """Helper function to generate density surface."""
    nb_of_x = plot_sample # grid size
    x1s = np.linspace(0.0, 1.0, num=nb_of_x)
    x2s = np.linspace(0.0, 1.0, num=nb_of_x)
    x1, x2 = np.meshgrid(x1s, x2s) # Generate grid
    pdf = np.zeros((nb_of_x, nb_of_x))
    # Fill the cost matrix for each combination of weights
    for i in range(nb_of_x):
        for j in range(nb_of_x):
            pdf[i,j] = multivariate_normal(
                np.matrix([[x1[i,j]], [x2[i,j]]]), 
                d, mean, covariance)
    return x1, x2, pdf

def create_ellipsoid2d(center, radii, rotation_matrix, num_points=100):
    # Generate points on a unit sphere
    u = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(0, np.pi, num_points)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))

    # Scale the points to the ellipsoid's radii
    ellipsoid_points = np.array([x, y]) * radii[:, np.newaxis, np.newaxis]

    # Rotate the points according to the rotation matrix
    ellipsoid_points = np.einsum("ij, jkl -> ikl", rotation_matrix, ellipsoid_points)

    # Shift the points to the ellipsoid's center
    ellipsoid_points += center[:, np.newaxis, np.newaxis]

    return ellipsoid_points
