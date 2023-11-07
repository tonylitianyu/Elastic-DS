import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plot_sample = 40

def multivariate_normal(x, d, mean, covariance):
    """pdf of the multivariate normal distribution."""
    x_m = x - mean
    return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(covariance))) * 
            np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))

def generate_surface_3d(mean, covariance, d):
    """Helper function to generate density surface."""
    nb_of_x = plot_sample # grid size
    x1s = np.linspace(0.0, 1.0, num=nb_of_x)
    x2s = np.linspace(-1.0, 1.0, num=nb_of_x)
    x3s = np.linspace(-1.0, 1.0, num=nb_of_x)
    x1, x2, x3 = np.meshgrid(x1s, x2s, x3s) # Generate grid
    pdf = np.zeros((nb_of_x, nb_of_x, nb_of_x))
    # Fill the cost matrix for each combination of weights

    for i in range(nb_of_x):
        for j in range(nb_of_x):
            for k in range(nb_of_x):
                pdf[i,j,k] = multivariate_normal(
                    np.matrix([[x1[i,j,k]], [x2[i,j,k]], [x3[i,j,k]]]), 
                    d, mean, covariance)

    return x1, x2, x3, pdf  # x1, x2, pdf(x1,x2)


def create_ellipsoid(center, radii, rotation_matrix, num_points=100):
    # Generate points on a unit sphere
    u = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(0, np.pi, num_points)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    # Scale the points to the ellipsoid's radii
    ellipsoid_points = np.array([x, y, z]) * radii[:, np.newaxis, np.newaxis]

    # Rotate the points according to the rotation matrix
    ellipsoid_points = np.einsum("ij, jkl -> ikl", rotation_matrix, ellipsoid_points)

    # Shift the points to the ellipsoid's center
    ellipsoid_points += center[:, np.newaxis, np.newaxis]

    return ellipsoid_points


if __name__ == "__main__":
    #fig, ax1= plt.subplots(nrows=1, ncols=1, figsize=(8,8), projection='3d')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')



    mu = np.array([0.5,0.5,0.5])
    sigma = np.array([[0.1,0,0],[0,0.1,0],[0,0,0.1]])
    eigval, eigvec = np.linalg.eigh(sigma)

    # rot = np.deg2rad(30)
    # T = np.array([[np.cos(rot), -np.sin(rot), 0.0],
    #             [np.sin(rot), np.cos(rot),  0.0],
    #             [0.0, 0.0, 1.0]])

    sample_pts = create_ellipsoid(mu, np.sqrt(eigval), eigvec)

    print(sample_pts.shape)



    #ax.scatter(sample_pts[:,0], sample_pts[:,1], sample_pts[:,2])
    ax.plot_surface(sample_pts[0], sample_pts[1], sample_pts[2], color='blue', alpha=0.7)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([0.0,1.0])
    ax.set_ylim([0.0,1.0])
    ax.set_zlim([0.0,1.0])
    ax.set_aspect('equal')
    plt.show()