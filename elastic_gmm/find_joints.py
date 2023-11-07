import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy
import pickle
import copy
from elastic_gmm.gaussian2d import generate_surface, plot_sample
from elastic_gmm.gaussian3d import generate_surface_3d, create_ellipsoid

np.set_printoptions(suppress=True)
# How to find the anchor points? 
# Sampling the multiplication of two gaussians

def find_most_similar_eig(matrix, vector):
    dot_products = matrix.T.dot(vector)
    matrix_column_norms = np.linalg.norm(matrix, axis=0)
    vector_norm = np.linalg.norm(vector)
    
    similarities = dot_products / (matrix_column_norms * vector_norm)
    # print(similarities)

    max_abs_similarity_col = np.argmax(np.abs(similarities))
    return max_abs_similarity_col

def get_product_gaussian(mu1, mu2, sigma1, sigma2):
    combine_sigma = np.linalg.inv(np.linalg.inv(sigma1) + np.linalg.inv(sigma2))
    max_pos = combine_sigma @ (np.linalg.inv(sigma1) @ mu1 + np.linalg.inv(sigma2) @ mu2)
    max_pos = max_pos.flatten()

    return max_pos, combine_sigma

def get_joints(mu, sigma, start, end, dim):

    if dim == 2:

        jnt_arr = []

        for i in range(len(mu)-1):
            # Plot
            p_1 = np.zeros((plot_sample, plot_sample))
            p_2 = np.zeros((plot_sample, plot_sample))

            idx1 = i
            idx2 = idx1+1

            # Plot of correlated Normals
            bivariate_mean1 = mu[idx1].reshape(-1,1)  # Mean
            bivariate_covariance1 = sigma[idx1]  # Covariance
            #x1, x2, p_1 = generate_surface(bivariate_mean1, bivariate_covariance1, 2)

            # Plot of correlated Normals
            bivariate_mean2 = mu[idx2].reshape(-1,1)  # Mean
            bivariate_covariance2 = sigma[idx2]  # Covariance
            #x1, x2, p_2 = generate_surface(bivariate_mean2, bivariate_covariance2, 2)

            combine_sigma = np.linalg.inv(np.linalg.inv(bivariate_covariance1) + np.linalg.inv(bivariate_covariance2))
            max_pos = combine_sigma @ (np.linalg.inv(bivariate_covariance1) @ bivariate_mean1 + np.linalg.inv(bivariate_covariance2) @ bivariate_mean2)
            max_pos = max_pos.flatten()

            jnt_arr.append(max_pos)

        jnt_arr.insert(0, start)
        jnt_arr.append(end)
        jnt_arr = np.array(jnt_arr)

        return np.array(jnt_arr)

    else:

        jnt_arr = []
        resol = 30

        for i in range(len(mu)-1):

            idx1 = i
            idx2 = idx1+1

            # Plot of correlated Normals
            bivariate_mean1 = mu[idx1].reshape(-1,1)  # Mean
            bivariate_covariance1 = sigma[idx1]  # Covariance
            #x1,x2,x3,p_1 = generate_surface_3d(bivariate_mean1, bivariate_covariance1, 3)

            # Plot of correlated Normals
            bivariate_mean2 = mu[idx2].reshape(-1,1)  # Mean
            bivariate_covariance2 = sigma[idx2]  # Covariance
            #x1,x2,x3,p_2 = generate_surface_3d(bivariate_mean2, bivariate_covariance2, 3)

            combine_sigma = np.linalg.inv(np.linalg.inv(bivariate_covariance1) + np.linalg.inv(bivariate_covariance2))
            max_pos = combine_sigma @ (np.linalg.inv(bivariate_covariance1) @ bivariate_mean1 + np.linalg.inv(bivariate_covariance2) @ bivariate_mean2)
            max_pos = max_pos.flatten()
            jnt_arr.append(max_pos)


        jnt_arr.insert(0, start)
        jnt_arr.append(end)
        jnt_arr = np.array(jnt_arr)
        # print(jnt_arr)


        return np.array(jnt_arr)


