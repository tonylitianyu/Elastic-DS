import numpy as np
from lpv_ds_a.utils_ds.pca_related.my_pca import my_pca
from lpv_ds_a.utils_ds.pca_related.project_pca import project_pca
from lpv_ds_a.utils_ds.pca_related.reconstruct_pca import reconstruct_pca


def sample_initial_points(x0_all, nb_points, type, plot_volumn):
    # Auxiliary Variable
    dim = x0_all.shape[0]

    # Output Variable
    init_points = np.zeros((dim, nb_points))

    # Estimate point distribution
    V, D, init_mu = my_pca(x0_all)
    D[0][0] = 1.5 * D[0][0] # extend the -smallest- dimension
    D[2][2] = 5 * D[2][2] # extend the -smallest- dim # WHAT?

    A_y, y0_all = project_pca(x0_all, init_mu, V, dim)
    Ymin_values = np.min(y0_all, axis=1)
    Ymin_values = Ymin_values + np.array([0, 0.5*Ymin_values[1], 0.5*Ymin_values[2]])
    Yrange_values = np.max(y0_all, axis=1, keepdims=True) - np.min(y0_all, axis=1, keepdims=True)
    Yrange_values[1] += 0.5 * Yrange_values[1]
    Yrange_values[2] += 0.5 * Yrange_values[2]
    init_points_y = np.tile(Ymin_values.reshape(dim, 1), nb_points) + np.random.rand(dim, nb_points) * np.tile(Yrange_values,nb_points)

    init_points = reconstruct_pca(init_points_y, A_y, init_mu)

    return init_points



