import numpy as np


class ds_gmms:
    Mu = None
    Sigma = None
    Priors = None


class Vxf0_struct:
    d = 0
    w = 0
    L = 0
    Mu = None
    Priors = None
    P = None
    SOS = False


class options_struct:
    tol_mat_bias = 0
    display = 1
    tol_stopping = 0
    max_iter = 0
    optimizePriors = False
    upperBoundEigenValue = True


class Vxf_struct:
    d = 0
    w = 0
    L = 0
    Mu = None
    Priors = None
    P = None


class ds_plot_options:
    sim_traj = 1
    x0_all = None
    init_type = 'cube'
    nb_points = 10
    plot_vol = 0
    limits = None
    dimensions = None
    attractor = None


class Opt_Sim:
    dt = None
    i_max = None
    tol = 0.001
    plot = 0

