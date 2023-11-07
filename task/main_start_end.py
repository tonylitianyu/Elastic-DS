#python3 -m task.main_start_end
import numpy as np
import os, sys
from os.path import exists
import pickle
import matplotlib.pyplot as plt
import time

# import transfer
from elastic_gmm.split_traj import split_traj
from elastic_gmm.gaussian_kinematics import create_transform_azi
from elastic_gmm.generate_transfer import start_adapting
from utils.pipeline_func import get_gmm, get_ds
from utils.plotting import plot_full_func
from lpv_ds_a.utils_ds.structures import ds_gmms

geo_test_arr = [
    [(0.1167, 0.4660, -np.pi/4), (0.8718, 0.4733, np.pi/4)],
    [(0.1167, 0.2660, -np.pi/4), (0.8718, 0.6733, np.pi/4)],
    [(0.2, 0.85, -np.pi/4), (0.7, 0.35, np.pi/4)]
]
#for p in range(len(plot_array)):
p = 2

task_name = 'startreach2d'
task_idx  = 'all_2d'
pkg_dir = os.getcwd()
final_dir = pkg_dir + "/task_dataset/" + task_name + "/" + str(task_idx) + ".p"
loaded_data = pickle.load(open(final_dir, 'rb'))
demo_geo = loaded_data['geo']
data = loaded_data['traj']
data_drawn = data.copy()

# Task parametrization
split_pts = np.array([])
geo_config = geo_test_arr[p]

O_s = np.array([create_transform_azi(np.array(geo_config[0][:2]), geo_config[0][2]), 
                create_transform_azi(np.array(geo_config[1][:2]), geo_config[1][2])])

segment_traj = split_traj(data, split_pts, 2)

first_segment_data = segment_traj[0]

old_gmm_struct = get_gmm(first_segment_data)

# For original ds
old_ds_struct = get_ds(old_gmm_struct, first_segment_data[0], None, demo_geo)

# Transform the Gaussians
traj_batch = segment_traj[0]
traj_data, pi, mean_arr, cov_arr, joint_arr = start_adapting(old_gmm_struct, traj_batch, O_s[0], O_s[1])

gmm_struct = ds_gmms()
gmm_struct.Mu = mean_arr.T
gmm_struct.Priors = pi
gmm_struct.Sigma = cov_arr

# Estimate a new motion policy
ds_struct = get_ds(gmm_struct, traj_data, joint_arr, geo_config)

# Plot
plot_full_func(ds_struct, old_ds_struct)

# if p == -1:
#     plot_DS_GMM(task_name, 0, gate=[(0.1167,0.6660,-3*np.pi/4-0.28), (0.8718, 0.6733, 3*np.pi/4)], saveplot=False, save_idx=p+1, task_sat=False)
# else:
#     plot_DS_GMM(task_name, 0, gate=[(plot_array[p][0][0],plot_array[p][0][1],plot_array[p][1]-np.pi/2), (plot_array[p][2][0],plot_array[p][2][1],plot_array[p][3]+np.pi/2)], saveplot=False, save_idx=p+1, task_sat=False)