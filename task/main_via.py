#python3 -m task.main_start_end
import numpy as np
import os, sys
from os.path import exists
import pickle
import matplotlib.pyplot as plt
import time
import copy

# import transfer
from elastic_gmm.split_traj import split_traj
from elastic_gmm.gaussian_kinematics import create_transform_azi
from elastic_gmm.generate_transfer import start_adapting, generate_continuous_traj
from utils.pipeline_func import get_gmm, get_ds, combine_gmms
from utils.plotting import plot_full_func
from lpv_ds_a.utils_ds.structures import ds_gmms

geo_test_arr = [
    (0.4, 0.4, -np.pi/4),
    (0.7, 0.7, -np.pi/2)
]
p = 1

task_name = 'via2d'
task_idx  = 'all_2d'
pkg_dir = os.getcwd()
final_dir = pkg_dir + "/task_dataset/" + task_name + "/" + str(task_idx) + ".p"
loaded_data = pickle.load(open(final_dir, 'rb'))
demo_geo = loaded_data['geo']
data = loaded_data['traj']
data_drawn = data.copy()

split_pts = np.array([[0.5, 0.5]])
segment_traj = split_traj(data, split_pts, dim=2)

new_via_pts = np.array([geo_test_arr[p][:2]])
tunnel_rot = np.array([geo_test_arr[p][2]])
O_s = np.array([[None, create_transform_azi(new_via_pts[0], tunnel_rot[0])],
                [create_transform_azi(new_via_pts[0], tunnel_rot[0]), None]])

old_gmm_struct_arr = []
old_traj_batch_arr = []
new_gmm_struct_arr = []
new_traj_batch_arr = []

joint_arr_arr = []
via_arr = [0]

for m in range(len(segment_traj)):
    first_segment_data = segment_traj[m]
    old_gmm_struct = get_gmm(first_segment_data)

    old_gmm_struct_arr.append(copy.deepcopy(old_gmm_struct))
    old_traj_batch_arr.append(copy.deepcopy(first_segment_data))

    traj_batch = segment_traj[m]
    traj_data, pi, mean_arr, cov_arr, joint_arr = start_adapting(old_gmm_struct, traj_batch, O_s[m][0], O_s[m][1])

    new_gmm_struct = ds_gmms()
    new_gmm_struct.Mu = mean_arr.T
    new_gmm_struct.Priors = pi
    new_gmm_struct.Sigma = cov_arr

    new_gmm_struct_arr.append(new_gmm_struct)

    if m == len(segment_traj) - 1:
        joint_arr_arr.append(joint_arr)
    else:
        joint_arr_arr.append(joint_arr[:-1])
        via_arr.append(via_arr[-1] + len(joint_arr) - 1)


# Estimate the original motion policy

all_pi, all_mu, all_sigma, _ = combine_gmms(old_gmm_struct_arr, None)

final_gmm_struct = ds_gmms()
final_gmm_struct.Mu = all_mu.T
final_gmm_struct.Priors = all_pi
final_gmm_struct.Sigma = all_sigma

old_ds_struct = get_ds(final_gmm_struct, data[0], None, demo_geo[0])


# Estimate a new motion policy

all_pi, all_mu, all_sigma, joint = combine_gmms(new_gmm_struct_arr, joint_arr_arr)

traj_data = generate_continuous_traj(joint, data, via_arr)

final_gmm_struct = ds_gmms()
final_gmm_struct.Mu = all_mu.T
final_gmm_struct.Priors = all_pi
final_gmm_struct.Sigma = all_sigma

new_ds_struct = get_ds(final_gmm_struct, traj_data, joint, geo_test_arr[p])

plot_full_func(new_ds_struct, old_ds_struct, tunnel=True)