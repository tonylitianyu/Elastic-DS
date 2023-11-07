import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from elastic_gmm.gaussian_kinematics import GaussianKinematics, GaussianKinematics3D,\
    get_angles_between_two_2d, create_transform_azi, create_transform_ele, get_angles_between_two_3d
from elastic_gmm.find_joints import get_joints
from elastic_gmm.IK import solveIK, solveTraj, solveTrajDense



def start_adapting(gmm, traj, target_start_pose, target_end_pose, dt=None):
    mu = gmm.Mu.T#(gmm['ds_gmm'][0][0][0].T)
    dim = mu.shape[1]
    sigma = gmm.Sigma
    pi = gmm.Priors


    start_point = []
    end_point = []
    for k in range(dim):
        #get xyz start point and end point
        start_point.append(sum([sum(traj[i][0][k,:5])/5 for i in range(len(traj))]) / len(traj))
        end_point.append(sum([sum(traj[i][0][k,-5:])/5 for i in range(len(traj))]) / len(traj))
    start_point = np.array(start_point)
    end_point = np.array(end_point)

    traj_dis = np.linalg.norm(end_point - start_point)

    anchor_arr = get_joints(mu, sigma, end_point, start_point, dim)[::-1]


    #reverse mu, sigma, and pi, go from the beginning to the end
    mu = mu[::-1]
    sigma = sigma[::-1]
    pi = pi[::-1]
    if dim == 2:
        gk = GaussianKinematics(pi, mu, sigma, anchor_arr)
    else:
        gk = GaussianKinematics3D(pi, mu, sigma, anchor_arr)

    show_original_traj(gk, traj)

    if dt is None:
        dt = get_dt(traj, dim)

    print('dt', dt)
    traj_data, mean_arr, cov_arr, new_anchor_point = generate_transfer_traj(gk, target_start_pose, target_end_pose, traj_dis, dt)
    
    
    
    return traj_data, pi, mean_arr, cov_arr, new_anchor_point

def show_original_traj(gk_var, original_traj):
    dim = gk_var.anchor_arr.shape[1]
    traj_bunch = [original_traj[i][0][:dim,:] for i in range(len(original_traj))]
    traj_bunch = np.hstack(traj_bunch)
    frame_arr, mean_arr, cov_arr  = gk_var.update_gaussian_transforms(gk_var.anchor_arr)
    #gk_var.plot(frame_arr, mean_arr, cov_arr, None, traj_bunch.T)

def get_dt(Data, dim):
    all_dt = []

    for trajectory in Data:
        traj = trajectory[0]
        temp_velocities = np.linalg.norm(traj[dim:,:], axis=0)
        normal_indices = np.where(np.abs(temp_velocities) > 1e-8)[0]
        
        positions = traj[:dim, normal_indices]
        velocities = traj[dim:, normal_indices]
        
        dt = np.abs(np.linalg.norm(positions[:,:-1] - positions[:,1:], axis=0) / np.linalg.norm(velocities[:,:-1], axis=0))
        all_dt.append(np.mean(dt))

    return np.mean(all_dt)


def generate_transfer_traj(gk_var, start_T, end_T, traj_dis, dt):
    dim = gk_var.anchor_arr.shape[1]

    new_anchor_point = solveIK(gk_var.anchor_arr, start_T, end_T, traj_dis)

    frame_arr, mean_arr, cov_arr = gk_var.update_gaussian_transforms(new_anchor_point)

    traj_data = []
    plot_traj, traj_dot_arr = solveTraj(np.copy(new_anchor_point), dt)
    pos_and_vel = np.vstack((plot_traj[1:].T, traj_dot_arr.T))
    if dim == 2:
        traj_data.append(pos_and_vel)
    else:
        traj_data.append(pos_and_vel)

    #gk_var.plot(frame_arr, mean_arr, cov_arr, None, plot_traj)

    return traj_data, mean_arr, cov_arr, new_anchor_point


def generate_continuous_traj(new_anchor_point, demo_traj, via_idxs):
    dim = new_anchor_point.shape[1]
    dt = get_dt(demo_traj, dim)
    traj_data = []
    plot_traj, traj_dot_arr = solveTrajDense(np.copy(new_anchor_point), via_idxs, dt)
    pos_and_vel = np.vstack((plot_traj[1:].T, traj_dot_arr.T))
    if dim == 2:
        traj_data.append(pos_and_vel)
    else:
        traj_data.append(pos_and_vel)
    return traj_data

