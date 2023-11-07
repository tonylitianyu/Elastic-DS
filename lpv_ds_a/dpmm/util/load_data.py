from __future__ import print_function
from matplotlib.widgets import Button
import matplotlib.pyplot as plt
from matplotlib import rc
from lpv_ds_a.dpmm.util.mousetrajectory_gui import MouseTrajectory
import lpv_ds_a.dpmm.util.mousetrajectory_gui as mt
from scipy.io import loadmat
import numpy as np
import csv



# rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text')

'''
 Brings up an empty world environment with the drawn trajectories using MouseTrajectory GUI
'''


def draw_data():
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    points, = ax.plot([], [], 'ro', markersize=2, lw=2)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('$x_1$', fontsize=15)
    plt.ylabel('$x_2$', fontsize=15)
    plt.title('Draw trajectories to learn a motion policy:', fontsize=15)

    # Add UI buttons for data/figure manipulation
    store_btn = plt.axes([0.67, 0.05, 0.075, 0.05])
    clear_btn = plt.axes([0.78, 0.05, 0.075, 0.05])
    snap_btn = plt.axes([0.15, 0.05, 0.075, 0.05])
    bstore = Button(store_btn, 'store')
    bclear = Button(clear_btn, 'clear')
    bsnap = Button(snap_btn, 'snap')

    # Calling class to draw data on top of environment
    indexing = 2  # Set to 1 if you want the snaps/data to be indexed with current time-stamp
    store_mat = 0  # Set to 1 if you want to store data in .mat structure for MATLAB
    draw_data = MouseTrajectory(points, indexing, store_mat)
    draw_data.connect()
    bstore.on_clicked(draw_data.store_callback)
    bclear.on_clicked(draw_data.clear_callback)
    bsnap.on_clicked(draw_data.snap_callback)

    # Show
    plt.show()


def load_data(data_name="human_demonstrated_trajectories_1.dat"):
    # Create figure/environment to draw trajectories on
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('$x_1$', fontsize=15)
    plt.ylabel('$x_2$', fontsize=15)
    plt.title('Trajectories drawn by human with mouse GUI:', fontsize=15)

    # Load trajectories from file and plot
    file_name = './data/' + data_name
    l, t, x, y = mt.load_trajectories(file_name)
    ax.plot(x, y, 'ro', markersize=2, lw=2)
    return l, t, x, y


def load_matlab_data(pkg_dir, dataset, sub_sample, nb_trajectories):
    dataset_name = []
    if dataset == 1:
        dataset_name = r'2D_concentric.mat'
    elif dataset == 2:
        dataset_name = r'2D_opposing.mat'
    elif dataset == 3:
        dataset_name = r'2D_multiple.mat'
    elif dataset == 4:
        dataset_name = r'2D_snake.mat'
    elif dataset == 5:
        dataset_name = r'2D_messy-snake.mat'
    elif dataset == 6:
        dataset_name = r'2D_viapoint.mat'
    elif dataset == 7:
        dataset_name = r'2D_Lshape.mat'
    elif dataset == 8:
        dataset_name = r'2D_Ashape.mat'
    elif dataset == 9:
        dataset_name = r'2D_Sshape.mat'
    elif dataset == 10:
        dataset_name = r'2D_multi-behavior.mat'
    elif dataset == 11:
        dataset_name = r'3D_viapoint_2.mat'
        traj_ids = [1, 2]
    elif dataset == 12:
        dataset_name = r'3D_sink.mat'
    elif dataset == 13:
        dataset_name = r'3D_bumpy-snake.mat'

    final_dir = pkg_dir  + dataset_name
    # print(final_dir)

    if dataset <= 6:
        Data = loadmat(final_dir)['Data']
    elif dataset <= 10:
        data_ = loadmat(r"{}".format(final_dir))
        data = np.array(data_["data"])
        N = len(data[0])
        Data = data[0][0]
        for n in np.arange(1, N):
            data_ = np.array(data[0][n])
            Data = np.concatenate((Data, data_), axis=1)
    else:
        data_ = loadmat(r"{}".format(final_dir))
        data_ = np.array(data_["data"])
        N = len(data_)
        traj = np.random.choice(np.arange(N), nb_trajectories, replace=False)
        data = data_[traj]
        for l in np.arange(nb_trajectories):
            data[l][0] = data[l][0][:, ::sub_sample]
            if l == 0:
                Data = data[l][0]
            else:
                Data = np.concatenate((Data, data[l][0]), axis=1)

    # Load trajectories from file and plot
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('$x_1$', fontsize=15)
    plt.ylabel('$x_2$', fontsize=15)
    plt.title('Trajectories drawn by human with mouse GUI:', fontsize=15)
    ax.plot(Data[0, :], Data[1, :], 'ro', markersize=2, lw=2)

    return Data[:, np.arange(0, Data.shape[1], sub_sample)]

