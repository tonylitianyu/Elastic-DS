import numpy as np
import os, sys
import copy
import json
import scipy
import pickle


# import utilities
from lpv_ds_a.utils_ds.datasets_related.load_dataset_DS import load_dataset_DS, processDataStructure
from lpv_ds_a.utils_ds.rearrange_clusters import rearrange_clusters
from lpv_ds_a.utils_ds.structures import ds_plot_options

from lpv_ds_a.dpmm.main import dpmm
from lpv_ds_a.utils_ds.structures import ds_gmms
from lpv_ds_a.ds_opt.main import ds_opt

class elastic_struct:
    gmm_struct : ds_gmms = None
    A_k = None
    b_k = None
    att = None
    joints = None
    dt = None
    demo_traj = None
    geo = None


def get_gmm(traj_segment):   #(N_traj, 1, 4 or 6)
    Data, Data_sh, att, x0_all, dt, data, traj_length = processDataStructure(copy.deepcopy(traj_segment))

    # Get Mu, Prior and Sigma
    # In here you could plug in any clustering algorithm you developed
    Priors, Mu, Sigma = dpmm(Data)
    # re-arrange the data
    gmm_struct = rearrange_clusters(Priors, Mu, Sigma, att)
    return gmm_struct

def get_ds(gmm, traj, joints, geo_descriptor):
    Data, Data_sh, att, x0_all, dt, data, traj_length = processDataStructure(copy.deepcopy([traj]))

    ds_opt_obj = ds_opt(Data, Data_sh, att, x0_all, dt, traj_length, \
                        len(gmm.Priors), 2, gmm.Priors, gmm.Mu.T, gmm.Sigma)
    A_k, b_k = ds_opt_obj.begin()

    es = elastic_struct()
    es.gmm_struct = gmm
    es.A_k = A_k
    es.b_k = b_k
    es.att = att
    es.joints = joints
    es.dt = dt
    es.demo_traj = [traj]
    es.geo = geo_descriptor

    return es

    #save_json(task_name, idx, os.getcwd(), gmm, A_k, b_k, att, joints, dt)

def combine_gmms(gmm_struct_arr, joint_arr):
    mu_arr = []
    pi_arr = []
    sigma_arr = []
    for gmm_struct in gmm_struct_arr:
        mu_arr.append(gmm_struct.Mu)
        pi_arr.append(gmm_struct.Priors)
        sigma_arr.append(gmm_struct.Sigma)

    mu = np.hstack(mu_arr).T
    pi = np.concatenate(pi_arr)
    sigma = np.vstack(sigma_arr)

    if joint_arr is not None:
        joint = np.vstack(joint_arr)
    else:
        joint = None

    return pi, mu, sigma, joint


def save_json(name, idx, pkg_dir, es : elastic_struct):
    gmm = es.gmm_struct
    A = es.A_k
    b = es.b_k
    att = es.att
    joints = es.joints
    dt = es.dt

    dict = {}
    dict["name"] = name
    dict["K"] = len(gmm.Priors)
    dict["M"] = len(att)
    dict["Priors"] = list((gmm.Priors / np.sum(gmm.Priors)).flatten())
    dict["Mu"] = list(gmm.Mu.T.flatten())
    dict["Sigma"] = list(gmm.Sigma.flatten())
    dict["A"] = list(A.flatten())
    dict["B"] = list(b.T.flatten())
    dict["attractor"] = list(att.flatten())
    dict["att_all"] = list(att.flatten())
    dict["dt"] = dt
    dict["gripper_open"] = 1
    if joints is not None:
        dict["joint"] = list(joints.flatten())

    json_object = json.dumps(dict, indent=4)
    target_dir = pkg_dir + "/task_dataset/" + name + "/model/"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    with open(target_dir + str(idx) +".json", "w") as outfile:
        outfile.write(json_object)