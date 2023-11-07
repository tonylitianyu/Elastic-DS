import numpy as np
import scipy
import os
import pickle
import matplotlib.pyplot as plt


def convert_mat_p(pkg_dir, task_name, task_idx, dim):
    final_dir = pkg_dir + "/" + task_name + "/" + str(task_idx) + ".mat"
    data_ = scipy.io.loadmat(r"{}".format(final_dir))
    if dim == 2:
        data_ = np.array(data_["data"])
    else:
        data_ = np.array(data_["data_ee_pose"])
    data = data_.reshape((data_.shape[1], 1))
    pickle.dump(data, open(pkg_dir + "/" + task_name + "/" + str(task_idx) + ".p", 'wb'))

if __name__ == "__main__":
    task_name = 'via2d_long'
    task_idx  = 'all_2d'
    pkg_dir = os.getcwd()
    convert_mat_p(pkg_dir, task_name, task_idx, 2)