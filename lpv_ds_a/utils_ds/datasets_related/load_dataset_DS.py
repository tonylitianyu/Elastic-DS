from scipy.io import loadmat
import numpy as np
from lpv_ds_a.utils_ds.datasets_related.processDataStructure import processDataStructure


def load_dataset_DS(pkg_dir, dataset, sub_sample, nb_trajectories):
    dataset_name = []
    if dataset == 1:
        dataset_name = r'2D_messy-snake.mat'
    elif dataset == 2:
        dataset_name = r'2D_Lshape.mat'
    elif dataset == 3:
        dataset_name = r'2D_Ashape.mat'
    elif dataset == 4:
        dataset_name = r'2D_Sshape.mat'
    elif dataset == 5:
        dataset_name = r'2D_multi-behavior.mat'
    elif dataset == 6:
        dataset_name = r'3D_viapoint_3.mat'
    elif dataset == 7:
        dataset_name = r'3D_sink.mat'
    elif dataset == 8:
        dataset_name = r'3D_Cshape_bottom.mat'
    elif dataset == 9:
        dataset_name = r'3D_Cshape_top.mat'
    elif dataset == 10:
        dataset_name = r'3D-pick-box.mat'
    elif dataset == 11:
        dataset_name = r'iCubHuman_demos.mat'

    if not sub_sample:
        sub_sample = 2

    final_dir = pkg_dir + "/datasets/" + dataset_name

    if dataset == 1:
        print("can not run in original matlab code, so this function we don't currently implement it")
        return None

    elif dataset <= 5:
        # 2022/09/10 检查出数据load错误
        data_ = loadmat(r"{}".format(final_dir))
        data_ = np.array(data_["data"])
        N = len(data_[0])
        data = data_.reshape((N, 1))
        Data, Data_sh, att, x0_all, dt, data, traj_length = processDataStructure(data)

    else:
        data_ = loadmat(r"{}".format(final_dir))
        data_ = np.array(data_["data"])
        N = len(data_)
        traj = np.random.choice(np.arange(N), nb_trajectories, replace=False)
        traj = np.array([6, 8, 3, 5]) - 1
        data = data_[traj]
        for l in np.arange(nb_trajectories):
            # Gather Data
            if dataset == 11:
                print('this should be fixed later')
            else:
                data[l][0] = data[l][0][:, ::sub_sample]
        Data, Data_sh, att, x0_all, dt, data, traj_length = processDataStructure(data)

    return Data, Data_sh, att, x0_all, data, dt, traj_length
