import numpy as np


def processDataStructure(data):
    N = int(len(data))
    M = int(len(data[0][0]) / 2)
    att_ = data[0][0][0:M, -1].reshape(M, 1)
    for n in np.arange(1, N):
        att = data[n][0][0:M, -1].reshape(M, 1)
        att_ = np.concatenate((att_, att), axis=1)

    att = np.mean(att_, axis=1, keepdims=True)
    shifts = att_ - np.repeat(att, N, axis=1)
    Data = np.array([])
    x0_all = np.array([])
    Data_sh = np.array([])
    traj_length = []
    for l in np.arange(N):
        # Gather Data
        data_ = data[l][0].copy()
        traj_length.append(data_.shape[1])
        shifts_ = np.repeat(shifts[:, l].reshape(len(shifts), 1), len(data_[0]), axis=1)
        data_[0:M, :] = data_[0:M, :] - shifts_
        data_[M:, -1] = 0
        data_[M:, -2] = (data_[M:, -1] + np.zeros(M)) / 2
        data_[M:, -3] = (data_[M:, -3] + data_[M:, -2])/2
        # All starting position for reproduction accuracy comparison
        if l == 0:
            Data = data_.copy()
            x0_all = np.copy(data_[0:M, 0].reshape(M, 1))
        else:
            Data = np.concatenate((Data, data_), axis=1)
            x0_all = np.concatenate((x0_all, data_[0:M, 0].reshape(M, 1)), axis=1)
        # Shift data to origin for Sina's approach + SEDS
        data_[0:M, :] = data_[0:M, :] - np.repeat(att, len(data_[0]), axis = 1)
        data_[M:, -1] = 0
        if l == 0:
            Data_sh = data_
        else:
            Data_sh = np.concatenate((Data_sh, data_), axis=1)
        data[l][0] = data_

    data_12 = data[0][0][:, 0:M]
    dt = np.abs((data_12[0][0] - data_12[0][1]) / data_12[M][0])
    return Data, Data_sh, att, x0_all, dt, data, np.array(traj_length)


