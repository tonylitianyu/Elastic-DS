import numpy as np
from scipy.signal import savgol_filter


def normalize_data(data):
    return np.divide(data - np.mean(data, axis=0), np.sqrt(np.diag(np.cov(data.T))))


def normalize_velocity_vector(data):
    if data.shape[0]==4:
        vel_data = data[2:4, :]
        vel_norm = np.linalg.norm(vel_data, axis=0)
        normalized_vel_data = np.divide(vel_data, vel_norm)
        return np.hstack((data[0:2, :].T, normalized_vel_data.T))
    elif data.shape[0]==6:
        vel_data = data[3:6, :]
        vel_norm = np.linalg.norm(vel_data, axis=0)
        normalized_vel_data = np.divide(vel_data, vel_norm)
        return np.hstack((data[0:3, :].T, normalized_vel_data.T))



def add_directional_features(line_index, time_index, x_coord, y_coord, if_normalize):
    for l_index in range(max(line_index) + 1):
        entry_index = (np.array(line_index) == l_index)
        pos_data = np.hstack((np.array(x_coord)[entry_index].reshape(-1, 1), np.array(y_coord)[entry_index].reshape(-1, 1)))

        entry = np.ones((pos_data.shape[0]), dtype=int)
        for index in np.arange(1, pos_data.shape[0]):
            if np.all(pos_data[index, :] == pos_data[index-1, :]):
                entry[index] = 0
        pos_data = pos_data[entry == 1, :]
        
        # pos_data[0, :] = savgol_filter(pos_data[0, :], 15, 1, mode='nearest')
        # pos_data[1, :] = savgol_filter(pos_data[1, :], 15, 1, mode='nearest')
        pos_data= savgol_filter(pos_data, 15, 1, mode='nearest')


        vel_data = np.zeros((pos_data.shape[0], 2))
        for index in np.arange(0, vel_data.shape[0]-1):
            vel_data[index, :] = (pos_data[index+1, :] - pos_data[index, :])
        vel_data[-1, :] = vel_data[-2, :]

        # vel_data = savgol_filter(vel_data, 15, 3, mode='nearest')

        if if_normalize:
            vel_data_norm = np.linalg.norm(vel_data, axis=1)
            vel_data = np.divide(vel_data, np.hstack((vel_data_norm.reshape(-1, 1), vel_data_norm.reshape(-1, 1))))

        # print(vel_data)
        #
        # if if_normalize:
        #     vel_data_norm = np.linalg.norm(vel_data, axis=1)
        #     vel_data = np.divide(vel_data, np.hstack((vel_data_norm.reshape(-1, 1), vel_data_norm.reshape(-1, 1))))

        if l_index == 0:
            pos_vel_data = np.hstack((pos_data, vel_data))
        else:
            pos_vel_data = np.vstack((pos_vel_data, np.hstack((pos_data, vel_data))))
    return pos_vel_data
