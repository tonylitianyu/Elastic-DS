import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def find_closest_point_indices(trajectory, points):
    closest_point_indices = []
    for point in points:
        distances = [euclidean_distance(trajectory[:, idx], point) for idx in range(trajectory.shape[1])]
        closest_point_idx = np.argmin(distances)
        closest_point_indices.append(closest_point_idx)
    return sorted(closest_point_indices)

def separate_trajectory(trajectory, time_list, points,):
    closest_point_indices = find_closest_point_indices(trajectory, points)

    time_traj = time_list.reshape(len(time_list),-1)
    trajectory = np.vstack((trajectory, time_traj))

    segments = []

    if len(closest_point_indices) == 0:
        segments.append(trajectory)
    else:

        for idx, point_idx in enumerate(closest_point_indices):
            if idx == 0:
                segments.append(trajectory[:, :point_idx])
            else:
                segments.append(trajectory[:, closest_point_indices[idx-1]:point_idx])
        
        segments.append(trajectory[:, closest_point_indices[-1]:])
    
    return segments

def split_traj(full_traj, split_points, dim): # split points are from tracking module
    expected_segment = len(split_points) + 1

    n_traj = len(full_traj)

    if dim == 3:
        sample_step = 10
        vel_thresh = 3e-3

    save_segment_arr = [[] for _ in range(expected_segment)]
    for i in range(n_traj):
        if dim == 2:
            pos_traj = full_traj[i][0][:dim]
            vel_traj = full_traj[i][0][dim:]
            #np.save('demonstration.npy', np.vstack((pos_traj, vel_traj)))
            segment_traj = separate_trajectory(pos_traj, vel_traj, split_points)
        else:
            #fix 3d later
            pos_traj = full_traj[i][:3, ::sample_step]
            time_traj = full_traj[i][-1, ::sample_step].reshape(1,-1)

            # raw_diff_pos = np.diff(pos_traj)
            # vel_mag = np.linalg.norm(raw_diff_pos, axis=0).flatten()
            # first_non_zero_index = np.argmax(vel_mag > vel_thresh)
            # last_non_zero_index = len(vel_mag) - 1 - np.argmax(vel_mag[::-1] > vel_thresh)

            # if first_non_zero_index >= last_non_zero_index:
            #     raise Exception("Sorry, vel are all zero")

            # pos_traj = pos_traj[:, first_non_zero_index:last_non_zero_index]
            # time_traj = time_traj[:, first_non_zero_index:last_non_zero_index]

            # pos_diff_traj = np.diff(pos_traj)
            # time_diff_traj = np.diff(time_traj)

            # vel_traj = pos_diff_traj / time_diff_traj
            # segment_traj = separate_trajectory(pos_traj[:,:-1], vel_traj, split_points)
            segment_traj = separate_trajectory(pos_traj, time_traj, split_points)
        
        print('Split into # of segments', len(segment_traj))

        for j in range(len(segment_traj)):
            save_segment_arr[j].append([segment_traj[j]])

    # for k in range(expected_segment):
    #     save_segment_arr[k] = np.hstack(save_segment_arr[k])


        # first_segment = segment_traj[0].T
        # second_segment = segment_traj[1].T
        # # # Creating figure
        # plt.scatter(first_segment[:,0], first_segment[:,1], color='blue')
        # plt.scatter(second_segment[:,0], second_segment[:,1], color='red')
        # plt.show()

    return save_segment_arr


if __name__ == "__main__":
    split_traj('test2d', np.array([[4.5, 0.0]]))
