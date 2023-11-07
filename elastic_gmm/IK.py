import numpy as np
from elastic_gmm.gaussian_kinematics import create_transform_azi, create_transform_ele
import cvxpy as cp
import matplotlib.pyplot as plt


def solveIK(anchor_arr, target_start_T, target_end_T, traj_dis) -> None:
    anchor_arr = anchor_arr

    dim = anchor_arr.shape[1]

    constraints = []
    constraints_idx = []

    constraints.append(anchor_arr[0])
    constraints_idx.append(0)

    scale_ratio = 1.0
    if target_start_T is not None and target_end_T is not None:
        scale_ratio = (np.linalg.norm(target_start_T[0:dim, -1] -target_end_T[0:dim, 0]))/(traj_dis)

    if target_start_T is not None:

        # scale_ratio = (np.linalg.norm(target_start_T[0:dim, -1] - anchor_arr[-1]))/(traj_dis)
        # print("############################################")
        # print("scale_ratio!!!!!!!!!!", scale_ratio)
        # print("start pos", target_start_T[0:dim, -1])
        # print("last pos")
        # print("new dis", np.linalg.norm(target_start_T[0:dim, -1] - anchor_arr[-1]))
        # print("traj dis", traj_dis)
        # print("############################################")

        constraints[-1] = target_start_T[0:dim, -1]

        start_vec_dir = (target_start_T[0:dim, 0] / np.linalg.norm(target_start_T[0:dim, 0]))
        start_vec = start_vec_dir * np.linalg.norm(anchor_arr[1] - anchor_arr[0]) * scale_ratio
        
        leave_pt = constraints[-1] + start_vec
        constraints.append(leave_pt)
        constraints_idx.append(1)


    end_pt = anchor_arr[-1]
    if target_end_T is not None:
        end_pt = target_end_T[0:dim, -1]

        end_vec_dir = - (target_end_T[0:dim, 0] / np.linalg.norm(target_end_T[0:dim, 0]))

        reversed_end_vec = end_vec_dir * np.linalg.norm(anchor_arr[-1] - anchor_arr[-2]) * scale_ratio
        
        approach_pt = end_pt + reversed_end_vec
        constraints.append(approach_pt)
        constraints_idx.append(-2)

    constraints.append(end_pt)
    constraints_idx.append(-1)

    print('# of constraints', len(constraints))
    print('constraint index', constraints_idx)


    LTE = LaplacianEdit(anchor_arr)

    Ps = LTE.get_modified_traj_cvxpy(constraints, constraints_idx)

    return Ps
    

def solveTraj(new_anchor, dt):
    total_step = 100
    init_traj = np.linspace(new_anchor[0], new_anchor[-1], total_step)

    force_last_segment = (new_anchor[-1] + new_anchor[-2]) / 2

    
    new_anchor = np.insert(new_anchor, -1, force_last_segment, axis=0)

    diffs = np.diff(new_anchor, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    total_distance = distances.sum()
    accumulated_distances = np.cumsum(distances)
    progress = accumulated_distances / total_distance


    step_idx = np.array((total_step-1) * progress, dtype=int)

    #fix start and end direction
    if 1 not in step_idx:
        start_dir = (new_anchor[1] - new_anchor[0]) / np.linalg.norm(new_anchor[1] - new_anchor[0])
        start_mag = np.linalg.norm(init_traj[1] - init_traj[0])
        start_vec = start_mag * start_dir
        new_anchor = np.insert(new_anchor, 1, init_traj[0] + start_vec, axis=0)
        step_idx = np.insert(step_idx, 0, 1)

    if total_step - 2 not in step_idx:
        end_dir = (new_anchor[-2] - new_anchor[-1]) / np.linalg.norm(new_anchor[-2] - new_anchor[-1])
        end_mag = np.linalg.norm(init_traj[-2] - init_traj[-1])
        end_vec = end_mag * end_dir
        new_anchor = np.insert(new_anchor, -1, init_traj[-1] + end_vec, axis=0)
        step_idx = np.insert(step_idx, -1, total_step-2)


    #fix start position
    step_idx = np.insert(step_idx, 0, 0)

    LTE = LaplacianEdit(init_traj)
    print("edited traj")
    print(new_anchor)
    print(step_idx)
    edited_traj = LTE.get_modified_traj_cvxpy(new_anchor, step_idx)

    edited_dot_traj = np.diff(edited_traj, axis=0) / dt

    # fig, ax = plt.subplots(2,1)
    # ax[0].plot(np.arange(len(edited_dot_traj[:,0])), edited_dot_traj[:,0])
    # ax[1].plot(np.arange(len(edited_dot_traj[:,0])), edited_dot_traj[:,1])
    # plt.show()

    # plt.scatter(new_anchor[:,0], new_anchor[:,1])
    # plt.plot(edited_traj[:,0], edited_traj[:,1])
    # plt.show()

    return edited_traj, edited_dot_traj


def solveTrajDense(new_anchor, via_idxs, dt):
    def interpolate_waypoints(waypoints, indices, num_points_between):

        for i in indices:

            if i < len(waypoints) - 1 and i > 0:
                curr = waypoints[i]
                prev = waypoints[i-1]

                next = waypoints[i+1]
                prev_inter_pts = np.linspace(prev, curr, num_points_between+2)[1:-1]

                next_inter_pts = np.linspace(curr, next, num_points_between+2)[1:-1]

                waypoints = np.insert(waypoints, i, prev_inter_pts, axis=0)
                waypoints = np.insert(waypoints, i+1+num_points_between, next_inter_pts, axis=0)

            # elif i == 0:
            #     curr = waypoints[i]
            #     next = waypoints[i+1]
            #     next_inter_pts = np.linspace(curr, next, num_points_between+2)[1:-1]
            #     waypoints = np.insert(waypoints, 1, next_inter_pts, axis=0)

            # elif i == len(waypoints) - 1:
            #     curr = waypoints[i]
            #     prev = waypoints[i-1]
            #     prev_inter_pts = np.linspace(prev, curr, num_points_between+2)[1:-1]
            #     waypoints = np.insert(waypoints, i-1, prev_inter_pts, axis=0)


        return waypoints

    
    total_step = 100
    init_traj = np.linspace(new_anchor[0], new_anchor[-1], total_step)


    new_anchor = interpolate_waypoints(np.copy(new_anchor), via_idxs, 2)

    diffs = np.diff(new_anchor, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    total_distance = distances.sum()
    accumulated_distances = np.cumsum(distances)
    progress = accumulated_distances / total_distance


    step_idx = np.array((total_step-1) * progress, dtype=int)

    #fix start and end direction
    if 1 not in step_idx:
        start_dir = (new_anchor[1] - new_anchor[0]) / np.linalg.norm(new_anchor[1] - new_anchor[0])
        start_mag = np.linalg.norm(init_traj[1] - init_traj[0])
        start_vec = start_mag * start_dir
        new_anchor = np.insert(new_anchor, 1, init_traj[0] + start_vec, axis=0)
        step_idx = np.insert(step_idx, 0, 1)

    if total_step - 2 not in step_idx:
        end_dir = (new_anchor[-2] - new_anchor[-1]) / np.linalg.norm(new_anchor[-2] - new_anchor[-1])
        end_mag = np.linalg.norm(init_traj[-2] - init_traj[-1])
        end_vec = end_mag * end_dir
        new_anchor = np.insert(new_anchor, -1, init_traj[-1] + end_vec, axis=0)
        step_idx = np.insert(step_idx, -1, total_step-2)


    #fix start position
    step_idx = np.insert(step_idx, 0, 0)

    LTE = LaplacianEdit(init_traj)
    print("edited traj")
    print(new_anchor)
    print(step_idx)
    edited_traj = LTE.get_modified_traj_cvxpy(new_anchor, step_idx)

    edited_dot_traj = np.diff(edited_traj, axis=0) / dt

    # fig, ax = plt.subplots(2,1)
    # ax[0].plot(np.arange(len(edited_dot_traj[:,0])), edited_dot_traj[:,0])
    # ax[1].plot(np.arange(len(edited_dot_traj[:,0])), edited_dot_traj[:,1])
    # plt.show()

    # plt.scatter(new_anchor[:,0], new_anchor[:,1])
    # plt.plot(edited_traj[:,0], edited_traj[:,1])
    # plt.show()

    return edited_traj, edited_dot_traj

class LaplacianEdit:
    def __init__(self, traj, end_point_fix=True) -> None:
        self.P = traj.copy()
        self.L = self.construct_L(traj)
        self.delta = self.L @ self.P
        self.C = traj.copy()
        self.P_bar = np.zeros(self.L.shape)
        if end_point_fix:
            self.P_bar[0,0] = 1.0
            self.P_bar[-1,-1] = 1.0

    def get_weights(self, type='uniform'):
        return 1.0

    def construct_L(self, traj):
        L = np.eye(len(traj))
        for i in range(len(traj)):
            #There are only two neighbors for a point on the trajectory
            if i != 0:
                j1 = i - 1
                L[i, j1] = -self.get_weights() / (2*self.get_weights())
            else:
                L[i, 1] = -self.get_weights() / self.get_weights()

            if i != len(traj) - 1:
                j2 = i + 1
                L[i, j2] = -self.get_weights() / (2*self.get_weights())
            else:
                L[i, -2] = -self.get_weights() / self.get_weights()
        return L
    
    
    def get_modified_traj_cvxpy(self, constr, constr_idx):
        x = cp.Variable(self.P.shape)
        A = np.vstack((self.L, self.P_bar))
        B = np.vstack((self.delta, self.C))
        objective = cp.Minimize(cp.sum_squares(self.L@x - self.delta))
        
        constraints = []
        for i in range(len(constr)):
            constraints.append(x[constr_idx[i]] == constr[i])

        prob = cp.Problem(objective, constraints)
        result = prob.solve(verbose=True)
        # print("The optimal value is", result)
        # print("The optimal x is")
        # print(x.value)
        return x.value