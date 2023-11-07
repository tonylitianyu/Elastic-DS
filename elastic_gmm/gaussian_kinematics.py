import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from elastic_gmm.gaussian2d import generate_surface, plot_sample
from elastic_gmm.gaussian3d import create_ellipsoid
# from find_joints import find_most_similar_eig

def create_transform_azi(trans, jnt_rot):
    if len(trans) == 2:
        rot = np.array([[np.cos(jnt_rot) , -np.sin(jnt_rot)],[np.sin(jnt_rot) , np.cos(jnt_rot)]]) 
        T_rot = np.hstack((rot, trans.reshape(-1,1)))
        T_rot = np.vstack((T_rot, np.array([0,0,1])))
        return T_rot
    else:
        rot = np.array([[np.cos(jnt_rot) , -np.sin(jnt_rot), 0.0],[np.sin(jnt_rot) , np.cos(jnt_rot), 0.0], [0.0, 0.0, 1.0]]) 
        T_rot = np.hstack((rot, trans.reshape(-1,1)))
        T_rot = np.vstack((T_rot, np.array([0,0,0,1])))
        return T_rot
    
def create_transform_ele(trans, jnt_rot):
    rot = np.array([[np.cos(jnt_rot), 0.0, np.sin(jnt_rot)], [0.0, 1.0, 0.0], [-np.sin(jnt_rot) , 0.0, np.cos(jnt_rot)]]) 
    T_rot = np.hstack((rot, trans.reshape(-1,1)))
    T_rot = np.vstack((T_rot, np.array([0,0,0,1])))
    return T_rot


def get_angles_between_two_2d(v1, v2):
    def angles_from_vector(vec):
        theta = np.arctan2(vec[1], vec[0])  # azimuth angle
        return theta
    vec1 = v1 / np.linalg.norm(v1)
    vec2 = v2 / np.linalg.norm(v2)
    theta1 = angles_from_vector(vec1)
    theta2 = angles_from_vector(vec2)        
    delta_theta = theta2 - theta1
    return delta_theta

def get_angles_between_two_3d(v1, v2):
    def angles_from_vector(vec):
        theta = np.arctan2(vec[1], vec[0])  # azimuth angle
        phi = np.arctan2(np.sqrt(vec[0]**2 + vec[1]**2), vec[2])  # elevation angle
        return theta, phi
    vec1 = v1 / np.linalg.norm(v1)
    vec2 = v2 / np.linalg.norm(v2)
    theta1, phi1 = angles_from_vector(vec1)
    theta2, phi2 = angles_from_vector(vec2)        
    delta_theta = theta2 - theta1
    delta_phi = phi2 - phi1
    return delta_theta, delta_phi



class GaussianKinematics:
    def __init__(self, pi, mu, sigma, anchor_arr, type=0) -> None:
        
        self.type = type  #type 0: end point free.   type 1: starting point free.   type 2: both ends free
        self.pi = pi
        self.mu = mu
        self.sigma = sigma

        self.anchor_arr = anchor_arr

                
        self.mean_transform_arr, self.cov_transform_arr, self.cov_eval_arr = self.calculate_gaussian_transforms()

        self.plot_sample = 200


    def T_multi_vec(self, T, vec):
        vec = vec.flatten()
        return (T @ np.append(vec, 1.0).reshape(-1,1)).flatten()[:2]

    def draw_frame_axis(self, T, ax):
        if ax is None:
            return
        
        x_axis = self.T_multi_vec(T, np.array([0.02,    0]))
        y_axis = self.T_multi_vec(T, np.array([0,    0.02]))

        center = self.T_multi_vec(T, np.array([0.0, 0.0]))
        stack_x = np.vstack((center, x_axis))
        stack_y = np.vstack((center, y_axis))

        ax.plot(stack_x[:,0], stack_x[:,1], color='red')
        ax.plot(stack_y[:,0], stack_y[:,1], color='green')


    def transform_from_curr_frame(self, curr_frame, curr_pts_in_world, target_pts_in_world):
        #1. get the next anchor point in the curr frame
        target_pts_in_curr = self.T_multi_vec(np.linalg.inv(curr_frame), target_pts_in_world)

        #2. get the angles from x-axis to the anchor
        azi = get_angles_between_two_2d(np.array([1.0,0.0]), target_pts_in_curr)

        #3. rotate
        pointing_T = create_transform_azi(np.array([0.0,0.0]),azi)

        #4. translate
        vec = target_pts_in_world - curr_pts_in_world
        curr_T = pointing_T @ create_transform_azi(np.array([np.linalg.norm(vec),0.0]), 0.0)

        return curr_T, azi, np.linalg.norm(vec), pointing_T
    
    def transform_mean_from_curr_frame(self, curr_frame, target_pts_in_world):
        target_pts_in_curr = self.T_multi_vec(np.linalg.inv(curr_frame), target_pts_in_world)
        curr_T = create_transform_azi(target_pts_in_curr, 0.0)
        return curr_T

    def transform_sigma_from_curr_frame(self, curr_frame, sigma_e_vec, center):
        sigma_e_vec = np.hstack((sigma_e_vec, center.reshape(-1,1)))
        sigma_e_vec = np.vstack((sigma_e_vec, np.array([0.0,0.0,1.0])))
        return np.linalg.inv(curr_frame) @ sigma_e_vec

    def calculate_gaussian_transforms(self, ax=None):
        mean_transform_arr = []
        cov_transform_arr = []
        cov_eval_arr = []

        #translate to starting point and orientation
        curr = self.anchor_arr[0]
        T = create_transform_azi(curr, 0.0)

        for i in range(len(self.mu)):
            curr_T, _, _, pointing_T = self.transform_from_curr_frame(T, self.anchor_arr[i], self.anchor_arr[i+1])
            
            #5. attach to the world frame
            #5.1 attach mu, change pointing first
            mean_T = self.transform_mean_from_curr_frame(T@pointing_T, self.mu[i])
            mean_transform_arr.append(mean_T)
            # T_mu = pointing_T @ mean_T
            # self.draw_frame_axis(T_mu,ax)

            #5.2 attch sigma
            e_val, e_vec = np.linalg.eig(self.sigma[i])
            sigma_T = self.transform_sigma_from_curr_frame(T@pointing_T, e_vec, self.mu[i])
            cov_transform_arr.append(sigma_T)
            cov_eval_arr.append(e_val)
            #self.draw_frame_axis(T_sigma, ax)

            #5.3 attach next anchor
            T = T @ curr_T
            self.draw_frame_axis(T,ax)

        return mean_transform_arr, cov_transform_arr, cov_eval_arr
    

    def get_mu_x_scale(self, new_pt, i1,i2):
        return np.linalg.norm(new_pt[i1] - new_pt[i2]) / np.linalg.norm(self.anchor_arr[i1] - self.anchor_arr[i2])

    def update_gaussian_transforms(self, waypt_arr):
        #translate to starting point and orientation
        curr = waypt_arr[0]
        frame_arr = [curr]
        mean_arr = []
        cov_arr = []

        T = create_transform_azi(curr, 0.0)

        for i in range(len(self.mu)):
            curr_T, curr_azi, curr_dis, pointing_T = self.transform_from_curr_frame(T, waypt_arr[i], waypt_arr[i+1])

            dis_scale = self.get_mu_x_scale(waypt_arr, i, i+1)

            self.mean_transform_arr[i][0, -1] *= dis_scale
            T_mu = T @ pointing_T @ self.mean_transform_arr[i]
            mean_arr.append(T_mu[:2,-1])

            self.cov_transform_arr[i][0, -1] *= dis_scale
            T_sigma = T @ pointing_T @ self.cov_transform_arr[i]
            new_e_vec = T_sigma[:2,:2]
            new_e_vec *= dis_scale
            cov_arr.append(new_e_vec @ np.diag(self.cov_eval_arr[i]) @ new_e_vec.T)



            #5.3 attach next anchor
            T = T @ curr_T

            frame_arr.append(T[:2, -1])

        return np.array(frame_arr), np.array(mean_arr), np.array(cov_arr)

            

    

    def plot(self, frame_arr, mean_arr, cov_arr, T_arr, traj_arr=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
        # if ax is None:
        #     return
        
        ax.set_aspect(1)

        ax.scatter(frame_arr[:,0], frame_arr[:,1], color='orange', label='joints')
        # ax.scatter(frame_arr[0,0], frame_arr[0,1], color='black', label='start')
        # ax.scatter(frame_arr[-1,0], frame_arr[-1,1], color='green', label='end')
        ax.scatter(mean_arr[:,0], mean_arr[:,1], color='red', s=2, label='means')

        # self.draw_frame_axis(np.eye(3), ax)

        # for i in range(len(T_arr)):
        #     self.draw_frame_axis(T_arr[i], ax)

        # for i in range(len(mean_arr)):
        #     e_val, e_vec = np.linalg.eig(cov_arr[i])
        #     sample_pts = create_ellipsoid(mean_arr[i], np.sqrt(e_val), e_vec)
        #     ax.plot_surface(sample_pts[0], sample_pts[1], sample_pts[2], color='blue', alpha=0.7)

        p_sum = np.zeros((plot_sample, plot_sample))
        for i in range(len(mean_arr)):
            x1, x2, p_curr = generate_surface(mean_arr[i].reshape(-1,1), cov_arr[i], 2)
            p_sum += p_curr

        step = 0.02
        m = np.amax(p_sum)
        levels = np.arange(0.0, m, step) + step
        ax.contourf(x1, x2, p_sum, zorder=0, alpha=1.0)


        if traj_arr is not None:
            ax.scatter(traj_arr[:,0], traj_arr[:,1], color='red', s=2)
        
        
        ax.legend()
        ax.set_xlim([0.0,1.0])
        ax.set_ylim([0.0,1.0])
        plt.show()

    def plot_original(self, traj_arr=None, ax=None):
        if traj_arr is not None and ax is not None:
            ax.set_aspect(1)
            ax.plot(traj_arr[:,0], traj_arr[:,1], color='brown', label='original')
            ax.scatter(traj_arr[0,0], traj_arr[0,1], color='black', label='start')
            ax.scatter(traj_arr[-1,0], traj_arr[-1,1], color='green', label='end')




class GaussianKinematics3D:
    def __init__(self, pi, mu, sigma, anchor_arr, type=0) -> None:
        
        self.type = type  #type 0: end point free.   type 1: starting point free.   type 2: both ends free
        self.pi = pi
        self.mu = mu
        self.sigma = sigma

        self.anchor_arr = anchor_arr

        self.mean_transform_arr, self.cov_transform_arr, self.cov_eval_arr = self.calculate_gaussian_transforms()

        self.plot_sample = 200


    def draw_frame_axis(self, T, ax):
        if ax is None:
            return
        
        x_axis = self.T_multi_vec(T, np.array([0.05,    0,    0]))
        y_axis = self.T_multi_vec(T, np.array([0,    0.05,    0]))
        z_axis = self.T_multi_vec(T, np.array([0,    0,    0.05]))

        center = self.T_multi_vec(T, np.array([0.0, 0.0, 0.0]))
        stack_x = np.vstack((center, x_axis))
        stack_y = np.vstack((center, y_axis))
        stack_z = np.vstack((center, z_axis))

        ax.plot(stack_x[:,0], stack_x[:,1], stack_x[:,2], color='red')
        ax.plot(stack_y[:,0], stack_y[:,1], stack_y[:,2], color='green')
        ax.plot(stack_z[:,0], stack_z[:,1], stack_z[:,2], color='blue')

    def T_multi_vec(self, T, vec):
        vec = vec.flatten()
        return (T @ np.append(vec, 1.0).reshape(-1,1)).flatten()[:3]

    def transform_from_curr_frame(self, curr_frame, curr_pts_in_world, target_pts_in_world):
        #1. get the next anchor point in the curr frame
        target_pts_in_curr = self.T_multi_vec(np.linalg.inv(curr_frame), target_pts_in_world)

        #2. get the angles from x-axis to the anchor
        azi, ele = get_angles_between_two_3d(np.array([1.0,0.0,0.0]), target_pts_in_curr)

        #3. rotate
        pointing_T = create_transform_azi(np.zeros(3), azi) @ create_transform_ele(np.zeros(3), ele)

        #4. translate
        vec = target_pts_in_world - curr_pts_in_world
        curr_T = pointing_T @ create_transform_azi(np.array([np.linalg.norm(vec),0.0,0.0]), 0.0)

        return curr_T, azi, ele, np.linalg.norm(vec), pointing_T
    
    def transform_mean_from_curr_frame(self, curr_frame, target_pts_in_world):
        target_pts_in_curr = self.T_multi_vec(np.linalg.inv(curr_frame), target_pts_in_world)
        curr_T = create_transform_azi(target_pts_in_curr, 0.0)
        return curr_T

    def transform_sigma_from_curr_frame(self, curr_frame, sigma_e_vec, center):
        sigma_e_vec = np.hstack((sigma_e_vec, center.reshape(-1,1)))
        sigma_e_vec = np.vstack((sigma_e_vec, np.array([0.0,0.0,0.0,1.0])))
        return np.linalg.inv(curr_frame) @ sigma_e_vec

    def calculate_gaussian_transforms(self, ax=None):
        mean_transform_arr = []
        cov_transform_arr = []
        cov_eval_arr = []

        #translate to starting point and orientation
        curr = self.anchor_arr[0]
        T = create_transform_azi(curr, 0.0)

        for i in range(len(self.mu)):
            curr_T, _, _, _, pointing_T = self.transform_from_curr_frame(T, self.anchor_arr[i], self.anchor_arr[i+1])
            
            #5. attach to the world frame
            #5.1 attach mu, change pointing first
            mean_T = self.transform_mean_from_curr_frame(T@pointing_T, self.mu[i])
            mean_transform_arr.append(mean_T)
            # T_mu = pointing_T @ mean_T
            # self.draw_frame_axis(T_mu,ax)

            #5.2 attch sigma
            e_val, e_vec = np.linalg.eig(self.sigma[i])
            sigma_T = self.transform_sigma_from_curr_frame(T@pointing_T, e_vec, self.mu[i])
            cov_transform_arr.append(sigma_T)
            cov_eval_arr.append(e_val)
            #self.draw_frame_axis(T_sigma, ax)

            #5.3 attach next anchor
            T = T @ curr_T
            self.draw_frame_axis(T,ax)

        return mean_transform_arr, cov_transform_arr, cov_eval_arr


    def get_mu_x_scale(self, new_pt, i1,i2):
        return np.linalg.norm(new_pt[i1] - new_pt[i2]) / np.linalg.norm(self.anchor_arr[i1] - self.anchor_arr[i2])

    def update_gaussian_transforms(self, waypt_arr):
        #translate to starting point and orientation
        curr = waypt_arr[0]
        frame_arr = [curr]
        mean_arr = []
        cov_arr = []

        T = create_transform_azi(curr, 0.0)

        for i in range(len(self.mu)):
            curr_T, _, _, _, pointing_T = self.transform_from_curr_frame(T, waypt_arr[i], waypt_arr[i+1])

            dis_scale = self.get_mu_x_scale(waypt_arr, i, i+1)

            self.mean_transform_arr[i][0, -1] *= dis_scale
            T_mu = T @ pointing_T @ self.mean_transform_arr[i]
            mean_arr.append(T_mu[:3,-1])

            self.cov_transform_arr[i][0, -1] *= dis_scale
            T_sigma = T @ pointing_T @ self.cov_transform_arr[i]
            new_e_vec = T_sigma[:3,:3]
            new_e_vec *= dis_scale
            cov_arr.append(new_e_vec @ np.diag(self.cov_eval_arr[i]) @ new_e_vec.T)



            #5.3 attach next anchor
            T = T @ curr_T

            frame_arr.append(T[:3, -1])

        return np.array(frame_arr), np.array(mean_arr), np.array(cov_arr)
    

    def plot(self, frame_arr, mean_arr, cov_arr, T_arr, traj_arr=None):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), subplot_kw={'projection': '3d'})
        
        #ax.set_box_aspect(aspect = (1.0,1.0,1.0))

        ax.plot(frame_arr[:,0], frame_arr[:,1], frame_arr[:,2], color='black')
        ax.scatter(mean_arr[:,0], mean_arr[:,1], mean_arr[:,2], color='brown')

        #self.draw_frame_axis(np.eye(4), ax)

        # for i in range(len(T_arr)):
        #     self.draw_frame_axis(T_arr[i], ax)

        # for i in range(len(mean_arr)):
        #     e_val, e_vec = np.linalg.eig(cov_arr[i])
        #     sample_pts = create_ellipsoid(mean_arr[i], np.sqrt(e_val), e_vec)
        #     ax.plot_surface(sample_pts[0], sample_pts[1], sample_pts[2], color='blue', alpha=0.7)

        if traj_arr is not None:
            ax.scatter(traj_arr[:,0], traj_arr[:,1], traj_arr[:,2],color='red', s=2)

        ax.set_xlim([0.0,1.0])
        ax.set_ylim([0.0,1.0])
        ax.set_zlim([0.0,1.0])
        ax.set_xlabel('x') 
        ax.set_ylabel('y')
        ax.set_zlabel('z') 
        plt.show() 
