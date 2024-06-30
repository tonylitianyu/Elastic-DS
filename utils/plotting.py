import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
import matplotlib.patches as patches
import matplotlib as mpl
import math
import json
import sys, os
import pickle

sys.path.insert(0, os.getcwd()+'/elastic_gmm')


from elastic_gmm.find_joints import get_product_gaussian, get_joints
from elastic_gmm.gaussian_kinematics import create_transform_azi, get_angles_between_two_2d
from lpv_ds_a.utils_ds.structures import ds_gmms
from lpv_ds_a.math_tool.gaussain.my_gaussPDF import my_gaussPDF
np.set_printoptions(threshold=sys.maxsize)

def load_json_model(task_name, task_idx):
    filename = 'task_dataset/' + task_name + '/model/' + str(task_idx) + '.json'
    f = open(filename)
    data = json.load(f)
    num = data['K']
    dim = data['M']
    A = np.array(data['A']).reshape(num, dim+dim)
    A_ = np.zeros((num, dim, dim))
    for a in range(len(A)):
        A_[a] = A[a].reshape(dim,dim)

    b = np.array(data['B']).reshape(num, dim).T

    ds_gmm_s = ds_gmms()
    ds_gmm_s.Priors = np.array(data['Priors'])
    ds_gmm_s.Mu = np.array(data['Mu']).reshape(num, dim).T
    ds_gmm_s.Sigma = np.array(data['Sigma']).reshape(num, dim, dim)

    way_pt = np.array(data['joint']).reshape(num + 1, dim)

    print('new A', A_)
    print('new_B', b)
    print('new mu', ds_gmm_s.Mu)
    print('new sigma', ds_gmm_s.Sigma)


    return num, dim, A_, b, ds_gmm_s, way_pt

def posterior_probs_gmm(x, gmm, type):
    N = len(x)
    M = len(x[0])

    # Unpack gmm
    Mu = gmm.Mu
    Priors = gmm.Priors
    Sigma = gmm.Sigma
    K = len(Priors)
    # Compute mixing weights for multiple dynamics
    Px_k = np.zeros((K, M))

    # Compute probabilities p(x^i|k)
    for k in np.arange(K):
        Px_k[k, :] = my_gaussPDF(x, Mu[:, k].reshape(N, 1), Sigma[k, :, :])

    # Compute posterior probabilities p(k|x) -- FAST WAY --- %%%
    alpha_Px_k = np.repeat(Priors.reshape(len(Priors),1), M, axis=1) * Px_k

    likelihood = np.copy(Px_k)

    if type == 'norm':
        Pk_x = alpha_Px_k / np.repeat(np.sum(alpha_Px_k, axis=0, keepdims=True), K, axis=0)
    else:
        Pk_x = alpha_Px_k

    return Pk_x, np.sum(likelihood, axis=0)

def plot_nominal_DS(ax, A_, b, ds_gmm_s, way_pt, task_sat, show_traj=True):
    def lpv_ds(x, ds_gmm, A_g, b_g):
        N, M = x.shape  # N for dim and M for data point number
        K = len(ds_gmm.Priors)

        # Posterior Probabilities per local DS
        beta_k_x, likelihood = posterior_probs_gmm(x, ds_gmm, 'norm')  # K*M matrix

        x_dot = np.zeros((N, M))
        for i in np.arange(M):
            f_g = np.zeros((N, K))  # dim * cluster p（k）* （A_k @ x + b_k)
            if b_g.shape[1] > 1:
                for k in np.arange(K):
                    f_g[:, k] = beta_k_x[k][i] * (A_g[k] @ x[:, i] + b_g[:, k])
                f_g = np.sum(f_g, axis=1)

            else:
                # Estimate the Global Dynamics component as Linear DS
                f_g = (A_g @ x[:, i] + b_g.reshape(-1))

            x_dot[:, i] = f_g

        return x_dot, likelihood#np.mean(beta_k_x, axis=0)
    
    
    def plot_rollout(ax, init_x, step):
        curr_x = np.copy(init_x).reshape(-1,1)
        traj = [curr_x.flatten()]
        for t in range(step):
            xdot,_ = lpv_ds(curr_x, ds_gmm_s, A_, b)
            curr_x += 0.01 * xdot
            traj.append(curr_x.flatten())
        traj = np.array(traj)
        ax.plot(traj[:,0], traj[:,1], color='black', zorder=2999, linewidth=3)
        return traj

    res = 200
    lim = 0.2
    x,y = np.meshgrid(np.linspace(0.0,1.0,res),np.linspace(0.0,1.0,res))
    X = np.vstack([x.ravel(), y.ravel()])
    
    #dX = As@X + Bs
    dX, max_K = lpv_ds(X, ds_gmm_s, A_, b)
    
    uu = dX[0,:].reshape((res,res))
    vv = dX[1,:].reshape((res,res))

    ax.streamplot(x,y,uu,vv, density=1.5, color="#10739E", arrowsize=1.3, arrowstyle="->")
    # ax.contourf(x,y, np.array(max_K).reshape((res,res)), levels = 200)

    rollout_traj = None
    if show_traj and way_pt is not None:
        rollout_traj = plot_rollout(ax, way_pt[0], 250)
    return x, y, uu, vv, rollout_traj

def plot_DS_GMM(task_name, task_idx, gate=None, tunnel=None, task_sat=False, saveplot=False, save_idx=0, viapt=False, demo=False, show_traj=True, show_ds=True, show_gmm=True, lim=[0.0, 1.0]):

    num, dim, A_, b, ds_gmm_s, way_pt = load_json_model(task_name, task_idx)

    neighbor_mu = np.stack((way_pt, np.roll(way_pt, -1, axis=0)), axis=1)[:-1][::-1]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5, 10))
    ax.set_aspect(1)

    if show_ds:
        x, y, uu, vv, rollout_traj = plot_nominal_DS(ax, A_, b, ds_gmm_s, way_pt, task_sat, show_traj)

    #ax.scatter(way_pt[:,0], way_pt[:,1], c='#AE4132', zorder=1000)
    #ax.plot(way_pt[:,0], way_pt[:,1], c='black', zorder=1000)

    #ax.plot(way_pt[[-2,-1],0], way_pt[[-2,-1],1], c='red', zorder=1000)
    print('gate', gate)
    if gate is not None:
        for g in gate:
            plot_gate(ax, g[0], g[1], g[2], 0.03, viapt=viapt)

    if tunnel is not None:
        plot_tunnel(ax, tunnel[0], tunnel[1], tunnel[2], 0.03, viapt=viapt)

    if show_gmm:
        plot_gaussian2D(ax, ds_gmm_s.Mu.T, ds_gmm_s.Sigma, show_mu=True, n_std=1.7)

    if demo:
        plot_demonstrations(ax, task_name, "all_2d", show_dir=True)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim(lim)
    ax.set_xlabel(r"$\xi_1$")
    ax.set_ylabel(r"$\xi_2$")
    plt.xticks([]) 
    plt.yticks([]) 

    if saveplot:
        plt.savefig('/home/skyrain/Pictures/DS-Transfer/' + task_name + '/' + str(save_idx) + '_gmm.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()
    #return x, y, uu, vv, rollout_traj
    

def plot_multiple_DS(task_name):
    path = 'task_dataset/' + task_name + '/model/'
    n_ds = len(os.listdir(path))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    ax.set_aspect(1)

    for i in range(n_ds):
        num, dim, A_, b, ds_gmm_s, way_pt = load_json_model(task_name, i)
        #plot_gaussian2D(ax, ds_gmm_s.Mu.T, ds_gmm_s.Sigma)

    plot_demonstrations(ax, task_name, 'all_2d')
    # plot_gate(ax, 0.5, 0.55, 0.0, w=0.02)
    # plot_gate(ax, 0.5, 0.45, np.pi, w=0.02)
    ax.scatter(0.5, 0.5, color='blue', zorder=20000)

    ax.set_xlim([0.0,1.0])
    ax.set_ylim([0.0,1.0])
    ax.set_xlabel(r"$\xi_1$")
    ax.set_ylabel(r"$\xi_2$")
    plt.xticks([]) 
    plt.yticks([]) 
    plt.show()

def draw_frame_axis(ax, T, axis_length=0.04, arrow=False):     
    def T_multi_vec(T, vec):
        vec = vec.flatten()
        return (T @ np.append(vec, 1.0).reshape(-1,1)).flatten()[:2]  
    
    x_axis = T_multi_vec(T, np.array([axis_length,    0]))
    y_axis = T_multi_vec(T, np.array([0,    axis_length]))

    center = T_multi_vec(T, np.array([0.0, 0.0]))
    stack_x = np.vstack((center, x_axis))
    stack_y = np.vstack((center, y_axis))

    if arrow:
        ax.arrow(stack_x[0,0], stack_x[0,1], stack_x[1,0] - stack_x[0,0], stack_x[1,1] - stack_x[0,1], color='#333333', width=0.002, zorder=10000)
        ax.arrow(stack_y[0,0], stack_y[0,1], stack_y[1,0] - stack_y[0,0], stack_y[1,1] - stack_x[0,1], color='#333333', width=0.002, zorder=10000)
    else:
        # ax.plot(stack_x[:,0], stack_x[:,1], color='red', zorder=3000)
        # ax.plot(stack_y[:,0], stack_y[:,1], color='green', zorder=3000)
        ax.arrow(stack_x[0,0], stack_x[0,1], stack_x[1,0] - stack_x[0,0], stack_x[1,1] - stack_x[0,1], color='grey', zorder=10000)
        ax.arrow(stack_y[0,0], stack_y[0,1], stack_y[1,0] - stack_y[0,0], stack_y[1,1] - stack_x[0,1], color='grey', zorder=10000)

def plot_gate(ax, xx,yy, theta, w, viapt=False, alpha=1.0, facecolor="#82B366", edgecolor="#509144", draw_frame=False, pointing_along=True):


    x = xx - 3*w
    y = yy + 1.5*w
    pt = np.array([[x, y],[x, y - 3*w],[x + 6*w, y - 3*w], [x + 6*w, y], [x + 6*w - w, y], [x + 6*w - w, y - 2*w], [x + 6*w - 5*w, y - 2*w], [x + 6*w - 5*w, y], [x, y]])

    r2 = patches.Polygon(pt, closed=True, facecolor=facecolor, edgecolor=edgecolor,lw=3.0, alpha=alpha, zorder=9999)
    if pointing_along:
        theta += -np.pi/2
    else:
        theta += np.pi/2
    t2 = mpl.transforms.Affine2D().rotate_around(xx,yy,theta) + ax.transData
    r2.set_transform(t2)

    if draw_frame:
        draw_frame_axis(ax, create_transform_azi(np.array([xx,yy]), theta), arrow=False)
    if not viapt:
        ax.add_patch(r2)

def plot_tunnel(ax, xx,yy, theta, w, viapt=False, alpha=1.0, facecolor="#82B366", edgecolor="#509144", draw_frame=False):
    x = xx - 3*w
    y = yy + 2*w
    pt1 = np.array([[x,y], [x, y-4*w], [x+1.0*w, y-4*w], [x+1.0*w, y], [x, y]])
    r1 = patches.Polygon(pt1, closed=True, facecolor=facecolor, edgecolor=edgecolor,lw=3.0, alpha=alpha, zorder=9999)
    t1 = mpl.transforms.Affine2D().rotate_around(xx,yy,theta - np.pi/2) + ax.transData
    r1.set_transform(t1)

    x = xx + 3*w
    y = yy + 2*w
    pt2 = np.array([[x,y], [x, y-4*w], [x-1.0*w, y-4*w], [x-1.0*w, y], [x, y]])
    r2 = patches.Polygon(pt2, closed=True, facecolor=facecolor, edgecolor=edgecolor,lw=3.0, alpha=alpha, zorder=9999)
    t2 = mpl.transforms.Affine2D().rotate_around(xx,yy,theta - np.pi/2) + ax.transData
    r2.set_transform(t2)

    ax.add_patch(r1)
    ax.add_patch(r2)


def plot_gaussian2D(ax, mus, covs, edge_color="#D79B00", face_color="#FFE6CC", show_mu=True, n_std=1.7, face_alpha=0.6):
    for i in range(len(mus)):
        e_val, e_vec = np.linalg.eigh(covs[i])
        order = e_val.argsort()[::-1]
        e_val, e_vec = e_val[order], e_vec[:, order]
        x, y = e_vec[:, 0]
        theta = np.degrees(np.arctan2(y, x))

        n_std = n_std
        width, height = 2 * n_std * np.sqrt(e_val)

        #sample_pts = create_ellipsoid2d(mus[i], np.sqrt(e_val), e_vec)
        ellip = patches.Ellipse(xy=mus[i], width=width, height=height, angle=theta, \
                                edgecolor=edge_color, facecolor=face_color, lw=2.0, zorder=100, alpha=face_alpha)
        if show_mu:
            ax.scatter(mus[i][0], mus[i][1], c=edge_color, s=8, zorder=101)

        ax.add_patch(ellip)


def plot_gaussian_joint(task_name, task_idx):
    filename = 'task_dataset/' + task_name + '/model/' + str(task_idx) + '.json'
    f = open(filename)
    data = json.load(f)
    num = data['K'] #3 
    dim = data['M'] #2

    ds_gmm_s = ds_gmms()
    ds_gmm_s.Priors = np.array(data['Priors'])
    ds_gmm_s.Mu = np.array(data['Mu']).reshape(num, dim)
    ds_gmm_s.Sigma = np.array(data['Sigma']).reshape(num, dim, dim)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    ax.set_aspect(1)

    mu1 = ds_gmm_s.Mu[2]
    mu2 = ds_gmm_s.Mu[3]
    sigma1 = ds_gmm_s.Sigma[2]
    sigma2 = ds_gmm_s.Sigma[3]
    mu, sigma = get_product_gaussian(mu1, mu2, sigma1, sigma2)

    mu_all = np.array([mu1,mu2])
    sigma_all = np.array([sigma1,sigma2])

    plot_gaussian2D(ax, mu_all, sigma_all, show_mu=False)
    plot_gaussian2D(ax, [mu], [sigma], edge_color="#B85450", face_color="#F89F9C", show_mu=True)


    ax.text(mu1[0], mu1[1], "2", zorder=2000, color="#D79B00", fontsize="large", fontweight="semibold", ha="center", va="center")
    ax.plot([mu1[0]-0.02, 0.140], [mu1[1]+0.01, 0.78], color="#555555", zorder=2001)
    ax.text(0.13, 0.80, r"$\{\mu_2, \Sigma_2\}$", zorder=2000, color="#000000", fontsize="x-large", fontweight="semibold", ha="center", va="center", parse_math=True)


    ax.text(mu2[0], mu2[1], "1", zorder=2000, color="#D79B00", fontsize="large", fontweight="semibold", ha="center", va="center")
    ax.plot([mu2[0] + 0.01, 0.27], [mu2[1] - 0.01, 0.45], color="#555555", zorder=2001)
    ax.text(0.29, 0.43, r"$\{\mu_1, \Sigma_1\}$", zorder=2000, color="#000000", fontsize="x-large", fontweight="semibold", ha="center", va="center", parse_math=True)


    ax.plot([mu[0]+0.005, 0.362], [mu[1]-0.005, 0.562], color="#555555", zorder=2001)
    ax.text(0.34, 0.562-0.04, r"$\{\beta_{12}, (\Sigma_1^{-1} + \Sigma_2^{-1})^{-1}\}$", zorder=2000, color="#000000", fontsize="x-large", fontweight="semibold", ha="left", va="center", parse_math=True)

    ax.set_xlim([-0.1,0.7])
    ax.set_ylim([0.35,0.85])
    plt.xticks([]) 
    plt.yticks([]) 
    plt.show()


def plot_demonstrations(ax, data_drawn, show_dir=False):

    for ii in range(len(data_drawn)):
        ax.plot(data_drawn[ii][0][0,:], data_drawn[ii][0][1,:], color="red", linewidth=5)
        
        xx = data_drawn[ii][0][0,:][::5]
        yy = data_drawn[ii][0][1,:][::5]
        if show_dir:
            dx = [xx[j+1] - xx[j] for j in range(0,len(xx)-1)]
            dy = [yy[j+1] - yy[j] for j in range(0,len(yy)-1)]
            ax.quiver(xx[:-1], yy[:-1], dx, dy, angles='xy', scale_units='xy', scale=0.8, zorder=100000)


def plot_mu_frame(task_name, task_idx):
    filename = 'task_dataset/' + task_name + '/model/' + str(task_idx) + '.json'
    f = open(filename)
    data = json.load(f)
    num = data['K'] #3 
    dim = data['M'] #2

    ds_gmm_s = ds_gmms()
    ds_gmm_s.Priors = np.array(data['Priors'])
    ds_gmm_s.Mu = np.array(data['Mu']).reshape(num, dim)
    ds_gmm_s.Sigma = np.array(data['Sigma']).reshape(num, dim, dim)

    way_pt = np.array(data['joint']).reshape(num + 1, dim)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    ax.set_aspect(1)

    # mu1 = ds_gmm_s.Mu[1]
    # mu2 = ds_gmm_s.Mu[2]
    # sigma1 = ds_gmm_s.Sigma[1]
    # sigma2 = ds_gmm_s.Sigma[2]
    

    plot_gaussian2D(ax, ds_gmm_s.Mu, ds_gmm_s.Sigma, show_mu=True)
    ax.scatter(way_pt[:,0], way_pt[:,1], c='#AE4132', zorder=1000)

    

    # ax.text(mu1[0], mu1[1], "2", zorder=2000, color="#D79B00", fontsize="large", fontweight="semibold", ha="center", va="center")
    ax.plot([way_pt[1,0], ds_gmm_s.Mu[2,0]], [way_pt[1,1], ds_gmm_s.Mu[2,1]], color="#555555", linestyle="--", zorder=2001)
    ax.plot([way_pt[1,0], way_pt[2,0]], [way_pt[1,1], way_pt[2,1]], color="#555555", linestyle="--", zorder=2002)


    angle_to_next_joint = get_angles_between_two_2d(np.array([1.0,0.0]), way_pt[2,:] - way_pt[1,:])

    draw_frame_axis(ax, create_transform_azi(way_pt[1,:], angle_to_next_joint), 0.045, arrow=True)


    e_val, e_vec = np.linalg.eigh(ds_gmm_s.Sigma[2])
    e_vec = np.vstack((np.hstack((e_vec, np.zeros((2,1)))), np.zeros((1,3))))
    e_vec[-1,-1] = 1

    sigma_trans = create_transform_azi(ds_gmm_s.Mu[2,:], 0.0) @ e_vec @ create_transform_azi(np.zeros(2), np.pi)
    draw_frame_axis(ax, sigma_trans, 0.03, arrow=False)

    ax.text(0.224, 0.628, r"$\beta_{12}$", zorder=2000, color="#000000", fontsize="xx-large", fontweight="semibold", ha="center", va="center", parse_math=True)
    ax.text(0.470, 0.758, r"$\beta_{23}$", zorder=2000, color="#000000", fontsize="xx-large", fontweight="semibold", ha="center", va="center", parse_math=True)


    ax.text(0.349, 0.755, r"$\hat{e}_{22}$", zorder=2000, color="#D79B00", fontsize="large", fontweight="semibold", ha="center", va="center")
    ax.text(0.398, 0.736, r"$\hat{e}_{21}$", zorder=2000, color="#D79B00", fontsize="large", fontweight="semibold", ha="center", va="center")
    ax.text(0.339, 0.718, r"$\mu_2$", zorder=2000, color="#D79B00", fontsize="large", fontweight="semibold", ha="center", va="center")
    # ax.plot([mu2[0] + 0.01, 0.27], [mu2[1] - 0.01, 0.45], color="#555555", zorder=2001)
    # ax.text(0.29, 0.43, r"$\{\mu_1, \Sigma_1\}$", zorder=2000, color="#333333", fontsize="large", fontweight="semibold", ha="center", va="center", parse_math=True)


    # ax.plot([mu[0] - 0.01, 0.17], [mu[1] + 0.005, 0.66], color="#555555", zorder=2001)
    # ax.text(0.15, 0.68, r"$\{\beta_{12}, (\Sigma_1^{-1} + \Sigma_2^{-1})^{-1}\}$", zorder=2000, color="#333333", fontsize="large", fontweight="semibold", ha="center", va="center", parse_math=True)

    ax.set_xlim([0.15,0.6])
    ax.set_ylim([0.55,0.85])
    plt.xticks([]) 
    plt.yticks([]) 
    plt.show()

def create_R(jnt_rot):
    return np.array([[np.cos(jnt_rot) , -np.sin(jnt_rot)],[np.sin(jnt_rot) , np.cos(jnt_rot)]]) 

def get_angle_from_rotation_matrix(matrix):
    cos_theta = matrix[0, 0]
    sin_theta = matrix[1, 0]

    theta_rad = np.arccos(cos_theta)  # angle in radians

    # Ensure the correct quadrant for the angle
    if sin_theta < 0:
        theta_rad = -theta_rad

    return theta_rad

class plot_struct:
    show_ds = None
    show_gmm = None
    geo = None
    saveplot = None
    demo = None
    show_traj = None
    tunnel = False


def plot_single(ax, ds_struct, vis_struct):
    gmm = ds_struct.gmm_struct
    if vis_struct.show_ds:
        x, y, uu, vv, rollout_traj = plot_nominal_DS(ax, ds_struct.A_k, ds_struct.b_k, gmm\
                                                     , ds_struct.joints, vis_struct.show_traj)

    if vis_struct.geo is not None:
        if vis_struct.tunnel:
            g = vis_struct.geo
            plot_tunnel(ax, g[0], g[1], g[2], 0.03)
        else:
            for i, g in enumerate(vis_struct.geo):
                if g is None:
                    continue
                p = False
                if i % 2 == 0:
                    p = True
                plot_gate(ax, g[0], g[1], g[2], 0.03, pointing_along=p)

    if vis_struct.show_gmm:
        plot_gaussian2D(ax, gmm.Mu.T, gmm.Sigma, show_mu=True, n_std=1.7)

    if vis_struct.demo:
        plot_demonstrations(ax, ds_struct.demo_traj, show_dir=True)




def plot_single_func(ds_struct):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    vs = plot_struct()
    vs.show_ds = True
    vs.show_gmm = True
    vs.geo = None
    vs.saveplot = False
    vs.demo = True
    vs.show_traj = False

    plot_single(ax, ds_struct, vs)
    plt.show()


def plot_full_func(ds_struct, original_ds_struct, tunnel=False):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5, 10))

    for i in range(2):
        ax[i].set_aspect(1)
        ax[i].set_xlim([0.0, 1.0])
        ax[i].set_ylim([0.0, 1.0])
        ax[i].set_xlabel(r"$\xi_1$", fontsize=10)
        ax[i].set_ylabel(r"$\xi_2$", fontsize=10)
        ax[i].set_xticks([]) 
        ax[i].set_yticks([])

    vs0 = plot_struct()
    vs0.show_ds = True
    vs0.show_gmm = True
    vs0.geo = original_ds_struct.geo
    vs0.saveplot = False
    vs0.demo = True
    vs0.show_traj = False
    vs0.tunnel = tunnel
    plot_single(ax[0], original_ds_struct, vs0)
    ax[0].set_title('The Original Policy', fontsize=15)


    vs1 = plot_struct()
    vs1.show_ds = True
    vs1.show_gmm = True
    vs1.geo = ds_struct.geo
    vs1.saveplot = False
    vs1.demo = False
    vs1.show_traj = True
    vs1.tunnel = tunnel
    plot_single(ax[1], ds_struct, vs1)
    ax[1].set_title('The New Policy', fontsize=15)

    plt.show()
    #plt.savefig('imgs/via.png')


    





if __name__ == "__main__":
    plot_DS_GMM('Lshape', 0, way_pt=[], gate=[], saveplot=False, demo=False, show_ds=True, show_gmm=True, show_traj=False)
