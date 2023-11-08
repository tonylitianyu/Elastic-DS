from ctypes import sizeof
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from ..math_tools import lyapunov_tools, pca_tools
from ..data_tools import structures, simulation
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter
import matplotlib.lines as mlines


font = {'family' : 'Times New Roman',
         'size'   : 20
        #  'serif':  'cmr10'
         }
mpl.rc('font', **font)
mpl.rc('text', usetex = True)



def plot_lyap_fct(Data, att, lyap_fun, title):
    # resolution
    nx = 200
    ny = 200
    # plot the data to get the auto limitation of matplotlib
    fig, ax = plt.subplots()
    plt.plot(Data[0], Data[1], 'ro', markersize=1)
    # plot attractor
    plt.scatter(att[0], att[1], s=100, c='blue', alpha=0.5)
    axis_limits = ax.viewLim
    x0 = axis_limits.x0
    y0 = axis_limits.y0
    x1 = axis_limits.x1
    y1 = axis_limits.y1
    ax_x = np.linspace(x0, x1, num = nx)
    ax_y = np.linspace(y0, y1, num = ny)
    x_0, x_1 = np.meshgrid(ax_x, ax_y)
    x_0_f = x_0.flatten()
    x_1_f = x_1.flatten()
    x_total = np.vstack((x_0_f, x_1_f))
    z_value = lyap_fun(x_total).reshape(nx, ny)
    cp = plt.contourf(x_0, x_1, z_value)
    plt.colorbar(cp)

    ax.set_title(title)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$y_2$')
    plt.show()


def plot_lyapunov_and_derivatives(Data, ds_handle, att, P_opt):
    Data_dim = int(len(Data) / 2)
    print('Doing visualization for 2D dataset')
    title_1 = 'Lyapunov derivative plot'
    title_2 = 'Lyapunov function value plot'
    # lyap_handle = lambda x : lyapunov_function_PQLF(x, att, P_opt)
    # lyap_derivative_handle = lambda x : lyapunov_function_deri_PQLF(x, att, P_opt, ds_handle)

    lyap_handle = lambda x: lyapunov_tools.lyapunov_function_PQLF(x, att, P_opt)
    lyap_derivative_handle = lambda x: lyapunov_tools.lyapunov_function_deri_PQLF(x, att, P_opt, ds_handle)
    plot_lyap_fct(Data[:Data_dim], att, lyap_derivative_handle, title_1)
    plot_lyap_fct(Data[:Data_dim], att, lyap_handle, title_2)


def plot_reference_trajectories_DS(Data, att, vel_sample, vel_size):
    fig = plt.figure(figsize=(8, 6))
    M = len(Data) / 2  # store 1 Dim of Data
    if M == 2:
        ax = fig.add_subplot(111)
        ax.set_xlabel(r'$\xi_1$')
        ax.set_ylabel(r'$\xi_2$')
        ax.set_title('Reference Trajectory')

        # Plot the position trajectories
        plt.plot(Data[0], Data[1], 'ro', markersize=1)
        # plot attractor
        plt.scatter(att[0], att[1], s=100, c='blue', alpha=0.5)
        # Plot Velocities of Reference Trajectories
        vel_points = Data[:, ::vel_sample]
        U = np.zeros(len(vel_points[0]))
        V = np.zeros(len(vel_points[0]))  # （385,)
        for i in np.arange(0, len(vel_points[0])):
            dir_ = vel_points[2:, i] / np.linalg.norm(vel_points[2:, i])
            U[i] = dir_[0]
            V[i] = dir_[1]
        q = ax.quiver(vel_points[0], vel_points[1], U, V, width=0.005, scale=vel_size)
    else:
        ax = fig.add_subplot(projection='3d')
        ax.plot(Data[0], Data[1], Data[2], 'ro', markersize=1.5)
        ax.scatter(att[0], att[1], att[2], s=200, c='blue', alpha=0.5)
        ax.axis('auto')
        ax.set_title('Reference Trajectory')
        ax.set_xlabel(r'$\xi_1(m)$')
        ax.set_ylabel(r'$\xi_2(m)$')
        ax.set_zlabel(r'$\xi_3(m)$')
        vel_points = Data[:, ::vel_sample]
        U = np.zeros(len(vel_points[0]))
        V = np.zeros(len(vel_points[0]))
        W = np.zeros(len(vel_points[0]))
        for i in np.arange(0, len(vel_points[0])):
            dir_ = vel_points[3:, i] / np.linalg.norm(vel_points[3:, i])
            U[i] = dir_[0]
            V[i] = dir_[1]
            W[i] = dir_[2]
        q = ax.quiver(vel_points[0], vel_points[1], vel_points[2], U, V, W, length=0.04, normalize=True,colors='k')

    plt.show()




def sample_initial_points(x0_all, nb_points, type, plot_volumn):
    # Auxiliary Variable
    dim = x0_all.shape[0]

    # Output Variable
    init_points = np.zeros((dim, nb_points))

    # Estimate point distribution
    V, D, init_mu = pca_tools.my_pca(x0_all)
    D[0][0] = 1.5 * D[0][0] # extend the -smallest- dimension
    D[2][2] = 5 * D[2][2] # extend the -smallest- dim # WHAT?


    A_y, y0_all = pca_tools.project_pca(x0_all, init_mu, V, dim)
    Ymin_values = np.min(y0_all, axis=1)
    Ymin_values = Ymin_values + np.array([0, 0.5*Ymin_values[1], 0.5*Ymin_values[2]])
    Yrange_values = np.max(y0_all, axis=1, keepdims=True) - np.min(y0_all, axis=1, keepdims=True)
    Yrange_values[1] += 0.5 * Yrange_values[1]
    Yrange_values[2] += 0.5 * Yrange_values[2]
    init_points_y = np.tile(Ymin_values.reshape(dim, 1), nb_points) + np.random.rand(dim, nb_points) * np.tile(Yrange_values,nb_points)

    init_points = pca_tools.reconstruct_pca(init_points_y, A_y, init_mu)

    return init_points



def VisualizeEstimatedDS(Xi_ref, ds_lpv, ds_plot_options, *args_):
    print(len(args_))
    dim = Xi_ref.shape[0]

    # Parse Options
    plot_repr = ds_plot_options.sim_traj  
    x0_all = ds_plot_options.x0_all
    att = ds_plot_options.attractor

    if dim == 3:
        plot_2D_only = 0

        init_type = ds_plot_options.init_type
        nb_pnts = ds_plot_options.nb_points
        plot_volumn = ds_plot_options.plot_vol

    if plot_repr:
        opt_sim = structures.Opt_Sim()
        opt_sim.dt = 0.005
        opt_sim.i_max = 10000
        opt_sim.tol = 0.001
        opt_sim.plot = 0
        x_sim = simulation.simulation(x0_all, ds_lpv, opt_sim)

    if dim == 3:
        if len(args_)==1:  
            prev_data = args_[0]
            new_data = Xi_ref
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(projection='3d')
            ax.plot(prev_data[0], prev_data[1], prev_data[2], 'o', color='r', markersize=1.5,  label='original data')
            # ax.scatter(att[0], att[1], att[2], s=200, c='blue', alpha=0.5)
            ax.plot(new_data[0], new_data[1], new_data[2], 'o', color = 'magenta', markersize=1.5,  label='new data')
        
            new_label = mlines.Line2D([], [], color='red',
                                linewidth=3, label='Old Demo')
            old_label = mlines.Line2D([], [], color='magenta',
                                linewidth=3, label='New Demo')
            ax.legend(handles=[new_label, old_label])
        else:
            fig = plt.figure(figsize=(10, 8))
            ax = plt.axes(projection='3d')
            ax.scatter(Xi_ref[0,::4], Xi_ref[1,::4], Xi_ref[2,::4], c='r', label='Demonstration', s=10)

    
        num_of_traj = x0_all.shape[1]
        print('Number of trajectory: ', num_of_traj)
        trajs = np.array(x_sim)
        for i in np.arange(num_of_traj):
            cur_traj = trajs[:, :, i].T
            if i != num_of_traj - 1:
                ax.plot3D(cur_traj[0], cur_traj[1], cur_traj[2], 'k', linewidth=6)
            else:
                ax.plot3D(cur_traj[0], cur_traj[1], cur_traj[2], 'k', linewidth=6, label='Reproduction')
        
        ax.scatter(att[0], att[1], att[2], marker=(8, 2, 0), s=150, c='k', label='Target')
        ax.axis('auto')
        ax.set_title('DAMM LPV-DS', fontsize=24)
        ax.set_xlabel(r'$\xi_1(m)$')
        ax.set_ylabel(r'$\xi_2(m)$')
        ax.set_zlabel(r'$\xi_3(m)$')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        plt.show()

        # random_initial_points = sample_initial_points(x0_all, nb_pnts, init_type, [])
        # ax1.scatter(random_initial_points[0], random_initial_points[1], random_initial_points[2], c='b', s=5)
        # trajs_rand = np.array(simulation.simulation(random_initial_points, ds_lpv, opt_sim))
        # for i in np.arange(nb_pnts):
        #     cur_traj = trajs_rand[:, :, i].T
        #     if i == nb_pnts - 1:
        #         ax1.plot3D(cur_traj[0], cur_traj[1], cur_traj[2], 'blue', label='random trajectories')
        #     else:
        #         ax1.plot3D(cur_traj[0], cur_traj[1], cur_traj[2], 'blue')




    elif dim == 2:
        num_of_traj = x0_all.shape[1]
        trajs = np.array(x_sim)
        fig, ax1 = plt.subplots(figsize=(10, 8))
        line1 = ax1.plot(Xi_ref[0], Xi_ref[1], marker='o', c='r', markersize=3, linestyle='None', label='Demonstration')
        for i in np.arange(num_of_traj):
            cur_traj = trajs[:, :, i].T
            if i != num_of_traj - 1:
                ax1.plot(cur_traj[0], cur_traj[1], 'k', linewidth=4)
            else:
                ax1.plot(cur_traj[0], cur_traj[1], 'k', linewidth=4, label='Reproduction')
        ax1.set_xlabel(r'$\xi_1$')
        ax1.set_ylabel(r'$\xi_2$')

        axis_limits = ax1.viewLim
        x0 = axis_limits.x0
        y0 = axis_limits.y0
        x1 = axis_limits.x1
        y1 = axis_limits.y1
        resolution = 15
        x_range = np.arange(x0, x1, (x1 - x0) / resolution)
        y_range = np.arange(y0, y1, (y1 - y0) / resolution)
        xx, yy = np.meshgrid(x_range, y_range)
        field_data = np.vstack((xx.flatten(), yy.flatten()))
        field_velo = ds_lpv(field_data)
        
        # Unsure about the normalization
        # field_velo[0] /= np.sqrt(field_velo[0] ** 2 + field_velo[1] ** 2)
        # field_velo[1] /= np.sqrt(field_velo[0] ** 2 + field_velo[1] ** 2)

        ax1.streamplot(xx, yy, field_velo[0].reshape(xx.shape), field_velo[1].reshape(yy.shape), density=[1.5, 1.5])
        
        ax1.scatter(att[0], att[1], marker=(8, 2, 0), s=150, c='k', label='Target')

        # 
        # random_initial_points = sample_initial_points(x0_all, nb_pnts, init_type, [])
        # ax1.scatter(random_initial_points[0], random_initial_points[1], random_initial_points[2], c='b', s=5)
        # trajs_rand = np.array(simulation(random_initial_points, ds_lpv, opt_sim))
        # for i in np.arange(nb_pnts):
        #     cur_traj = trajs_rand[:, :, i].T
        #     if i == nb_pnts - 1:
        #         ax1.plot3D(cur_traj[0], cur_traj[1], cur_traj[2], 'blue', label='random trajectories')
        #     else:
        #         ax1.plot3D(cur_traj[0], cur_traj[1], cur_traj[2], 'blue')
        
        from matplotlib.legend_handler import HandlerLine2D
        ax1.legend(handler_map={line1[0]: HandlerLine2D(numpoints=5)}, bbox_to_anchor=(1, 1.15), ncol=4, fancybox=True, fontsize=14)

        ax1.set_title("DAMM LPV-DS", fontsize=30)

        plt.show()



def plot_incremental_ds(Xi_ref, ds_lpv, ds_plot_options, prev_data):
    dim = Xi_ref.shape[0]

    # Parse Options
    plot_repr = ds_plot_options.sim_traj  # 是否画reproduction
    x0_all = ds_plot_options.x0_all
    att = ds_plot_options.attractor

    plot_2D_only = 0

    init_type = ds_plot_options.init_type
    nb_pnts = ds_plot_options.nb_points
    plot_volumn = ds_plot_options.plot_vol

    opt_sim = structures.Opt_Sim()
    opt_sim.dt = 0.005
    opt_sim.i_max = 10000
    opt_sim.tol = 0.001
    opt_sim.plot = 0
    x_sim = simulation.simulation(x0_all, ds_lpv, opt_sim)

    new_data = Xi_ref
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(prev_data[0,::4], prev_data[1, ::4], prev_data[2,::4], color='r', s=10,  label='original data')
    # ax.scatter(att[0], att[1], att[2], s=200, c='blue', alpha=0.5)
    ax.scatter(new_data[0,::4], new_data[1,::4], new_data[2,::4], color = 'magenta', s=10,  label='new data')


    new_label = mlines.Line2D([], [], color='red',
                        linewidth=3, label='Old Demo')
    old_label = mlines.Line2D([], [], color='magenta',
                        linewidth=3, label='New Demo')
    ax.legend(handles=[new_label, old_label])


    num_of_traj = x0_all.shape[1]
    print('Number of trajectory: ', num_of_traj)
    trajs = np.array(x_sim)
    for i in np.arange(num_of_traj):
        cur_traj = trajs[:, :, i].T
        if i != num_of_traj - 1:
            ax.plot3D(cur_traj[0], cur_traj[1], cur_traj[2], 'k', linewidth=3.5)
        else:
            ax.plot3D(cur_traj[0], cur_traj[1], cur_traj[2], 'k', linewidth=3.5, label='Reproduction')
    
    ax.scatter(att[0], att[1], att[2], marker=(8, 2, 0), s=150, c='k', label='Target')
    ax.axis('auto')
    ax.set_title('DAMM LPV-DS', fontsize=24)
    ax.set_xlabel(r'$\xi_1(m)$')
    ax.set_ylabel(r'$\xi_2(m)$')
    ax.set_zlabel(r'$\xi_3(m)$')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    plt.show()

    # random_initial_points = sample_initial_points(x0_all, nb_pnts, init_type, [])
    # ax1.scatter(random_initial_points[0], random_initial_points[1], random_initial_points[2], c='b', s=5)
    # trajs_rand = np.array(simulation.simulation(random_initial_points, ds_lpv, opt_sim))
    # for i in np.arange(nb_pnts):
    #     cur_traj = trajs_rand[:, :, i].T
    #     if i == nb_pnts - 1:
    #         ax1.plot3D(cur_traj[0], cur_traj[1], cur_traj[2], 'blue', label='random trajectories')
    #     else:
    #         ax1.plot3D(cur_traj[0], cur_traj[1], cur_traj[2], 'blue')

    # legend = ax1.legend(loc="best")
    # legend.set_draggable(True)
    # ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax1.set_xlabel(r'$\xi_1(m)$')
    # ax1.set_ylabel(r'$\xi_2(m)$')
    # ax1.set_zlabel(r'$\xi_3(m)$')
    # plt.show()

