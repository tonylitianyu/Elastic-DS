import numpy as np
from lpv_ds_a.utils_ds.structures import Opt_Sim
from lpv_ds_a.utils_ds.Simulation import Simulation
import matplotlib.pyplot as plt
from lpv_ds_a.plotting_tool.sample_initial_points import sample_initial_points


def VisualizeEstimatedDS(Xi_ref, ds_lpv, ds_plot_options):
    dim = Xi_ref.shape[0]

    # Parse Options
    plot_repr = ds_plot_options.sim_traj
    x0_all = ds_plot_options.x0_all

    if dim == 3:
        plot_2D_only = 0

        init_type = ds_plot_options.init_type
        nb_pnts = ds_plot_options.nb_points
        plot_volumn = ds_plot_options.plot_vol

    if plot_repr:
        opt_sim = Opt_Sim()
        opt_sim.dt = 0.005
        opt_sim.i_max = 10000
        opt_sim.tol = 0.001
        opt_sim.plot = 0
        x_sim = Simulation(x0_all, ds_lpv, opt_sim)

    if dim == 3:
        num_of_traj = x0_all.shape[1]
        trajs = np.array(x_sim)
        fig = plt.figure()
        ax1 = plt.axes(projection='3d')
        ax1.scatter(Xi_ref[0], Xi_ref[1], Xi_ref[2], c='r', label='original demonstration', s=5)
        for i in np.arange(num_of_traj):
            cur_traj = trajs[:, :, i].T
            if i != num_of_traj - 1:
                ax1.plot3D(cur_traj[0], cur_traj[1], cur_traj[2], 'black')
            else:
                ax1.plot3D(cur_traj[0], cur_traj[1], cur_traj[2], 'black', label='reproduced trajectories')
        # random_initial_points = sample_initial_points(x0_all, nb_pnts, init_type, [])
        # ax1.scatter(random_initial_points[0], random_initial_points[1], random_initial_points[2], c='b', s=5)
        # trajs_rand = np.array(Simulation(random_initial_points, ds_lpv, opt_sim))
        # for i in np.arange(nb_pnts):
        #     cur_traj = trajs_rand[:, :, i].T
        #     if i == nb_pnts - 1:
        #         ax1.plot3D(cur_traj[0], cur_traj[1], cur_traj[2], 'blue', label='random trajectories')
        #     else:
        #         ax1.plot3D(cur_traj[0], cur_traj[1], cur_traj[2], 'blue')
        ax1.legend(loc="best")
        plt.show()
    elif dim == 2:
        num_of_traj = x0_all.shape[1]
        trajs = np.array(x_sim)
        fig, ax1 = plt.subplots()
        ax1.scatter(Xi_ref[0], Xi_ref[1], c='r', label='original demonstration', s=3)
        for i in np.arange(num_of_traj):
            cur_traj = trajs[:, :, i].T
            if i != num_of_traj - 1:
                ax1.plot(cur_traj[0], cur_traj[1], 'blue')
            else:
                ax1.plot(cur_traj[0], cur_traj[1], 'blue', label='reproduced trajectories')

        axis_limits = ax1.viewLim
        x0 = axis_limits.x0
        y0 = axis_limits.y0
        x1 = axis_limits.x1
        y1 = axis_limits.y1
        resolution = 60
        x_range = np.arange(x0, x1, (x1 - x0) / resolution)
        y_range = np.arange(y0, y1, (y1 - y0) / resolution)
        xx, yy = np.meshgrid(x_range, y_range)
        field_data = np.vstack((xx.flatten(), yy.flatten()))
        field_velo = ds_lpv(field_data)
        # field_velo[0] /= np.sqrt(field_velo[0] ** 2 + field_velo[1] ** 2)
        # field_velo[1] /= np.sqrt(field_velo[0] ** 2 + field_velo[1] ** 2)
        ax1.streamplot(xx, yy, field_velo[0].reshape(xx.shape), field_velo[1].reshape(yy.shape), density=[3.5, 3.5])

        # random_initial_points = sample_initial_points(x0_all, nb_pnts, init_type, [])
        # ax1.scatter(random_initial_points[0], random_initial_points[1], random_initial_points[2], c='b', s=5)
        # trajs_rand = np.array(Simulation(random_initial_points, ds_lpv, opt_sim))
        # for i in np.arange(nb_pnts):
        #     cur_traj = trajs_rand[:, :, i].T
        #     if i == nb_pnts - 1:
        #         ax1.plot3D(cur_traj[0], cur_traj[1], cur_traj[2], 'blue', label='random trajectories')
        #     else:
        #         ax1.plot3D(cur_traj[0], cur_traj[1], cur_traj[2], 'blue')
        ax1.legend(loc="best")
        plt.show()
