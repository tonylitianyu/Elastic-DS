import numpy as np
import matplotlib.pyplot as plt


# Plot the Lyapunov Function Contours
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


