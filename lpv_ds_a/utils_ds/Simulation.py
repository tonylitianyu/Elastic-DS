import numpy as np

def Simulation(x0, fn_handle, *args):
    options = args[0]

    dim = x0.shape[0]
    xT = np.zeros((dim, 1))

    # setting initial value
    nbSPoint = x0.shape[1]
    x = [x0]
    t = [0]  # Starting time
    xd = []

    for i in np.arange(0, options.i_max):
        # calculate the speed:
        xd.append(fn_handle(x[i]))

        # Integration
        x.append(x[i] + xd[i] * options.dt)
        t.append(t[i] + options.dt)

        if i > 2 and np.all(np.abs(np.concatenate(xd[-3:], axis=1).reshape(-1)) < options.tol):
            print('Final Time is ', t[i])
            print('Final Point is', x[i])
            break

        if i == options.i_max - 1:
            print('reach the maximum and please increase the iteration number')

    return x



