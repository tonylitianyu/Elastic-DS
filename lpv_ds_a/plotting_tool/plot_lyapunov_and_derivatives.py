from lpv_ds_a.math_tool.lyapunov.compute_lyapunov_function import lyapunov_function_PQLF
from lpv_ds_a.math_tool.lyapunov.compute_lyapunov_derivative import lyapunov_function_deri_PQLF

from lpv_ds_a.plotting_tool.plot_lyap_fct import plot_lyap_fct


def plot_lyapunov_and_derivatives(Data, ds_handle, att, P_opt):
    Data_dim = int(len(Data) / 2)
    print('Doing visualization for 2D dataset')
    title_1 = 'Lyapunov derivative plot'
    title_2 = 'Lyapunov function value plot'
    # lyap_handle = lambda x : lyapunov_function_PQLF(x, att, P_opt)
    # lyap_derivative_handle = lambda x : lyapunov_function_deri_PQLF(x, att, P_opt, ds_handle)

    lyap_handle = lambda x: lyapunov_function_PQLF(x, att, P_opt)
    lyap_derivative_handle = lambda x: lyapunov_function_deri_PQLF(x, att, P_opt, ds_handle)
    plot_lyap_fct(Data[:Data_dim], att, lyap_derivative_handle, title_1)
    plot_lyap_fct(Data[:Data_dim], att, lyap_handle, title_2)