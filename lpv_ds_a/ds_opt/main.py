import numpy as np
import json

from lpv_ds_a.ds_opt.util.math_tools import ds_tools, optimization_tools
from lpv_ds_a.ds_opt.util.data_tools import plot_tools, structures, rearrange_clusters


def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def write_json(data, path):
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def read_param(data):
    K = data['K']
    M = data['M']
    Priors = np.array(data['Priors'])
    Mu = np.array(data['Mu']).reshape(K, -1)
    Sigma = np.array(data['Sigma']).reshape(K, M, M)
    
    return K, M, Priors, Mu, Sigma


def read_data(data):
    return data["Data"], data["Data_sh"], data["att"], data["x0_all"], data["dt"], data["traj_length"]


class ds_opt:
    def __init__(self, Data, Data_sh, att, x0_all, dt, traj_length, K, M, Priors, Mu, Sigma):

        # data and path
        self.Data = Data
        self.Data_sh = Data_sh
        self.att = att
        self.x0_all = x0_all
        self.dt = dt
        self.traj_length = traj_length


        # gmm parameters
        self.K, self.M, self.Priors, self.Mu, self.Sigma = K, M, Priors, Mu, Sigma
        self.ds_struct = rearrange_clusters.rearrange_clusters(self.Priors, self.Mu, self.Sigma, self.att)

        # ds parameters
        self.A_k = np.zeros((self.K, self.M, self.M))
        self.b_k = np.zeros((self.M, self.K))
        self.P_opt = np.zeros((self.M, self.M))


    def begin(self):
        
        # run ds-opt
        self.P_opt = optimization_tools.optimize_P(self.Data_sh)
        self.A_k, self.b_k = optimization_tools.optimize_lpv_ds_from_data(self.Data, self.att, 2, self.ds_struct, self.P_opt, 0)
        
        # process ds parameters for json output
        new_A_k = np.copy(self.A_k)
        new_b_k = np.copy(self.b_k)
        new_Sig = np.copy(self.Sigma)

        return self.A_k, self.b_k

        print('done')

        # for k in range(self.K):
        #     new_A_k[k] = new_A_k[k].T
        #     new_Sig[k] = new_Sig[k].T

        # Mu_trans = self.ds_struct.Mu.T
        # new_A_k = new_A_k.reshape(-1).tolist()

        # self.original_js['Sigma'] = new_Sig.reshape(-1).tolist()
        # self.original_js['Mu'] = Mu_trans.reshape(-1).tolist()
        # self.original_js['Prior'] = self.ds_struct.Priors.tolist()
        # self.original_js['A'] = new_A_k
        # self.original_js['b'] = new_b_k.reshape(-1).tolist()
        # self.original_js['attractor']= self.att.ravel().tolist()
        # self.original_js['att_all']= self.att.ravel().tolist()
        # self.original_js["dt"] = self.dt
        # self.original_js["gripper_open"] = 0

        # write_json(self.original_js, self.js_path)


    def evaluate(self):
        rmse, e_dot, dwtd = ds_tools.reproduction_metrics(self.Data, self.A_k, self.b_k,
                                                 self.traj_length, self.x0_all, self.ds_struct)
        print("the reproduced RMSE is ", rmse)
        print("the reproduced e_dot is", e_dot)
        print("the reproduced dwtd is ", dwtd)


    def plot(self, *args_):
        Data_dim = self.M
        ds_handle = lambda x_velo: ds_tools.lpv_ds(x_velo, self.ds_struct, self.A_k, self.b_k)
        ds_opt_plot_option = structures.ds_plot_options()
        ds_opt_plot_option.attractor = self.att
        
        print(self.x0_all.shape)
        # The plotting function for lyapunov only valid for data with 2 dimension
        if Data_dim == 2:
            plot_tools.plot_lyapunov_and_derivatives(self.Data, ds_handle, self.att, self.P_opt)

        # Visualized the reproduced trajectories
        if len(args_)==0:
            ds_opt_plot_option.x0_all = self.x0_all
            plot_tools.VisualizeEstimatedDS(self.Data[:Data_dim], ds_handle, ds_opt_plot_option)
        else:
            ds_opt_plot_option.x0_all = self.x0_all
            # plot_tools.VisualizeEstimatedDS(self.Data[:Data_dim], ds_handle, ds_opt_plot_option)
            ds_opt_plot_option.x0_all = np.hstack((self.x0_all, args_[2]))
            print(ds_opt_plot_option.x0_all)
            plot_tools.plot_incremental_ds(args_[0], ds_handle, ds_opt_plot_option, args_[1])
        
            
