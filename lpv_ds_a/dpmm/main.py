from lpv_ds_a.dpmm.util.load_data import *
from lpv_ds_a.dpmm.util.process_data import *
from lpv_ds_a.dpmm.util.modelRegression import *  
import argparse, subprocess, os, sys, csv, random
plot = False

def dpmm(*args_):
    
    ###############################################################
    ################## command-line arguments #####################
    ###############################################################
    
    parser = argparse.ArgumentParser(
                        prog = 'Parallel Implemention of Dirichlet Process Mixture Model',
                        description = 'parallel implementation',
                        epilog = '2023, Sunan Sun <sunan@seas.upenn.edu>')


    parser.add_argument('--input', type=int, default=4, help='Choose Data Input Option: 4')
    parser.add_argument('-d', '--data', type=int, default=10, help='Choose Dataset, default=10')
    parser.add_argument('-t', '--iteration', type=int, default=40, help='Number of Sampler Iterations; default=50')
    parser.add_argument('-a', '--alpha', type=float, default = 1, help='Concentration Factor; default=1')
    parser.add_argument('--init', type=int, default = 1, help='number of initial clusters, 0 is one cluster per data; default=1')
    parser.add_argument('--base', type=int, default = 1, help='clustering option; 0: position; 1: position+directional')
    args = parser.parse_args()

    if len(sys.argv) == 1:                              # pass arguments manually
        input_opt         = 4
        dataset_no        = 8
        iteration         = 500
        alpha             = 1
        init_opt          = 15
        base              = 1
    else:                                               # pass arguments by command line
        input_opt         = args.input
        dataset_no        = args.data
        iteration         = args.iteration
        alpha             = args.alpha
        init_opt          = args.init
        base              = args.base
    
    
    ###############################################################
    ######################### load data ###########################
    ###############################################################  
    filepath = os.path.dirname(os.path.realpath(__file__))
    input_path = filepath + '/data/input.csv'
    output_path = filepath + '/data/output.csv'
    
    if len(args_) == 1:
        Data = args_[0]
    else:   
        if input_opt == 4:                                     # Using Haihui's loading/plotting code(default)
            pkg_dir = filepath + '/data/'
            chosen_dataset = dataset_no   
            sub_sample = 1   
            nb_trajectories = 4   
            Data, Data_sh, att, x0_all, data, dt = load_dataset_DS(pkg_dir, chosen_dataset, sub_sample, nb_trajectories)
            vel_samples = 10
            vel_size = 20
            plot_reference_trajectories_DS(Data, att, vel_samples, vel_size)

    Data = normalize_velocity_vector(Data)                  
    Data = Data[np.logical_not(np.isnan(Data[:, -1]))]         # get rid of nan points
    num, dim = Data.shape                                   

    with open(input_path, mode='w') as data_file:
        data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(Data.shape[0]):
            data_writer.writerow(Data[i, :])


    ###############################################################
    ####################### hyperparameters #######################
    ###############################################################  

    if base == 0:                                           # Only Eucliden distance is taken into account
        mu_0 = np.zeros((int(Data.shape[1]/2), ))
        sigma_0 = 0.1 * np.eye(mu_0.shape[0])
        lambda_0 = {
            "nu_0": sigma_0.shape[0] + 3,
            "kappa_0": 1,
            "mu_0": mu_0, 
            "sigma_0":  sigma_0
        }
    elif base == 1:
        mu_0 = np.zeros((Data.shape[1], ))
        mu_0[-1] = 1                                        # prior belief on direction; i.e. the last two entries [0, 1]
        sigma_0 = 0.1 * np.eye(int(mu_0.shape[0]/2) + 1)    # reduced dimension of covariance
        sigma_0[-1, -1] = 1                                 # scalar directional variance with no correlation with position
        lambda_0 = {
            "nu_0": sigma_0.shape[0] + 3,
            "kappa_0": 1,
            "mu_0": mu_0,
            "sigma_0":  sigma_0
        }


    ###############################################################
    ####################### perform dpmm ##########################
    ###############################################################  

    params = np.r_[np.array([lambda_0['nu_0'], lambda_0['kappa_0']]), lambda_0['mu_0'].ravel(), lambda_0['sigma_0'].ravel()]


    args = ['time ' + filepath + '/main',
            '-n {}'.format(num),
            '-m {}'.format(dim), 
            '-i {}'.format(input_path),
            '-o {}'.format(output_path),       
            '-t {}'.format(iteration),
            '-a {}'.format(alpha),
            '--init {}'.format(init_opt), 
            '--base {}'.format(base),
            '-p ' + ' '.join([str(p) for p in params])
    ]

    completed_process = subprocess.run(' '.join(args), shell=True)
    assignment_array = np.genfromtxt(filepath + '/data/output.csv', dtype=int, delimiter=',')
    logNum           = np.genfromtxt(filepath + '/data/logNum.csv', dtype=int, delimiter=',')
    logLogLik        = np.genfromtxt(filepath + '/data/logLogLik.csv', dtype=float, delimiter=',')


    ###############################################################
    ####################### plot results ##########################
    ###############################################################
    if plot:
        colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
        "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]
        
        if dim == 4:
            fig, ax = plt.subplots()
            for i in range(Data.shape[0]):
                ax.scatter(Data[i, 0], Data[i, 1], c=colors[assignment_array[i]])
            ax.set_aspect('equal')
        else:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            for k in range(assignment_array.max()+1):
                index_k = np.where(assignment_array==k)[0]
                Data_k = Data[index_k, :]
                ax.scatter(Data_k[:, 0], Data_k[:, 1], Data_k[:, 2], c=colors[k], s=5)

        assignment_array = regress(Data, assignment_array)       

        if dim == 4:
            fig, ax = plt.subplots()
            for i in range(Data.shape[0]):
                ax.scatter(Data[i, 0], Data[i, 1], c=colors[assignment_array[i]])
            ax.set_aspect('equal')
        else:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            for k in range(assignment_array.max()+1):
                index_k = np.where(assignment_array==k)[0]
                Data_k = Data[index_k, :]
                ax.scatter(Data_k[:, 0], Data_k[:, 1], Data_k[:, 2], c=colors[k], s=5)
        ax.set_title('Clustering Result: Dataset %i Base %i Init %i Iteration %i' %(dataset_no, base, init_opt, iteration))
        
        _, axes = plt.subplots(2, 1)
        axes[0].plot(np.arange(logNum.shape[0]), logNum, c='k')
        axes[0].set_title('Number of Components')
        axes[1].plot(np.arange(logLogLik.shape[0]), logLogLik, c='k')
        axes[1].set_title('Log Joint Likelihood')
        
        plt.show()

    ###############################################################
    ################# extract parameters ##########################
    ###############################################################  

    num_comp = assignment_array.max()+1
    Priors = np.zeros((num_comp, ))
    Mu = np.zeros((num_comp, int(dim/2)))
    Sigma = np.zeros((num_comp, int(dim/2), int(dim/2) ))

    for k in range(num_comp):
        data_k = Data[assignment_array==k, 0:int(dim/2)]
        Mu[k, :] = np.mean(data_k, axis=0)
        Sigma[k, :, :] = np.cov(data_k.T)
        Priors[k] = data_k.shape[0]
    Mu = Mu.T
    
    return Priors, Mu, Sigma

if __name__ == "__main__":
    
    filepath = os.path.dirname(os.path.realpath(__file__))
    pkg_dir = filepath + '/data/'
    chosen_dataset = 8
    sub_sample = 1   
    nb_trajectories = 4   
    Data, Data_sh, att, x0_all, data, dt = load_dataset_DS(pkg_dir, chosen_dataset, sub_sample, nb_trajectories)
    dpmm(Data)
