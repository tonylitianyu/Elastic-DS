# Dynamical System Optimization in Linear Parameter Varying Formulation

This module is a modified rendition of Haihui Gao's Python implementation of Dynamical System Optimization in Linear Parameter Varying Formulation ([DS-OPT](https://github.com/HuiTakami/ds_opt_ood)). The module has been adapted and integrated as a part of the comprehensive pipeline: Directionality-Aware Mixture Model Parallel Sampling for Efficient Dynamical System Learning. Please refer to [DAMM-LPVDS](https://github.com/SunannnSun/damm_lpvds) for complete usage.


---
### Input
The input of ds_opt consists of data dictionary of the following form:
```
data_dictionary = {
    "Data": Data,         # Data point of shape, [dimension, number_of_data]
    "Data_sh": Data_sh,   # Shifted attractor to 0 for 'Data' (used in Lyapunov function learning)
    "att": att,           # Attractor of shape, [dimension, 1]
    "x0_all": x0_all,     # Start points for all demonstrations of shape, [dimension, number_of_traj]
    "dt": dt,             # Sample time
}
```

and gmm parameters results, which should be saved in `output.json`:
```
{
    "name": "Clustering Result",
    "K": # Number of clusters,
    "M": # Dimension,
    "Priors": # List of Prior,
    "Mu": # ravel K x M shape Mu to a list,
    "Sigma": #ravel K x M x M Sigma to a list,
}
```
---
### Usage
import ds_opt class and initialize the object:
```
ds_opt = ds_opt(data_dictionary, OUTPUT_JSON_PATH)
```

Train:
```
ds_opt.begin()
```

Evaluate:
Return the rmse, e_dot, dwtd for learned trajectory
```
ds_opt.evaluate()
```

Plot: Make plots for Lyapunov derivative, value, and reproduced streamlines
```
ds_opt.make_plot()
```
![21931692138054_ pic](https://github.com/HuiTakami/ds_opt_ood/assets/97799818/7207f6f9-a93c-494d-84a3-bb691609160e)


