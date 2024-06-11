# Robust Functional Predict-Then-Optimize
Recent interest has honed in on leveraging such neural operators for robust control in nonlinear dynamical systems, where hand-crafted models of dynamics tend to lack insufficient fidelity to produce high-quality control. However, the deployment of such neural operator-based control is practically dubious due to uncertainty in the dynamics specification.  For this reason, we propose a new method building upon conformal prediction that allows for robust decision making over the predicted function space in settings where regular mesh discretizations are present. In such cases, we demonstrate that our proposed robust method achieves a significant reduction in regret over other the nominal approach.

To reproduce results of the associated manuscript, first download the following files to the `experiments/` folder. Models can be searched and downloaded from https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2987:

###### 2D Diffusion Reaction Eqn

```
# data: 2D_diff-react_NA_NA.h5
https://darus.uni-stuttgart.de/api/access/datafile/133017

# model: 2D_diff-react_NA_NA_FNO.pt
```

-------------

###### 2D Darcy Flow Eqn

```
# data: 2D_DarcyFlow_beta1.0_Train.hdf5
https://darus.uni-stuttgart.de/api/access/datafile/133219

# model: 2D_DarcyFlow_beta10.0_Train_FNO.pt
```

------------------

###### 2D Shallow Water Eqn

```
# data: 2D_rdb_NA_NA.h5
https://darus.uni-stuttgart.de/api/access/datafile/133021

# model: 2D_rdb_NA_NA_FNO.pt
```

Experiments were run on a Nvidia RTX 2080 Ti GPU with Python 3.11.8 and PyTorch version 2.2.2+cu121 (installed via pip).