# Robust Functional Predict-Then-Optimize
Recent interest has honed in on leveraging such neural operators for robust control in nonlinear dynamical systems, where hand-crafted models of dynamics tend to lack insufficient fidelity to produce high-quality control. However, the deployment of such neural operator-based control is practically dubious due to uncertainty in the dynamics specification.  For this reason, we propose a new method building upon conformal prediction that allows for robust decision making over the predicted function space in settings where regular mesh discretizations are present. In such cases, we demonstrate that our proposed robust method achieves a significant reduction in regret over other the nominal approach.

## Reproducing results
The code is primarily broken into the following files and builds upon the tremendous work in https://github.com/TSummersLab/polgrad-multinoise:
- `calibrate.py`: Used to determine conformal quantiles for the PDEs of interest. Also produces calibration curves
- `fno_utils.py`: Utility functions and class definitions of neural operators used for prediction
- `wavelet.py`: Defines helper functions to do wavelet decompositions to efficiently solve the inner maximization problem of the robust formulation
- `fpo.py`: Runs the full functional predict-then-optimize pipeline over a PDE of interest using the calibrated CP quantile of the calibration script

To reproduce results of the associated manuscript, first download the following files to the `experiments/` folder. Models can be searched and downloaded from https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2987:

###### 2D Diffusion Reaction Eqn

```
# data: 2D_diff-react_NA_NA.h5
https://darus.uni-stuttgart.de/api/access/datafile/133017

# model: 2D_diff-react_NA_NA_FNO.pt
https://darus.uni-stuttgart.de/file.xhtml?persistentId=doi:10.18419/darus-2987/9&version=2.0
```

-------------

###### 2D Darcy Flow Eqn

```
# data: 2D_DarcyFlow_beta1.0_Train.hdf5
https://darus.uni-stuttgart.de/api/access/datafile/133219

# model: 2D_DarcyFlow_beta1.0_Train_FNO.pt
https://darus.uni-stuttgart.de/file.xhtml?persistentId=doi:10.18419/darus-2987/27&version=2.0
```

------------------

###### 2D Shallow Water Eqn

```
# data: 2D_rdb_NA_NA.h5
https://darus.uni-stuttgart.de/api/access/datafile/133021

# model: 2D_rdb_NA_NA_FNO.pt
https://darus.uni-stuttgart.de/file.xhtml?persistentId=doi:10.18419/darus-2987/13&version=2.0
```

Experiments were run on a Nvidia RTX 2080 Ti GPU with Python 3.11.8 and PyTorch version 2.2.2+cu121 (installed via pip).