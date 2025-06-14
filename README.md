# Conformally Robust Quantum State Discrimination
## Calibration Results
To reproduce the calibration results, run the following commands:
```
python calibration_sweep.py --pde_type step_index_fiber --k_snn_output_res_values 32 48 64 --grf_alpha_values 1.0 1.25 1.5 1.75 --n_grid_sim_input_ds 64 
python calibration_sweep.py --pde_type grin_fiber --k_snn_output_res_values 32 48 64 --grf_alpha_values 1.0 1.25 1.5 1.75 --n_grid_sim_input_ds 64 
python calibration_sweep.py --pde_type poisson --k_snn_output_res_values 4 6 8 --grf_alpha_values 0.0 0.25 0.5 0.75 --n_grid_sim_input_ds 64
python calibration_sweep.py --pde_type heat_equation --k_snn_output_res_values 32 48 64 --grf_alpha_values 1.0 1.25 1.5 1.75 --n_grid_sim_input_ds 64
```
Note that the results are stochastic by the sampling of the training and calibration data, so the exact curves may differ but
the general trends should match those presented in the paper.