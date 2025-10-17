# Conformally Robust Functional Predict-Then-Optimize
## Calibration Results
To reproduce the calibration results, run the following commands:
```
python calibration_sweep.py --pde_type step_index_fiber --k_snn_output_res_values 32 48 64 --grf_alpha_values 1.0 1.25 1.5 1.75 --n_grid_sim_input_ds 64 
python calibration_sweep.py --pde_type grin_fiber --k_snn_output_res_values 32 48 64 --grf_alpha_values 1.0 1.25 1.5 1.75 --n_grid_sim_input_ds 64 
python calibration_sweep.py --pde_type poisson --k_snn_output_res_values 4 6 8 --grf_alpha_values 0.0 0.25 0.5 0.75 --n_grid_sim_input_ds 64
python calibration_sweep.py --pde_type heat_equation --k_snn_output_res_values 32 40 48 --grf_alpha_values 1.0 1.25 1.5 1.75 --n_grid_sim_input_ds 64
```
Note that the results are stochastic by the sampling of the training and calibration data, so the exact curves may differ but
the general trends should match those presented in the paper.

## Collection Problem Results
To reproduce the results from the collection problem, run (for Poisson):
```
python collection_sweep.py --pde_type poisson --k_snn_output_res_values 6 8 --grf_alpha_values 0.25 0.50 0.75 --n_grid_sim_input_ds 64 --dataset_dir datasets_sweep_final_v3 --model_dir trained_snn_models_sweep_final_v3  --calib_results_dir results_conformal_validation_sweep_final_v3 --trials 200 --alpha_for_radius 0.1 --num_gpus 8 --num_workers_per_job 4

python collection_sweep.py --pde_type heat_equation --k_snn_output_res_values 32 40 --grf_alpha_values 1.25 1.5 1.75 --n_grid_sim_input_ds 64 --dataset_dir datasets_sweep_final_v3 --model_dir trained_snn_models_sweep_final_v3  --calib_results_dir results_conformal_validation_sweep_final_v3 --trials 200 --alpha_for_radius 0.1 --num_gpus 8 --num_workers_per_job 4 --seed 10
```