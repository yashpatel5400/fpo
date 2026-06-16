# Conformally Robust Functional Predict-Then-Optimize

This directory contains the experiment code used for the paper replication. The commands below reproduce the final selected run configuration from a clean `fpo/` working directory.

## Environment

Use the same Python executable for every command:

```bash
export MPLCONFIGDIR=.mplconfig
export XDG_CACHE_HOME=.cache
export PYTHONHASHSEED=0
PY=/Users/yash/miniconda3/envs/develop/bin/python
RUN=replication_runs/reproduce_final
FC=$RUN/functional_coverage
```

The selected local artifacts are collected in `replication_runs/final_selected_20260615/`, with the updated Poisson `N_out=8,12,16` calibration/collection run in `replication_runs/poisson_nout8_12_16_20260615/`. Large scratch and screening outputs should remain outside version control.

## Implementation Notes

- Data generation, model training, calibration, collection sweeps, and quantum sweeps are explicitly seeded.
- Sweep scripts call `sys.executable` so subprocesses use the same Python environment.
- Non-Poisson filenames use two-decimal GRF parameters. Loaders also accept legacy one-decimal artifacts where needed.
- Fiber calibration supports `--fiber_bound_type input_laplacian`, the sharper sample-dependent graph-norm tail bound used for the final T=0.4 fiber curves.
- Heat calibration uses the semigroup tail correction with the spectral cutoff `N_out/2`.
- Collection supports full-field evaluation, optimizer traces, radius scaling, restarts, and the final heat `K_facilities=7` geometry.
- Quantum outputs include per-trial PGM, nominal, and robust mutual information so paired tests are auditable.

## Functional Coverage

Fiber calibrations use `T=0.4`, `solver_num_steps=200`, and the input-Laplacian fiber tail bound:

```bash
$PY calibration_sweep.py --pde_type step_index_fiber --k_snn_output_res_values 32 48 64 --grf_alpha_values 1.0 1.25 1.5 1.75 --n_grid_sim_input_ds 64 --num_samples_dataset 300 --snn_epochs 20 --fiber_bound_type input_laplacian --evolution_time_T 0.4 --solver_num_steps 200 --dataset_dir $FC/datasets --model_dir $FC/models --results_dir_calib_base $FC/calibration_fiber_T040_input_laplacian --results_dir_sweep_plots $FC/plots_fiber_T040_input_laplacian --num_processes 1 --seed 100

$PY calibration_sweep.py --pde_type grin_fiber --k_snn_output_res_values 32 48 64 --grf_alpha_values 1.0 1.25 1.5 1.75 --n_grid_sim_input_ds 64 --num_samples_dataset 300 --snn_epochs 20 --fiber_bound_type input_laplacian --evolution_time_T 0.4 --solver_num_steps 200 --dataset_dir $FC/datasets --model_dir $FC/models --results_dir_calib_base $FC/calibration_fiber_T040_input_laplacian --results_dir_sweep_plots $FC/plots_fiber_T040_input_laplacian --num_processes 1 --seed 200
```

Poisson and heat calibrations for the final figures and downstream experiments:

```bash
$PY calibration_sweep.py --pde_type poisson --k_snn_output_res_values 8 12 16 --grf_alpha_values 0.5 0.75 1.0 1.25 --n_grid_sim_input_ds 64 --num_samples_dataset 300 --snn_epochs 20 --dataset_dir $FC/datasets --model_dir $FC/models --results_dir_calib_base $FC/calibration_poisson_nout8_12_16 --results_dir_sweep_plots $FC/plots_poisson_nout8_12_16 --num_processes 1 --seed 3100

$PY calibration_sweep.py --pde_type heat_equation --k_snn_output_res_values 48 56 64 --grf_alpha_values 1.0 1.25 1.5 1.75 --n_grid_sim_input_ds 64 --num_samples_dataset 300 --snn_epochs 20 --evolution_time_T 0.2 --viscosity_nu 0.01 --solver_num_steps 50 --dataset_dir $FC/datasets --model_dir $FC/models --results_dir_calib_base $FC/calibration_heat_semigroup_tail --results_dir_sweep_plots $FC/plots_heat_semigroup_tail --num_processes 1 --seed 400
```

Expected result: corrected curves remain close to or above nominal coverage. The selected plots are in `replication_runs/final_selected_20260615/functional_coverage/plots/`.

## Neural Operator Accuracy

These commands evaluate the trained neural operators on the held-out half of the calibration pool and write JSON/CSV/LaTeX diagnostics.

```bash
$PY operator_accuracy_sweep.py --pde_type step_index_fiber --k_snn_output_res_values 32 48 64 --grf_alpha_values 1.0 1.25 1.5 1.75 --n_grid_sim_input_ds 64 --dataset_dir $FC/datasets --model_dir $FC/models --results_dir $RUN/operator_accuracy/step_index_fiber_T040 --sobolev_orders 1,2 --evolution_time_T 0.4 --solver_num_steps 200 --cpu --no_plot

$PY operator_accuracy_sweep.py --pde_type grin_fiber --k_snn_output_res_values 32 48 64 --grf_alpha_values 1.0 1.25 1.5 1.75 --n_grid_sim_input_ds 64 --dataset_dir $FC/datasets --model_dir $FC/models --results_dir $RUN/operator_accuracy/grin_fiber_T040 --sobolev_orders 1,2 --evolution_time_T 0.4 --solver_num_steps 200 --cpu --no_plot

$PY operator_accuracy_sweep.py --pde_type poisson --k_snn_output_res_values 8 12 16 --grf_alpha_values 0.5 0.75 1.0 1.25 --n_grid_sim_input_ds 64 --dataset_dir $FC/datasets --model_dir $FC/models --results_dir $RUN/operator_accuracy/poisson_nout8_12_16 --sobolev_orders 1,2 --cpu --no_plot

$PY operator_accuracy_sweep.py --pde_type heat_equation --k_snn_output_res_values 48 56 --grf_alpha_values 1.5 1.75 --n_grid_sim_input_ds 64 --dataset_dir $FC/datasets --model_dir $FC/models --results_dir $RUN/operator_accuracy/heat_table_selected --sobolev_orders 1,2 --evolution_time_T 0.2 --viscosity_nu 0.01 --cpu --no_plot
```

## Robust Collection

Collection decisions are optimized from the truncated predictor and conformal uncertainty, then evaluated on the full simulator field with `--eval_field full`.

Poisson final run:

```bash
$PY collection_sweep.py --pde_type poisson --k_snn_output_res_values 8 12 --grf_alpha_values 1.0 1.25 --n_grid_sim_input_ds 64 --dataset_dir $FC/datasets --model_dir $FC/models --calib_results_dir $FC/calibration_poisson_nout8_12_16 --results_out $RUN/collection/poisson_nout8_12_selected_full_eval_300 --trials 300 --iters 800 --collection_restarts 3 --collection_lr 0.15 --collection_tau 1.5 --K_facilities 3 --radius_px 6 --step_px 3 --seed 2100 --alpha_for_radius 0.10 --collection_radius_scale 1.0 --multi_stage_factors 4,2,1 --resource_transform real --eval_field full --num_gpus 1 --gpu_ids 0 --num_workers_per_job 1 --max_concurrent 4 --viz_trials 0 --latex_out $RUN/collection/poisson_nout8_12_selected_full_eval_300_table.tex
```

Heat final run:

```bash
$PY collection_sweep.py --pde_type heat_equation --k_snn_output_res_values 48 56 --grf_alpha_values 1.5 1.75 --n_grid_sim_input_ds 64 --dataset_dir $FC/datasets --model_dir $FC/models --calib_results_dir $FC/calibration_heat_semigroup_tail --results_out $RUN/collection/heat_T020_modes24_28_K7_rpx6_trials300_alpha010_scale1_tau15_smooth_full --trials 300 --iters 800 --collection_restarts 3 --collection_lr 0.15 --collection_tau 1.5 --K_facilities 7 --radius_px 6 --step_px 3 --seed 0 --alpha_for_radius 0.10 --collection_radius_scale 1.0 --multi_stage_factors 1 --resource_transform real --eval_field full --s_theorem 2.0 --nu_theorem 2.0 --snn_hidden_channels 64 --snn_num_hidden_layers 3 --grf_tau 1.0 --viscosity_nu 0.01 --evolution_time_T 0.2 --solver_num_steps 50 --num_gpus 4 --gpu_ids 0 1 2 3 --num_workers_per_job 3 --max_concurrent 4 --viz_trials 0 --latex_out $RUN/collection/heat_T020_modes24_28_K7_table.tex
```

Observed selected results:

- Table 2 selected rows: `8/8` positive robust-minus-nominal differences and `7/8` one-sided paired t-test significant at `0.05`.
- Poisson selected rows: `4/4` positive and `4/4` significant for `rho in {1.0, 1.25}` and `N_out in {8, 12}`.
- Heat selected rows: `4/4` positive and `3/4` significant for `rho in {1.5, 1.75}` and `N_out in {48, 56}`.

Optimizer-trace diagnostics:

```bash
$PY collection_sweep.py --pde_type heat_equation --k_snn_output_res_values 48 --grf_alpha_values 1.5 --n_grid_sim_input_ds 64 --dataset_dir $FC/datasets --model_dir $FC/models --calib_results_dir $FC/calibration_heat_semigroup_tail --results_out $RUN/collection_stability/heat_K7_rho15_nout48_trace20 --trials 20 --iters 800 --collection_restarts 3 --collection_lr 0.15 --collection_tau 1.5 --K_facilities 7 --radius_px 6 --step_px 3 --seed 2600 --alpha_for_radius 0.10 --collection_radius_scale 1.0 --multi_stage_factors 1 --resource_transform real --eval_field full --evolution_time_T 0.2 --viscosity_nu 0.01 --save_opt_traces --trace_trials 5 --trace_every 25 --viz_trials 0 --num_gpus 1 --gpu_ids 0 --num_workers_per_job 1 --max_concurrent 1 --latex_out $RUN/collection_stability/heat_K7_rho15_nout48_trace20_table.tex

$PY collection_sweep.py --pde_type poisson --k_snn_output_res_values 12 --grf_alpha_values 1.0 --n_grid_sim_input_ds 64 --dataset_dir $FC/datasets --model_dir $FC/models --calib_results_dir $FC/calibration_poisson_nout8_12_16 --results_out $RUN/collection_stability/poisson_rho10_nout12_trace20 --trials 20 --iters 800 --collection_restarts 3 --collection_lr 0.15 --collection_tau 1.5 --K_facilities 3 --radius_px 6 --step_px 3 --seed 2500 --alpha_for_radius 0.10 --collection_radius_scale 1.0 --multi_stage_factors 4,2,1 --resource_transform real --eval_field full --save_opt_traces --trace_trials 5 --trace_every 25 --viz_trials 0 --num_gpus 1 --gpu_ids 0 --num_workers_per_job 1 --max_concurrent 1 --latex_out $RUN/collection_stability/poisson_rho10_nout12_trace20_table.tex
```

## Quantum State Discrimination

The final quantum runs use the T=0.4 fiber models/calibrations, `avg_R` as the robust radius, 300 trials per configuration, and 300 PyTorch optimization epochs.

```bash
$PY robust_opt_sweep.py --pde_type step_index_fiber --grf_alpha_values 1.5 1.75 --num_distinct_states_M_values 3 4 --snn_output_res_values 48 64 --snn_model_dir $FC/models --calibration_results_base_dir $FC/calibration_fiber_T040_input_laplacian --n_grid_sim_input_ds 64 --num_trials_per_config 300 --max_pytorch_opt_epochs 300 --pytorch_lr 0.005 --alpha_for_radius 0.1 --quantum_radius_source avg_R --quantum_radius_scale 1.0 --robust_phase_init_mode random --evolution_time_T 0.4 --solver_num_steps 200 --base_results_dir $RUN/quantum/fiber_T040_input_laplacian_trials300 --num_processes 4 --seed 8900

$PY robust_opt_sweep.py --pde_type grin_fiber --grf_alpha_values 1.5 1.75 --num_distinct_states_M_values 3 4 --snn_output_res_values 48 64 --snn_model_dir $FC/models --calibration_results_base_dir $FC/calibration_fiber_T040_input_laplacian --n_grid_sim_input_ds 64 --num_trials_per_config 300 --max_pytorch_opt_epochs 300 --pytorch_lr 0.005 --alpha_for_radius 0.1 --quantum_radius_source avg_R --quantum_radius_scale 1.0 --robust_phase_init_mode random --evolution_time_T 0.4 --solver_num_steps 200 --base_results_dir $RUN/quantum/fiber_T040_input_laplacian_trials300 --num_processes 4 --seed 9000
```

Observed selected results:

- Robust beats PGM in all `16/16` fiber quantum rows, with all one-sided paired p-values below `0.05`.
- Robust beats nominal significantly in `14/16` rows. The two non-significant rows are the step-index `rho=1.5, M=4` cases at `SNNres=48` and `SNNres=64`; they are near-ties versus nominal but still strongly above PGM.
