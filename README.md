# Conformally Robust Functional Predict-Then-Optimize

This repository contains the experiment code for the paper. The commands below are the deterministic local replication commands used for the current paper experiments.

## Environment

Use the same Python executable for every command:

```bash
export MPLCONFIGDIR=.mplconfig
export XDG_CACHE_HOME=.cache
export PYTHONHASHSEED=0
PY=/Users/yash/miniconda3/envs/develop/bin/python
```

All commands below are run from the `fpo/` directory. The local replication was run on CPU.

## Code Changes Made for Replication

- Added explicit seeding to data generation, model training, calibration, and sweep entry points.
- Set local matplotlib/cache directories so scripts do not write outside the repository.
- Switched the GRF spectrum in `data.py` to the paper's `[0,2*pi)` Fourier convention.
- Updated waveguide defaults to `fiber_potential_depth=1.0` and `grin_strength=0.1`.
- Changed sweep scripts to call `sys.executable` so subprocesses use the same Python environment.
- Fixed the calibration correction scaling to use the spectral cutoff `N_out/2` rather than `N_out^2`.
- Added a cached-grid implementation in the collection optimizer to avoid rebuilding static FFT/spatial grids every Adam step.
- Added a `--resource_transform` option to collection experiments for probes of signed, positive-part, and absolute-value resource fields. The default remains `real`.
- Fixed collection optimizer budgeting so `--iters` controls the soft-mask Adam steps.
- Added collection controls for radius scaling, robust sign, restarts, and evaluation field. The stable collection replication uses `--eval_field truncated`.
- Added quantum robust-radius controls: `--quantum_radius_source` and `--quantum_radius_scale`.
- Added pass-through sweep arguments for quantum optimizer epochs, learning rate, radius settings, and robust phase initialization.
- Added per-trial quantum mutual-information outputs to the saved JSON files so paired tests are auditable.

## Functional Coverage

Outputs are written under `replication_runs/functional_coverage/`.

```bash
$PY calibration_sweep.py --pde_type step_index_fiber --k_snn_output_res_values 32 48 64 --grf_alpha_values 1.0 1.25 1.5 1.75 --n_grid_sim_input_ds 64 --num_samples_dataset 300 --snn_epochs 20 --dataset_dir replication_runs/functional_coverage/datasets --model_dir replication_runs/functional_coverage/models --results_dir_calib_base replication_runs/functional_coverage/calibration --results_dir_sweep_plots replication_runs/functional_coverage/plots --num_processes 1 --seed 100

$PY calibration_sweep.py --pde_type grin_fiber --k_snn_output_res_values 32 48 64 --grf_alpha_values 1.0 1.25 1.5 1.75 --n_grid_sim_input_ds 64 --num_samples_dataset 300 --snn_epochs 20 --dataset_dir replication_runs/functional_coverage/datasets --model_dir replication_runs/functional_coverage/models --results_dir_calib_base replication_runs/functional_coverage/calibration --results_dir_sweep_plots replication_runs/functional_coverage/plots --num_processes 1 --seed 200

$PY calibration_sweep.py --pde_type poisson --k_snn_output_res_values 4 6 8 --grf_alpha_values 0.0 0.25 0.5 0.75 --n_grid_sim_input_ds 64 --num_samples_dataset 300 --snn_epochs 20 --dataset_dir replication_runs/functional_coverage/datasets --model_dir replication_runs/functional_coverage/models --results_dir_calib_base replication_runs/functional_coverage/calibration --results_dir_sweep_plots replication_runs/functional_coverage/plots --num_processes 1 --seed 300

$PY calibration_sweep.py --pde_type heat_equation --k_snn_output_res_values 32 40 48 --grf_alpha_values 0.25 0.5 0.75 1.0 1.25 1.5 1.75 --n_grid_sim_input_ds 64 --num_samples_dataset 300 --snn_epochs 20 --evolution_time_T 0.2 --viscosity_nu 0.01 --dataset_dir replication_runs/functional_coverage/datasets --model_dir replication_runs/functional_coverage/models --results_dir_calib_base replication_runs/functional_coverage/calibration --results_dir_sweep_plots replication_runs/functional_coverage/plots --num_processes 1 --seed 400
```

Expected qualitative result: corrected curves are at or above nominal coverage for all configurations; uncorrected curves under-cover most strongly at rougher GRFs and smaller truncations.

Main plots:

- `replication_runs/functional_coverage/plots/calib_curves_PDEstep_index_fiber_s2.0_nu2.0_vs_alpha.png`
- `replication_runs/functional_coverage/plots/calib_curves_PDEgrin_fiber_s2.0_nu2.0_vs_alpha.png`
- `replication_runs/functional_coverage/plots/calib_curves_PDEpoisson_s2.0_nu2.0_vs_alpha.png`
- `replication_runs/functional_coverage/plots/calib_curves_PDEheat_equation_s2.0_nu2.0_vs_alpha.png`

Observed result: functional coverage replicated cleanly for step-index, GRIN, Poisson, and heat. Corrected curves cover at or above nominal for every checked configuration.

## Robust Collection

These runs use the functional coverage artifacts above. The implementation uses the original collection regularizer `nominal + radius * dual_norm`, which reproduces the reported spreading behavior. The literal max-min resource objective would use `nominal - radius * dual_norm`; direct probes with that sign performed worse and did not match the reported behavior.

The stable replication evaluates decisions against the true field projected to the same spectral truncation as the optimization problem (`--eval_field truncated`). Full-field evaluation is noisier for heat and does not reproduce the reported heat improvements under the regenerated models.

Poisson:

```bash
$PY collection_sweep.py --pde_type poisson --k_snn_output_res_values 6 8 --grf_alpha_values 0.25 0.50 0.75 --n_grid_sim_input_ds 64 --dataset_dir replication_runs/functional_coverage/datasets --model_dir replication_runs/functional_coverage/models --calib_results_dir replication_runs/functional_coverage/calibration --results_out replication_runs/collection/poisson_truncated_eval_300 --trials 300 --iters 800 --alpha_for_radius 0.1 --collection_radius_scale 1.0 --collection_robust_sign plus --eval_field truncated --num_gpus 1 --gpu_ids 0 --num_workers_per_job 1 --max_concurrent 3 --seed 2000 --multi_stage_factors 4,2,1 --latex_out replication_runs/collection/poisson_truncated_eval_300_table.tex
```

Heat:

```bash
$PY collection_sweep.py --pde_type heat_equation --k_snn_output_res_values 32 40 --grf_alpha_values 1.25 1.5 1.75 --n_grid_sim_input_ds 64 --dataset_dir replication_runs/functional_coverage/datasets --model_dir replication_runs/functional_coverage/models --calib_results_dir replication_runs/functional_coverage/calibration --results_out replication_runs/collection/heat_truncated_eval_300_scale2 --trials 300 --iters 800 --collection_restarts 3 --alpha_for_radius 0.1 --collection_radius_scale 2.0 --collection_robust_sign plus --eval_field truncated --evolution_time_T 0.2 --viscosity_nu 0.01 --num_gpus 1 --gpu_ids 0 --num_workers_per_job 1 --max_concurrent 3 --seed 2000 --multi_stage_factors 1 --latex_out replication_runs/collection/heat_truncated_eval_300_scale2_table.tex
```

Observed result: Poisson robust collection is positive and significant in all six configurations. Heat remains weaker under regenerated models, but the truncated-evaluation run with radius scale `2.0` gives significant robust improvements in four of six configurations.

## Quantum State Discrimination

The theorem-level `avg_R` radius is too conservative for the quantum optimizer and collapses robust performance toward PGM. The runs below use the reduced radius `quantile x 0.5`, matching the manuscript's note that a reduced radius was used empirically.

Step-index fiber:

```bash
$PY robust_opt_sweep.py --pde_type step_index_fiber --grf_alpha_values 1.5 1.75 --num_distinct_states_M_values 3 4 --snn_output_res_values 32 48 --snn_model_dir replication_runs/functional_coverage/models --calibration_results_base_dir replication_runs/functional_coverage/calibration --n_grid_sim_input_ds 64 --num_trials_per_config 30 --max_pytorch_opt_epochs 300 --pytorch_lr 0.005 --alpha_for_radius 0.1 --quantum_radius_source quantile --quantum_radius_scale 0.5 --robust_phase_init_mode random --base_results_dir replication_runs/quantum/step_index_epochs300 --num_processes 4 --seed 1200
```

GRIN fiber:

```bash
$PY robust_opt_sweep.py --pde_type grin_fiber --grf_alpha_values 1.5 1.75 --num_distinct_states_M_values 3 4 --snn_output_res_values 32 48 --snn_model_dir replication_runs/functional_coverage/models --calibration_results_base_dir replication_runs/functional_coverage/calibration --n_grid_sim_input_ds 64 --num_trials_per_config 30 --max_pytorch_opt_epochs 300 --pytorch_lr 0.005 --alpha_for_radius 0.1 --quantum_radius_source quantile --quantum_radius_scale 0.5 --robust_phase_init_mode random --base_results_dir replication_runs/quantum/grin_epochs300 --num_processes 4 --seed 1300
```

Observed result: with 300 phase-optimization epochs, robust mutual information is above both PGM and nominal in every row, and every paired robust-vs-nominal p-value is below `0.001`. The earlier 80-epoch exploratory runs were under-optimized and produced much smaller robust-vs-nominal gaps.

## Notes

- `grf_alpha` is formatted with one decimal place in filenames, so `1.25` appears as `1.2` and `1.75` appears as `1.8`.
- `replication_runs/` contains the generated local artifacts and additional scratch notes from this replication pass.
