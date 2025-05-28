import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import argparse
from itertools import product
import multiprocessing
import json # For loading results from individual runs

def run_script_with_args(script_name, args_list, log_prefix=""):
    """Helper function to run a python script with arguments."""
    command = ["python", script_name] + args_list
    print(f"{log_prefix}Executing: {' '.join(command)}")
    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True, timeout=1800) # Added timeout
        # print(f"{log_prefix}Output from {script_name}:")
        # if process.stdout: print(process.stdout)
        if process.stderr:
            if "error" in process.stderr.lower() or "traceback" in process.stderr.lower():
                print(f"{log_prefix}Stderr from {script_name} (potential error):")
                print(process.stderr)
            # else: # Print non-error stderr for info if needed
            #     print(f"{log_prefix}Stderr (info) from {script_name}: {process.stderr[:200]}...")
        return True, process.stdout
    except subprocess.CalledProcessError as e:
        print(f"{log_prefix}Error running {script_name}:")
        print(f"  Return code: {e.returncode}")
        print(f"  Stdout: {e.stdout}")
        print(f"  Stderr: {e.stderr}")
        return False, e.stderr
    except FileNotFoundError:
        print(f"{log_prefix}Error: Script {script_name} not found.")
        return False, f"Script {script_name} not found."
    except subprocess.TimeoutExpired as e:
        print(f"{log_prefix}Error: Script {script_name} timed out after {e.timeout} seconds.")
        print(f"  Stdout: {e.stdout.decode() if e.stdout else 'None'}")
        print(f"  Stderr: {e.stderr.decode() if e.stderr else 'None'}")
        return False, f"Timeout for {script_name}"


def run_full_experiment_pipeline(config_tuple_for_pipeline):
    """
    Runs the full pipeline: data-gen -> snn-train -> robust-opt-eval
    for a single combination of sweep parameters.
    """
    sweep_config, pipeline_idx, total_pipelines, process_id_str = config_tuple_for_pipeline
    log_prefix = f"[P{process_id_str}][Exp {pipeline_idx+1}/{total_pipelines}] "
    print(f"{log_prefix}Starting pipeline for M={sweep_config['num_distinct_states_M']}, NoiseStd={sweep_config['true_channel_sobolev_noise_level_base']:.3f}, PhaseStd={sweep_config['true_channel_phase_noise_std_rad']:.3f}, L2Rad={sweep_config['L2_uncertainty_ball_radius']:.3f}")

    # --- Script Names ---
    data_gen_script = "data.py" 
    snn_train_script = "model.py"    
    robust_opt_script = "robust_opt.py"

    # --- Construct noise_config_str for filenames ---
    noise_parts = []
    if sweep_config['true_channel_apply_attenuation']: noise_parts.append(f"att{sweep_config['true_channel_attenuation_loss_factor']:.2f}")
    if sweep_config['true_channel_apply_sobolev_noise']: noise_parts.append(f"sob{sweep_config['true_channel_sobolev_noise_level_base']:.3f}s{sweep_config['true_channel_sobolev_order_s']:.1f}")
    if sweep_config['true_channel_apply_phase_noise']: noise_parts.append(f"ph{sweep_config['true_channel_phase_noise_std_rad']:.2f}")
    noise_config_str_filename = "_".join(noise_parts) if noise_parts else "no_noise"

    # --- 1. Data Generation ---
    dataset_filename = f"phenomenological_channel_dataset_Nmax{sweep_config['K_TRUNC_SNN']}_Nfull{sweep_config['k_trunc_full_fixed']}_{noise_config_str_filename}.npz"
    dataset_full_path = os.path.join(sweep_config['dataset_dir'], dataset_filename)

    if not sweep_config.get('skip_data_gen', False) or not os.path.exists(dataset_full_path):
        print(f"{log_prefix}Generating dataset: {dataset_filename}...")
        data_gen_args = [
            "--num_samples", str(sweep_config['num_samples_dataset']),
            "--n_grid_sim", str(sweep_config['n_grid_sim_dataset']),
            "--k_psi0_limit", str(sweep_config['k_psi0_limit_dataset']),
            "--k_trunc_snn", str(sweep_config['K_TRUNC_SNN']),
            "--k_trunc_full", str(sweep_config['k_trunc_full_fixed']),
            "--output_dir", sweep_config['dataset_dir'],
            *(["--apply_attenuation"] if sweep_config['true_channel_apply_attenuation'] else ["--no-apply_attenuation"]),
            "--attenuation_loss_factor", str(sweep_config['true_channel_attenuation_loss_factor']),
            *(["--apply_additive_sobolev_noise"] if sweep_config['true_channel_apply_sobolev_noise'] else ["--no-apply_additive_sobolev_noise"]),
            "--sobolev_noise_level_base", str(sweep_config['true_channel_sobolev_noise_level_base']),
            "--sobolev_order_s", str(sweep_config['true_channel_sobolev_order_s']),
            *(["--apply_phase_noise"] if sweep_config['true_channel_apply_phase_noise'] else ["--no-apply_phase_noise"]),
            "--phase_noise_std_rad", str(sweep_config['true_channel_phase_noise_std_rad'])
        ]
        success, _ = run_script_with_args(data_gen_script, data_gen_args, log_prefix)
        if not success:
            print(f"{log_prefix}Data generation FAILED. Skipping rest of pipeline for this config.")
            return {**sweep_config, "status": "data_gen_failed"}
    else:
        print(f"{log_prefix}Skipping data generation, dataset found: {dataset_full_path}")

    # --- 2. SNN Model Training ---
    snn_model_filename = f"snn_K{sweep_config['K_TRUNC_SNN']}_H{sweep_config['snn_hidden_channels']}_L{sweep_config['snn_num_hidden_layers']}_{noise_config_str_filename}.pth"
    snn_model_full_path = os.path.join(sweep_config['model_dir'], snn_model_filename)
    snn_training_plot_dir = os.path.join(sweep_config['results_dir_sweep_plots'], f"snn_training_K{sweep_config['K_TRUNC_SNN']}_{noise_config_str_filename}")

    if not sweep_config.get('skip_snn_train', False) or not os.path.exists(snn_model_full_path):
        print(f"{log_prefix}Training SNN: {snn_model_filename}...")
        os.makedirs(snn_training_plot_dir, exist_ok=True)
        train_args = [
            "--k_trunc_snn", str(sweep_config['K_TRUNC_SNN']),
            "--k_trunc_full", str(sweep_config['k_trunc_full_fixed']),
            "--dataset_dir", sweep_config['dataset_dir'], # To construct dataset filename
            "--model_save_dir", sweep_config['model_dir'],
            "--plot_save_dir", snn_training_plot_dir,
            "--snn_hidden_channels", str(sweep_config['snn_hidden_channels']),
            "--snn_num_hidden_layers", str(sweep_config['snn_num_hidden_layers']),
            "--epochs", str(sweep_config['snn_epochs']),
            # Pass noise parameters for dataset filename construction within snn_train_script
            *(["--apply_attenuation"] if sweep_config['true_channel_apply_attenuation'] else ["--no-apply_attenuation"]),
            "--attenuation_loss_factor", str(sweep_config['true_channel_attenuation_loss_factor']),
            *(["--apply_additive_sobolev_noise"] if sweep_config['true_channel_apply_sobolev_noise'] else ["--no-apply_additive_sobolev_noise"]),
            "--sobolev_noise_level_base", str(sweep_config['true_channel_sobolev_noise_level_base']),
            "--sobolev_order_s", str(sweep_config['true_channel_sobolev_order_s']),
            *(["--apply_phase_noise"] if sweep_config['true_channel_apply_phase_noise'] else ["--no-apply_phase_noise"]),
            "--phase_noise_std_rad", str(sweep_config['true_channel_phase_noise_std_rad'])
        ]
        success, _ = run_script_with_args(snn_train_script, train_args, log_prefix)
        if not success:
            print(f"{log_prefix}SNN training FAILED. Skipping robust opt for this config.")
            return {**sweep_config, "status": "snn_train_failed"}
    else:
        print(f"{log_prefix}Skipping SNN training, model found: {snn_model_full_path}")
    
    # --- 3. Run Robust Optimization and Evaluation Script ---
    # This script saves its own JSON output. We will load it after it runs.
    robust_opt_results_dir = os.path.join(sweep_config['results_dir_robust_opt_base'], 
                                          f"M{sweep_config['num_distinct_states_M']}_L2rad{sweep_config['L2_uncertainty_ball_radius']:.3f}_{noise_config_str_filename}")
    os.makedirs(robust_opt_results_dir, exist_ok=True) # robust_opt_script will save here
    
    # Filename where robust_opt_script will save its detailed JSON output
    robust_opt_output_json_filename = f"results_M{sweep_config['num_distinct_states_M']}_L2rad{sweep_config['L2_uncertainty_ball_radius']:.3f}_{noise_config_str_filename}.json"
    robust_opt_output_json_full_path = os.path.join(robust_opt_results_dir, robust_opt_output_json_filename)

    # If we want to skip running robust_opt if its output JSON already exists
    if sweep_config.get('skip_robust_opt', False) and os.path.exists(robust_opt_output_json_full_path):
        print(f"{log_prefix}Skipping robust optimization run, results file found: {robust_opt_output_json_full_path}")
    else:
        robust_opt_args = [
            "--snn_model_path", snn_model_full_path,
            "--K_TRUNC_SNN", str(sweep_config['K_TRUNC_SNN']),
            "--snn_hidden_channels", str(sweep_config['snn_hidden_channels']),
            "--snn_num_hidden_layers", str(sweep_config['snn_num_hidden_layers']),
            "--n_grid_snn_input", str(sweep_config['n_grid_snn_input']),
            "--l_domain_snn_input", str(sweep_config['l_domain_snn_input']),
            "--k_gamma0_band_limit", str(sweep_config['k_gamma0_band_limit']),
            "--delta_n_vector_gus_components", *[str(c) for c in sweep_config['delta_n_vector_gus']],
            # True channel parameters
            *(["--true_channel_apply_attenuation"] if sweep_config['true_channel_apply_attenuation'] else ["--no-true_channel_apply_attenuation"]),
            "--true_channel_attenuation_loss_factor", str(sweep_config['true_channel_attenuation_loss_factor']),
            *(["--true_channel_apply_sobolev_noise"] if sweep_config['true_channel_apply_sobolev_noise'] else ["--no-true_channel_apply_sobolev_noise"]),
            "--true_channel_sobolev_noise_level_base", str(sweep_config['true_channel_sobolev_noise_level_base']),
            "--true_channel_sobolev_order_s", str(sweep_config['true_channel_sobolev_order_s']),
            *(["--true_channel_apply_phase_noise"] if sweep_config['true_channel_apply_phase_noise'] else ["--no-true_channel_apply_phase_noise"]),
            "--true_channel_phase_noise_std_rad", str(sweep_config['true_channel_phase_noise_std_rad']),
            # Discrimination and Robust Opt parameters
            "--num_distinct_states_M", str(sweep_config['num_distinct_states_M']),
            "--L2_uncertainty_ball_radius", str(sweep_config['L2_uncertainty_ball_radius']),
            "--num_trials_per_config", str(sweep_config['num_trials_per_config_in_worker']), # Trials within the worker
            "--max_pytorch_opt_epochs", str(sweep_config['max_pytorch_opt_epochs']),
            "--pytorch_lr", str(sweep_config['pytorch_lr']),
            "--priors_q_j", *[str(p) for p in sweep_config['priors_q_j']],
            "--results_dir", robust_opt_results_dir # Directory where robust_opt_script saves its JSON
        ]
        success, _ = run_script_with_args(robust_opt_script, robust_opt_args, log_prefix)
        if not success:
            print(f"{log_prefix}Robust optimization script FAILED.")
            return {**sweep_config, "status": "robust_opt_failed"}

    # --- 4. Load and return results from the robust_opt_script's JSON output ---
    try:
        with open(robust_opt_output_json_full_path, 'r') as f:
            single_run_results = json.load(f)
        # Augment with the sweep parameters for easy aggregation
        single_run_results['sweep_config'] = sweep_config 
        single_run_results['status'] = "success"
        print(f"{log_prefix}Successfully processed and loaded results from {robust_opt_output_json_full_path}")
        return single_run_results
    except Exception as e:
        print(f"{log_prefix}Error loading results JSON from {robust_opt_output_json_full_path}: {e}")
        return {**sweep_config, "status": "result_load_failed"}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Master sweep script for robust POVM optimization with SNN.")
    # Sweepable parameters
    parser.add_argument('--M_values', nargs='+', type=int, default=[3,5], help='List of M (number of distinct states) values.')
    parser.add_argument('--L2_radius_values', nargs='+', type=float, default=[0.01], help='List of L2 uncertainty ball radii.')
    parser.add_argument('--sobolev_noise_base_values', nargs='+', type=float, default=[0.005, 0.01], help='List of Sobolev noise base levels for data gen.')
    parser.add_argument('--phase_noise_std_values', nargs='+', type=float, default=[0.01, 0.05], help='List of phase noise std devs for data gen.')
    
    # Fixed parameters for this sweep run (can be made sweepable too)
    parser.add_argument('--num_trials_worker', type=int, default=20, help="Number of trials within each worker (robust_opt script).")
    parser.add_argument('--K_TRUNC_SNN', type=int, default=80)
    parser.add_argument('--k_trunc_full_fixed', type=int, default=96) # Keep same as K_TRUNC_SNN for simplicity here
    parser.add_argument('--num_samples_dataset', type=int, default=1000)
    parser.add_argument('--n_grid_sim_dataset', type=int, default=96)
    parser.add_argument('--k_psi0_limit_dataset', type=int, default=12)
    parser.add_argument('--delta_n_x_gus', type=int, default=1) # For delta_n_vector_gus = [val, 0]
    parser.add_argument('--delta_n_y_gus', type=int, default=0)
    parser.add_argument('--snn_epochs', type=int, default=30)
    parser.add_argument('--snn_hidden_channels', type=int, default=64)
    parser.add_argument('--snn_num_hidden_layers', type=int, default=3)
    parser.add_argument('--true_channel_att_factor', type=float, default=0.2) # Fixed for this sweep
    parser.add_argument('--true_channel_sob_order_s', type=float, default=1.0) # Fixed for this sweep
    
    parser.add_argument('--max_pytorch_opt_epochs', type=int, default=300)
    parser.add_argument('--pytorch_lr', type=float, default=0.005)

    # Directories
    parser.add_argument('--dataset_base_dir', type=str, default="datasets_sweep")
    parser.add_argument('--model_base_dir', type=str, default="trained_snn_models_sweep")
    parser.add_argument('--results_robust_opt_base_dir', type=str, default="results_robust_opt_runs") # Base for individual JSONs
    parser.add_argument('--results_sweep_summary_dir', type=str, default="results_sweep_summary")
    parser.add_argument('--snn_training_plots_base_dir', type=str, default="results_snn_training_sweep")


    parser.add_argument('--skip_existing_runs', action='store_true', help="Skip experiment if its output JSON already exists.")
    parser.add_argument('--num_processes', type=int, default=min(multiprocessing.cpu_count(), 4), help="Number of parallel processes.")


    args = parser.parse_args()

    os.makedirs(args.dataset_base_dir, exist_ok=True)
    os.makedirs(args.model_base_dir, exist_ok=True)
    os.makedirs(args.results_robust_opt_base_dir, exist_ok=True)
    os.makedirs(args.results_sweep_summary_dir, exist_ok=True)
    os.makedirs(args.snn_training_plots_base_dir, exist_ok=True)

    print(f"--- Starting Master Parameter Sweep ---")

    param_configurations_for_pool = []
    # Create product of sweepable parameters
    sweep_product = list(product(
        args.M_values, 
        args.sobolev_noise_base_values, 
        args.phase_noise_std_values,
        args.L2_radius_values 
    ))
    total_experiments = len(sweep_product)

    for i_exp, (num_distinct_states_M, sob_noise, ph_noise, l2_rad) in enumerate(sweep_product):
        current_config = {
            'num_distinct_states_M': num_distinct_states_M,
            'L2_uncertainty_ball_radius': l2_rad,
            # True channel config based on sweep params
            'true_channel_apply_attenuation': True, # Assuming always apply for this example
            'true_channel_attenuation_loss_factor': args.true_channel_att_factor,
            'true_channel_apply_sobolev_noise': True, # Assuming always apply
            'true_channel_sobolev_noise_level_base': sob_noise,
            'true_channel_sobolev_order_s': args.true_channel_sob_order_s,
            'true_channel_apply_phase_noise': True, # Assuming always apply
            'true_channel_phase_noise_std_rad': ph_noise,
            # Fixed params
            'num_trials_per_config_in_worker': args.num_trials_worker,
            'K_TRUNC_SNN': args.K_TRUNC_SNN,
            'k_trunc_full_fixed': args.k_trunc_full_fixed, # For dataset naming
            'n_grid_snn_input': args.n_grid_sim_dataset,
            'l_domain_snn_input': 2 * np.pi, # Assuming fixed, from previous script
            'k_gamma0_band_limit': args.k_psi0_limit_dataset,
            'num_samples_dataset': args.num_samples_dataset,
            'n_grid_sim_dataset': args.n_grid_sim_dataset,
            'k_psi0_limit_dataset': args.k_psi0_limit_dataset,
            'delta_n_vector_gus': [args.delta_n_x_gus, args.delta_n_y_gus],
            'snn_hidden_channels': args.snn_hidden_channels,
            'snn_num_hidden_layers': args.snn_num_hidden_layers,
            'snn_epochs': args.snn_epochs,
            'max_pytorch_opt_epochs': args.max_pytorch_opt_epochs,
            'pytorch_lr': args.pytorch_lr,
            'dataset_dir': args.dataset_base_dir, # Base directory
            'model_dir': args.model_base_dir,     # Base directory
            'results_dir_robust_opt_base': args.results_robust_opt_base_dir, # Base for individual JSONs
            'results_dir_sweep_plots': args.snn_training_plots_base_dir, # For SNN training plots
            'skip_data_gen': args.skip_existing_runs, # If true, will skip if dataset for this noise config exists
            'skip_snn_train': args.skip_existing_runs, # If true, will skip if model for this noise config exists
            'skip_robust_opt': args.skip_existing_runs # If true, will skip if robust opt JSON for this config exists
        }
        if num_distinct_states_M == 3: current_config['priors_q_j'] = [0.7, 0.15, 0.15]
        elif num_distinct_states_M == 5: current_config['priors_q_j'] = [0.4,0.25,0.15,0.1,0.1]
        else: current_config['priors_q_j'] = [1.0/num_distinct_states_M]*num_distinct_states_M
        current_config['priors_q_j'] = (np.array(current_config['priors_q_j']) / np.sum(current_config['priors_q_j'])).tolist()
        
        param_configurations_for_pool.append((current_config, i_exp, total_experiments, str(os.getpid()))) # Add placeholder for process_id string

    print(f"Total experiment configurations to run: {len(param_configurations_for_pool)}")
    
    all_results_summary_list = []
    if args.num_processes > 1 and len(param_configurations_for_pool) > 0:
        print(f"Using {args.num_processes} processes for parallel execution.")
        # Re-package for pool.map to include process_id correctly if needed by worker
        # For now, os.getpid() inside worker is sufficient
        with multiprocessing.Pool(processes=args.num_processes) as pool:
            all_results_summary_list = pool.map(run_full_experiment_pipeline, param_configurations_for_pool)
    elif len(param_configurations_for_pool) > 0: 
        print("Running experiments sequentially...")
        for config_tuple_val in param_configurations_for_pool:
            # Manually pass a dummy process_id string for sequential run
            config_tuple_with_pid_str = (config_tuple_val[0], config_tuple_val[1], config_tuple_val[2], "main")
            all_results_summary_list.append(run_full_experiment_pipeline(config_tuple_with_pid_str))
            
    # Filter out failed runs before sorting and printing
    successful_results = [res for res in all_results_summary_list if res and res.get("status") == "success"]
    failed_runs = len(all_results_summary_list) - len(successful_results)
    print(f"\nNumber of successfully completed experiment configurations: {len(successful_results)}")
    if failed_runs > 0: print(f"Number of failed/skipped experiment configurations: {failed_runs}")

    if successful_results:
        successful_results.sort(key=lambda r: (r['params']['num_distinct_states_M'], 
                                               r['params']['true_channel_sobolev_noise_level_base'], 
                                               r['params']['true_channel_phase_noise_std_rad'], 
                                               r['params']['L2_uncertainty_ball_radius']))
        print("\n\n--- Master Sweep Results Summary ---")
        # LaTeX-style headers for console
        header_console = "| $M$ | $\\eta_{n}$ | $\\mu_{\\theta}$ | $p(I(\\phi_{rob}^{*}) > I(\\phi_{nom}^{*}))$ | $p(I(\\phi_{rob}^{*}) > I(\\phi_{PGM}^{*}))$ |"
        print(header_console)
        print("|" + "-"*(len(header_console)-2) + "|")
        for res in successful_results:
            cfg = res['params'] # Use 'params' as the key for config
            print(f"| {cfg['num_distinct_states_M']:<1d} | "
                  f"{cfg['true_channel_sobolev_noise_level_base']:<20.3f} | "
                  f"{cfg['true_channel_phase_noise_std_rad']:<23.3f} | "
                  f"{res.get('p_value_rob_gt_nom', float('nan')):<33.4f} | "
                  f"{res.get('p_value_rob_gt_pgm', float('nan')):<34.4f} |")

        csv_path = os.path.join(args.results_sweep_summary_dir, f"MASTER_SWEEP_SUMMARY_SNN_final_selected_cols.csv")
        with open(csv_path, 'w') as f:
            # Plain text headers for CSV
            f.write("num_distinct_states_M,SobolevNoiseBase,PhaseNoiseStd,L2Radius,p_Rob_gt_Nom,p_Rob_gt_PGM,AvgPgmIAB,AvgNominalIAB,AvgRobustIAB\n")
            for res in successful_results:
                cfg = res['params'] # Use 'params' as the key for config
                f.write(f"{cfg['num_distinct_states_M']},{cfg['true_channel_sobolev_noise_level_base']},{cfg['true_channel_phase_noise_std_rad']},{cfg['L2_uncertainty_ball_radius']},"
                        f"{res.get('p_value_rob_gt_nom', '')},{res.get('p_value_rob_gt_pgm', '')},"
                        f"{res.get('avg_pgm_IAB', '')},{res.get('avg_nominal_IAB', '')},{res.get('avg_robust_IAB', '')}\n") # Added all IABs for context
        print(f"\nSummary results saved to CSV: {csv_path}")
    else: print("No successful experiments to summarize.")
    print("\nMaster sweep complete.")
