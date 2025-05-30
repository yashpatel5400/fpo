import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess # To call other scripts
import argparse
from itertools import product
import multiprocessing # For parallel execution
import time # For adding small delay to print statements

def run_script(script_name, args_list, log_prefix=""):
    """Helper function to run a python script with arguments."""
    command = ["python", script_name] + args_list
    max_print_len = 200 # Limit length of printed command
    command_str = ' '.join(command)
    # if len(command_str) > max_print_len:
    #     command_str = command_str[:max_print_len-3] + "..."
    print(f"{log_prefix}Executing: {command_str}")
    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True, timeout=7200) 
        if process.stderr:
            if "error" in process.stderr.lower() or \
               "traceback" in process.stderr.lower() or \
               "warning" in process.stderr.lower(): # Catch warnings too
                print(f"{log_prefix}Stderr from {script_name} (first 500 chars):")
                print(process.stderr[:])
        return True
    except subprocess.CalledProcessError as e:
        print(f"{log_prefix}Error running {script_name}:")
        print(f"  Return code: {e.returncode}")
        stdout_preview = e.stdout[:] + "..." if e.stdout and len(e.stdout) > 500 else e.stdout
        stderr_preview = e.stderr[:] + "..." if e.stderr and len(e.stderr) > 500 else e.stderr
        print(f"  Stdout: {stdout_preview}")
        print(f"  Stderr: {stderr_preview}")
        return False
    except FileNotFoundError:
        print(f"{log_prefix}Error: Script {script_name} not found.")
        return False
    except subprocess.TimeoutExpired as e:
        stdout_decoded = e.stdout.decode(errors='ignore') if e.stdout else 'None'
        stderr_decoded = e.stderr.decode(errors='ignore') if e.stderr else 'None'
        print(f"{log_prefix}Error: Script {script_name} timed out after {e.timeout} seconds.")
        print(f"  Stdout (first 500 chars): {stdout_decoded[:]}...")
        print(f"  Stderr (first 500 chars): {stderr_decoded[:]}...")
        return False

def run_single_calibration_pipeline(params_tuple):
    """
    Worker function for multiprocessing. Runs the full pipeline for one config.
    """
    k_snn_output_res, k_bound, args_namespace_obj, filename_suffix_for_run_arg, \
    data_gen_script, snn_train_script, conformal_calib_script, \
    exp_idx, total_exps_for_current_config_set = params_tuple
    
    args = argparse.Namespace(**vars(args_namespace_obj)) 
    args.grf_alpha = args_namespace_obj.grf_alpha 

    time.sleep(np.random.uniform(0, 0.1)) 
    log_prefix = f"[Worker {os.getpid()} Exp {exp_idx+1}/{total_exps_for_current_config_set} for GRF_alpha={args.grf_alpha:.1f}, K_SNN_Out={k_snn_output_res}, K_Bound={k_bound}] "
    print(f"{log_prefix}Processing...")

    dataset_filename = f"dataset_{args.pde_type}_Nin{args.n_grid_sim_input_ds}_Nout{k_snn_output_res}_{filename_suffix_for_run_arg}.npz"
    dataset_full_path = os.path.join(args.dataset_dir, dataset_filename)

    snn_model_filename = f"snn_PDE{args.pde_type}_Kin{args.n_grid_sim_input_ds}_Kout{k_snn_output_res}_H{args.snn_hidden_channels}_L{args.snn_num_hidden_layers}_{filename_suffix_for_run_arg}.pth"
    snn_model_full_path = os.path.join(args.model_dir, snn_model_filename)
    
    snn_training_plot_dir = os.path.join(args.results_dir_sweep_plots, f"snn_training_PDE{args.pde_type}_Kin{args.n_grid_sim_input_ds}_Kout{k_snn_output_res}_{filename_suffix_for_run_arg}")

    calib_results_subdir_name = (f"PDE{args.pde_type}_Nin{args.n_grid_sim_input_ds}_SNNres{k_snn_output_res}_"
                                 f"KfullThm{args.n_grid_sim_input_ds}_KboundB{k_bound}_s{args.theorem_s}_nu{args.theorem_nu}_{filename_suffix_for_run_arg}")
    calib_results_subdir = os.path.join(args.results_dir_calib_base, calib_results_subdir_name)
    
    coverage_data_filename_npz = os.path.join(
        calib_results_subdir, 
        f"coverage_data_PDE{args.pde_type}_thm_s{args.theorem_s}_nu{args.theorem_nu}_d{args.theorem_d}"
        f"_Nin{args.n_grid_sim_input_ds}_SNNout{k_snn_output_res}_NfullThm{args.n_grid_sim_input_ds}_KboundB{k_bound}"
        f"_{filename_suffix_for_run_arg}.npz"
    )

    if not args.skip_data_gen or not os.path.exists(dataset_full_path):
        print(f"{log_prefix}Generating dataset: {dataset_filename}...")
        data_gen_args_list = args.base_sub_script_args_for_current_pde + [
            "--num_samples", str(args.num_samples_dataset), 
            "--n_grid_sim_input", str(args.n_grid_sim_input_ds), 
            "--k_psi0_limit", str(args.k_psi0_limit_dataset), 
            "--k_trunc_snn_output", str(k_snn_output_res), 
            "--output_dir", args.dataset_dir
        ]
        if not run_script(data_gen_script, data_gen_args_list, log_prefix):
            print(f"{log_prefix}Data generation FAILED for {dataset_filename}. Returning None.")
            return None
    else:
        print(f"{log_prefix}Skipping data generation, dataset found: {dataset_full_path}")

    if not args.skip_snn_train or not os.path.exists(snn_model_full_path):
        print(f"{log_prefix}Training SNN: {snn_model_filename}...")
        os.makedirs(snn_training_plot_dir, exist_ok=True)
        train_args_list = args.base_sub_script_args_for_current_pde + [
            "--n_grid_sim_input_ds", str(args.n_grid_sim_input_ds), 
            "--k_snn_target_res", str(k_snn_output_res),          
            "--dataset_dir", args.dataset_dir, 
            "--model_save_dir", args.model_dir,
            "--plot_save_dir", snn_training_plot_dir, 
            "--snn_hidden_channels", str(args.snn_hidden_channels),
            "--snn_num_hidden_layers", str(args.snn_num_hidden_layers), 
            "--epochs", str(args.snn_epochs),
        ]
        if not run_script(snn_train_script, train_args_list, log_prefix):
            print(f"{log_prefix}SNN training FAILED for {snn_model_filename}. Returning None.")
            return None
    else:
        print(f"{log_prefix}Skipping SNN training, model found: {snn_model_full_path}")
            
    os.makedirs(calib_results_subdir, exist_ok=True)
    calib_args_list = args.base_sub_script_args_for_current_pde + [
        "--n_grid_sim_input_ds", str(args.n_grid_sim_input_ds), 
        "--k_snn_output_res", str(k_snn_output_res), 
        "--snn_model_dir", args.model_dir, 
        "--snn_hidden_channels", str(args.snn_hidden_channels), 
        "--snn_num_hidden_layers", str(args.snn_num_hidden_layers),
        "--snn_model_filename_override", snn_model_filename, 
        "--dataset_dir", args.dataset_dir, 
        "--results_dir", calib_results_subdir, 
        "--s_theorem", str(args.theorem_s), 
        "--nu_theorem", str(args.theorem_nu),
        "--d_dimensions", str(args.theorem_d), 
        "--k_trunc_bound", str(k_bound), 
        "--no_plot" 
    ]
    if not run_script(conformal_calib_script, calib_args_list, log_prefix):
        print(f"{log_prefix}Conformal calibration FAILED for SNN_Output_Res={k_snn_output_res}, K_BOUND={k_bound}. Returning None.")
        return None
    
    try:
        coverage_data = np.load(coverage_data_filename_npz)
        result_key = (k_snn_output_res, k_bound, args.pde_type, args.grf_alpha, filename_suffix_for_run_arg)
        print(f"{log_prefix}Successfully processed results for key: {result_key}")
        return result_key, { 
            "nominal_coverages": coverage_data["nominal_coverages"],
            "empirical_coverages_theorem": coverage_data["empirical_coverages_theorem"]
        }
    except Exception as e:
        print(f"{log_prefix}Error loading coverage data for K_SNN_IO={k_snn_output_res}, K_BOUND={k_bound} from {coverage_data_filename_npz}: {e}")
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sweep for conformal calibration: SNN I/O Res, K_TRUNC_BOUND, GRF Alpha.")
    
    # --- PDE Type ---
    parser.add_argument('--pde_type', type=str, default="poisson", 
                        choices=["phenomenological_channel", "poisson"],
                        help="Type of data generation process for the dataset.")
    
    # --- Sweep Parameters ---
    parser.add_argument('--k_snn_output_res_values', nargs='+', type=int, default=[32],
                        help='List of SNN output resolutions to sweep over.')
    parser.add_argument('--k_trunc_bound_values', nargs='+', type=int, default=[32, 48, 64],
                        help='List of k_trunc_bound values for B_sq calculation in calibration.')
    parser.add_argument('--grf_alpha_values', nargs='+', type=float, default=[2.5, 4.0],
                        help="List of GRF alpha values to sweep over.")
    
    # --- Fixed Parameters for Dataset and SNN Structure ---
    parser.add_argument('--n_grid_sim_input_ds', type=int, default=64,
                        help='Resolution for full input spectrum (Nin) in dataset generation AND N_full for theorem evaluation.')
    parser.add_argument('--num_samples_dataset', type=int, default=200) 
    parser.add_argument('--k_psi0_limit_dataset', type=int, default=12,
                        help="K_psi0_band_limit for random_low_order_state (if phenom. channel and not using GRF input).")
    
    parser.add_argument('--snn_epochs', type=int, default=20) 
    parser.add_argument('--snn_hidden_channels', type=int, default=64)
    parser.add_argument('--snn_num_hidden_layers', type=int, default=3)

    # --- Fixed Theorem Parameters ---
    parser.add_argument('--theorem_s', type=float, default=2.0)
    parser.add_argument('--theorem_nu', type=float, default=2.0) 
    parser.add_argument('--theorem_d', type=int, default=2)

    # --- Noise Channel Parameters (if pde_type is phenomenological_channel) ---
    parser.add_argument('--use_grf_for_phenom_input', action=argparse.BooleanOptionalAction, default=True)
    
    parser.add_argument('--apply_attenuation', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--attenuation_loss_factor', type=float, default=0.2)
    
    parser.add_argument('--apply_additive_sobolev_noise', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--sobolev_noise_level_base', type=float, default=0.01)
    parser.add_argument('--sobolev_order_s', type=float, default=1.0)
    
    parser.add_argument('--apply_phase_noise', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--phase_noise_std_rad', type=float, default=0.05)
    
    # --- GRF Parameters (used for Poisson, and for phenom. if use_grf_for_phenom_input) ---
    parser.add_argument('--grf_tau', type=float, default=1.0)   
    parser.add_argument('--grf_offset_sigma', type=float, default=0.5)

    # --- Directories and Control Flags ---
    parser.add_argument('--dataset_dir', type=str, default="datasets_sweep_grf_alpha")
    parser.add_argument('--model_dir', type=str, default="trained_snn_models_sweep_grf_alpha")
    parser.add_argument('--results_dir_calib_base', type=str, default="results_conformal_validation_sweep_grf_alpha")
    parser.add_argument('--results_dir_sweep_plots', type=str, default="results_calibration_sweep_plots_grf_alpha")
    
    parser.add_argument('--skip_data_gen', action='store_true')
    parser.add_argument('--skip_snn_train', action='store_true')
    parser.add_argument('--skip_completed_calib_runs', action='store_true')
    parser.add_argument('--num_processes', type=int, default=min(os.cpu_count(), 2))

    args = parser.parse_args()

    if any(k_snn_out > args.n_grid_sim_input_ds for k_snn_out in args.k_snn_output_res_values):
        print(f"Error: Some k_snn_output_res_values are > n_grid_sim_input_ds ({args.n_grid_sim_input_ds}).")
        exit()

    os.makedirs(args.dataset_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir_calib_base, exist_ok=True)
    os.makedirs(args.results_dir_sweep_plots, exist_ok=True)

    all_sweep_results_dict = {} 
    data_gen_script_name = "data.py"
    snn_train_script_name = "model.py"
    conformal_calib_script_name = "calibration.py" 

    # Outer loop for GRF Alpha values
    for current_grf_alpha_sweep_val in args.grf_alpha_values:
        print(f"\n\nPROCESSING FIGURE FOR GRF ALPHA = {current_grf_alpha_sweep_val:.1f}\n" + "="*80)
        
        current_iter_args = argparse.Namespace(**vars(args))
        current_iter_args.grf_alpha = current_grf_alpha_sweep_val 

        current_filename_suffix = ""
        current_base_sub_script_args = ["--pde_type", current_iter_args.pde_type] 
        
        if current_iter_args.pde_type == "phenomenological_channel":
            noise_parts = []
            if current_iter_args.use_grf_for_phenom_input: 
                noise_parts.append(f"grfInA{current_iter_args.grf_alpha:.1f}T{current_iter_args.grf_tau:.1f}")
            if current_iter_args.apply_attenuation:
                noise_parts.append(f"att{current_iter_args.attenuation_loss_factor:.2f}")
            if current_iter_args.apply_additive_sobolev_noise:
                noise_parts.append(f"sob{current_iter_args.sobolev_noise_level_base:.3f}s{current_iter_args.sobolev_order_s:.1f}")
            if current_iter_args.apply_phase_noise:
                noise_parts.append(f"ph{current_iter_args.phase_noise_std_rad:.2f}")
            current_filename_suffix = "_".join(noise_parts) if noise_parts else "no_noise_or_grf_input"
            
            current_base_sub_script_args.extend([
                *(["--use_grf_for_phenom_input"] if current_iter_args.use_grf_for_phenom_input else []),
                *(["--grf_alpha", str(current_iter_args.grf_alpha), "--grf_tau", str(current_iter_args.grf_tau)] if current_iter_args.use_grf_for_phenom_input else []),
                *(["--apply_attenuation"] if current_iter_args.apply_attenuation else ["--no-apply_attenuation"]), 
                "--attenuation_loss_factor", str(current_iter_args.attenuation_loss_factor),
                *(["--apply_additive_sobolev_noise"] if current_iter_args.apply_additive_sobolev_noise else ["--no-apply_additive_sobolev_noise"]), 
                "--sobolev_noise_level_base", str(current_iter_args.sobolev_noise_level_base), 
                "--sobolev_order_s", str(current_iter_args.sobolev_order_s),
                *(["--apply_phase_noise"] if current_iter_args.apply_phase_noise else ["--no-apply_phase_noise"]), 
                "--phase_noise_std_rad", str(current_iter_args.phase_noise_std_rad)
            ])
        elif current_iter_args.pde_type == "poisson":
            current_filename_suffix = f"poisson_grfA{current_iter_args.grf_alpha:.1f}T{current_iter_args.grf_tau:.1f}"
            current_base_sub_script_args.extend([
                "--grf_alpha", str(current_iter_args.grf_alpha), 
                "--grf_tau", str(current_iter_args.grf_tau), 
                "--grf_offset_sigma", str(current_iter_args.grf_offset_sigma)
            ])
        
        current_iter_args.filename_suffix_for_this_run = current_filename_suffix 
        current_iter_args.base_sub_script_args_for_current_pde = current_base_sub_script_args
        
        print(f"  Using PDE type '{current_iter_args.pde_type}' with suffix '{current_iter_args.filename_suffix_for_this_run}'")

        for k_snn_io_res_current in current_iter_args.k_snn_output_res_values:
            print(f"\n    --- Preparing for SNN I/O Resolution (N_out) = {k_snn_io_res_current} ---")
            
            dataset_filename = f"dataset_{current_iter_args.pde_type}_Nin{current_iter_args.n_grid_sim_input_ds}_Nout{k_snn_io_res_current}_{current_iter_args.filename_suffix_for_this_run}.npz"
            dataset_full_path = os.path.join(current_iter_args.dataset_dir, dataset_filename)
            snn_model_filename = f"snn_PDE{current_iter_args.pde_type}_Kin{current_iter_args.n_grid_sim_input_ds}_Kout{k_snn_io_res_current}_H{current_iter_args.snn_hidden_channels}_L{current_iter_args.snn_num_hidden_layers}_{current_iter_args.filename_suffix_for_this_run}.pth"
            snn_model_full_path = os.path.join(current_iter_args.model_dir, snn_model_filename)
            snn_training_plot_dir = os.path.join(current_iter_args.results_dir_sweep_plots, f"snn_training_PDE{current_iter_args.pde_type}_Kin{current_iter_args.n_grid_sim_input_ds}_Kout{k_snn_io_res_current}_{current_iter_args.filename_suffix_for_this_run}")

            if not current_iter_args.skip_data_gen or not os.path.exists(dataset_full_path):
                print(f"      Generating dataset: {dataset_filename}...")
                data_gen_args_list = current_iter_args.base_sub_script_args_for_current_pde + [
                    "--num_samples", str(current_iter_args.num_samples_dataset), 
                    "--n_grid_sim_input", str(current_iter_args.n_grid_sim_input_ds), 
                    "--k_psi0_limit", str(current_iter_args.k_psi0_limit_dataset), 
                    "--k_trunc_snn_output", str(k_snn_io_res_current), 
                    "--output_dir", current_iter_args.dataset_dir
                ]
                if not run_script(data_gen_script_name, data_gen_args_list, f"      [GRF_A={current_grf_alpha_sweep_val:.1f}, K_SNN_Out={k_snn_io_res_current}] "):
                    continue
            else:
                print(f"      Skipping data generation, dataset found: {dataset_full_path}")

            if not current_iter_args.skip_snn_train or not os.path.exists(snn_model_full_path):
                print(f"      Training SNN: {snn_model_filename}...")
                os.makedirs(snn_training_plot_dir, exist_ok=True)
                train_args_list = current_iter_args.base_sub_script_args_for_current_pde + [
                    "--n_grid_sim_input_ds", str(current_iter_args.n_grid_sim_input_ds), 
                    "--k_snn_target_res", str(k_snn_io_res_current),          
                    "--dataset_dir", current_iter_args.dataset_dir, 
                    "--model_save_dir", current_iter_args.model_dir,
                    "--plot_save_dir", snn_training_plot_dir, 
                    "--snn_hidden_channels", str(current_iter_args.snn_hidden_channels),
                    "--snn_num_hidden_layers", str(current_iter_args.snn_num_hidden_layers), 
                    "--epochs", str(current_iter_args.snn_epochs),
                ]
                if not run_script(snn_train_script_name, train_args_list, f"      [GRF_A={current_grf_alpha_sweep_val:.1f}, K_SNN_Out={k_snn_io_res_current}] "):
                    continue
            else:
                print(f"      Skipping SNN training, model found: {snn_model_full_path}")

            calibration_tasks_for_this_set = []
            for i_calib_task, k_bound_current_val in enumerate(current_iter_args.k_trunc_bound_values):
                # CORRECTED TUPLE: Added missing script names
                calibration_tasks_for_this_set.append(
                    (k_snn_io_res_current, k_bound_current_val, current_iter_args, 
                     current_iter_args.filename_suffix_for_this_run, 
                     data_gen_script_name,  # Added
                     snn_train_script_name, # Added
                     conformal_calib_script_name,
                     i_calib_task, len(current_iter_args.k_trunc_bound_values))
                )
            
            print(f"      Starting {len(calibration_tasks_for_this_set)} calibration tasks for K_SNN_IO_Res={k_snn_io_res_current} using {current_iter_args.num_processes} processes...")
            
            current_k_snn_results_list = []
            if current_iter_args.num_processes > 1 and len(calibration_tasks_for_this_set) > 0:
                with multiprocessing.Pool(processes=current_iter_args.num_processes) as pool:
                    current_k_snn_results_list = pool.map(run_single_calibration_pipeline, calibration_tasks_for_this_set)
            elif len(calibration_tasks_for_this_set) > 0:
                current_k_snn_results_list = [run_single_calibration_pipeline(params) for params in calibration_tasks_for_this_set]

            for result_item in current_k_snn_results_list:
                if result_item: 
                    key, data = result_item
                    all_sweep_results_dict[key] = data 
    
    # --- Plotting Section (after ALL experiments are done) ---
    print("\n\n--- Aggregating and Plotting All Calibration Curves ---")
    
    if not all_sweep_results_dict:
        print("No results to plot.")
    else:
        # Determine primary sweep variable for plotting based on which list has more than one unique value
        plot_sweep_var_is_grf_alpha = len(set(args.grf_alpha_values)) > 1 and len(set(args.k_trunc_bound_values)) == 1
        plot_sweep_var_is_k_bound = len(set(args.k_trunc_bound_values)) > 1 and len(set(args.grf_alpha_values)) == 1
        plot_both_sweeping = len(set(args.grf_alpha_values)) > 1 and len(set(args.k_trunc_bound_values)) > 1
        
        if plot_sweep_var_is_grf_alpha or plot_sweep_var_is_k_bound or (not plot_both_sweeping and (len(set(args.grf_alpha_values)) >=1 and len(set(args.k_trunc_bound_values)) >=1 ) ): # Single figure with subplots
            
            primary_sweep_values_plot = []
            subplot_var_label_plot = ""
            figure_title_fixed_var_plot = ""
            plot_filename_tag_plot = ""

            if plot_sweep_var_is_grf_alpha:
                primary_sweep_values_plot = sorted(list(set(args.grf_alpha_values)))
                subplot_var_label_plot = "GRF $\\alpha$"
                figure_title_fixed_var_plot = f"$k_{{bound}}={args.k_trunc_bound_values[0]}$"
                plot_filename_tag_plot = f"vs_GRFalpha_Kbound{args.k_trunc_bound_values[0]}"
            elif plot_sweep_var_is_k_bound:
                primary_sweep_values_plot = sorted(list(set(args.k_trunc_bound_values)))
                subplot_var_label_plot = "$k_{bound}$"
                figure_title_fixed_var_plot = f"GRF $\\alpha={args.grf_alpha_values[0]:.1f}$"
                plot_filename_tag_plot = f"vs_Kbound_GRFalpha{args.grf_alpha_values[0]:.1f}"
            else: # Both have length 1, or neither is explicitly a sweep for subplots
                primary_sweep_values_plot = [args.k_trunc_bound_values[0]] # Default to k_bound as primary for structure
                subplot_var_label_plot = "$k_{bound}$"
                figure_title_fixed_var_plot = f"GRF $\\alpha={args.grf_alpha_values[0]:.1f}$"
                plot_filename_tag_plot = f"single_GRFalpha{args.grf_alpha_values[0]:.1f}_Kbound{args.k_trunc_bound_values[0]}"

            num_subplots_plot = len(primary_sweep_values_plot)
            if num_subplots_plot == 0:
                print("No primary sweep values to plot.")
            else:
                ncols_fig = num_subplots_plot if num_subplots_plot <= 4 else int(np.ceil(np.sqrt(num_subplots_plot))) # Max 4 cols for readability
                nrows_fig = 1 if num_subplots_plot <= 4 else int(np.ceil(num_subplots_plot / ncols_fig))
                if nrows_fig == 0: nrows_fig = 1
                if ncols_fig == 0: ncols_fig = 1

                fig, axes = plt.subplots(nrows_fig, ncols_fig, 
                                         figsize=(ncols_fig * 6, nrows_fig * 5.5), 
                                         sharex=True, sharey=True, squeeze=False)
                axes_flat = axes.flatten()

                for i_plot, current_primary_val in enumerate(primary_sweep_values_plot):
                    ax = axes_flat[i_plot]
                    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal Coverage')
                    
                    # Determine the fixed secondary value and the suffix for this plot
                    current_grf_alpha_for_key = current_primary_val if plot_sweep_var_is_grf_alpha else args.grf_alpha_values[0]
                    current_k_bound_for_key = current_primary_val if plot_sweep_var_is_k_bound else args.k_trunc_bound_values[0]
                    if not plot_sweep_var_is_grf_alpha and not plot_sweep_var_is_k_bound: # Single config plot
                        current_grf_alpha_for_key = args.grf_alpha_values[0]
                        current_k_bound_for_key = args.k_trunc_bound_values[0]


                    plot_iter_filename_suffix = "" 
                    if args.pde_type == "phenomenological_channel":
                        _parts = []
                        if args.use_grf_for_phenom_input: _parts.append(f"grfInA{current_grf_alpha_for_key:.1f}T{args.grf_tau:.1f}")
                        if args.apply_attenuation: _parts.append(f"att{args.attenuation_loss_factor:.2f}")
                        if args.apply_additive_sobolev_noise: _parts.append(f"sob{args.sobolev_noise_level_base:.3f}s{args.sobolev_order_s:.1f}")
                        if args.apply_phase_noise: _parts.append(f"ph{args.phase_noise_std_rad:.2f}")
                        plot_iter_filename_suffix = "_".join(_parts) if _parts else "no_noise_or_grf_input"
                    elif args.pde_type == "poisson":
                        plot_iter_filename_suffix = f"poisson_grfA{current_grf_alpha_for_key:.1f}T{args.grf_tau:.1f}"

                    has_data = False
                    for k_snn_val in sorted(args.k_snn_output_res_values):
                        result_key = (k_snn_val, current_k_bound_for_key, args.pde_type, current_grf_alpha_for_key, plot_iter_filename_suffix)
                        if result_key in all_sweep_results_dict:
                            data = all_sweep_results_dict[result_key]
                            ax.plot(data["nominal_coverages"], data["empirical_coverages_theorem"], 
                                    marker='o', linestyle='-', markersize=4, label=f'$N_{{max}}$={k_snn_val}')
                            has_data = True
                    
                    ax.set_xlabel("Nominal Coverage ($1-\\alpha$)")
                    if i_plot % ncols_fig == 0: ax.set_ylabel("Empirical Coverage") # Corrected y-axis label
                    title_val_str = f"{current_primary_val:.1f}" if isinstance(current_primary_val, float) else str(current_primary_val)
                    ax.set_title(f"{subplot_var_label_plot} = {title_val_str}")
                    ax.legend(fontsize='small', loc='lower right'); ax.grid(True); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
                    if not has_data: ax.text(0.5,0.5,"No data",ha='center',va='center',transform=ax.transAxes)
                
                for j_ax_hide in range(num_subplots_plot, nrows_fig * ncols_fig):
                    if j_ax_hide < len(axes_flat): fig.delaxes(axes_flat[j_ax_hide])
                
                fig.suptitle(f"Calibration Curves Under Correction (PDE: {args.pde_type}, Fixed: {figure_title_fixed_var_plot})", fontsize=16) # Corrected title
                plt.tight_layout() # Corrected: removed rect
                
                combined_plot_filename = f"calib_curves_PDE{args.pde_type}_s{args.theorem_s}_nu{args.theorem_nu}_{plot_filename_tag_plot}.png"
                combined_plot_full_path = os.path.join(args.results_dir_sweep_plots, combined_plot_filename)
                plt.savefig(combined_plot_full_path)
                print(f"\nPlot for fixed {figure_title_fixed_var_plot} saved to {combined_plot_full_path}")
                plt.show()

        elif plot_both_sweeping: # Both grf_alpha and k_trunc_bound are lists with >1 element
            print("Plotting for each GRF Alpha value (subplots for K_TRUNC_BOUND)...")
            for grf_alpha_for_fig in sorted(list(set(args.grf_alpha_values))):
                # ... (This is the plotting logic from the previous version, which is correct for this case) ...
                # ... It creates one figure per grf_alpha, with subplots for k_trunc_bound.
                # ... The legend entry $N_{SNNout}$ should be $N_{max}$
                current_plot_filename_suffix = "" 
                if args.pde_type == "phenomenological_channel":
                    noise_parts_plot = []; 
                    if args.use_grf_for_phenom_input: noise_parts_plot.append(f"grfInA{grf_alpha_for_fig:.1f}T{args.grf_tau:.1f}")
                    if args.apply_attenuation: noise_parts_plot.append(f"att{args.attenuation_loss_factor:.2f}")
                    if args.apply_additive_sobolev_noise: noise_parts_plot.append(f"sob{args.sobolev_noise_level_base:.3f}s{args.sobolev_order_s:.1f}")
                    if args.apply_phase_noise: noise_parts_plot.append(f"ph{args.phase_noise_std_rad:.2f}")
                    current_plot_filename_suffix = "_".join(noise_parts_plot) if noise_parts_plot else "no_noise_or_grf_input"
                elif args.pde_type == "poisson":
                    current_plot_filename_suffix = f"poisson_grfA{grf_alpha_for_fig:.1f}T{args.grf_tau:.1f}"

                primary_sweep_values_plot = sorted(list(set(args.k_trunc_bound_values)))
                num_subplots_plot = len(primary_sweep_values_plot)
                subplot_var_label_plot = "$k_{bound}$"
                figure_title_fixed_var_plot = f"GRF $\\alpha={grf_alpha_for_fig:.1f}$"
                plot_filename_tag_plot = f"GRFalpha{grf_alpha_for_fig:.1f}"

                if num_subplots_plot == 0: continue
                ncols_fig = num_subplots_plot if num_subplots_plot <= 3 else int(np.ceil(np.sqrt(num_subplots_plot)))
                nrows_fig = 1 if num_subplots_plot <= 3 else int(np.ceil(num_subplots_plot / ncols_fig))
                if nrows_fig == 0: nrows_fig = 1; 
                if ncols_fig == 0: ncols_fig = 1;

                fig, axes = plt.subplots(nrows_fig, ncols_fig, figsize=(ncols_fig * 5.5, nrows_fig * 5), sharex=True, sharey=True, squeeze=False)
                axes_flat = axes.flatten()
                plot_successful_for_this_fig = False
                for i_plot, current_primary_val in enumerate(primary_sweep_values_plot): # current_primary_val is k_bound here
                    ax = axes_flat[i_plot]
                    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal Coverage')
                    has_data_for_this_subplot = False
                    for k_snn_val in sorted(args.k_snn_output_res_values):
                        result_key = (k_snn_val, current_primary_val, args.pde_type, grf_alpha_for_fig, current_plot_filename_suffix)
                        if result_key in all_sweep_results_dict:
                            data = all_sweep_results_dict[result_key]
                            ax.plot(data["nominal_coverages"], data["empirical_coverages_theorem"], marker='o', linestyle='-', markersize=4, label=f'$N_{{max}}$={k_snn_val}') # Corrected Legend
                            has_data_for_this_subplot = True; plot_successful_for_this_fig = True
                    ax.set_xlabel("Nominal Coverage ($1-\\alpha$)")
                    if i_plot % ncols_fig == 0: ax.set_ylabel("Empirical Coverage") # Corrected Y-label
                    title_val_str = f"{current_primary_val:.1f}" if isinstance(current_primary_val, float) else str(current_primary_val)
                    ax.set_title(f"{subplot_var_label_plot} = {title_val_str}")
                    ax.legend(fontsize='small', loc='lower right'); ax.grid(True); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
                    if not has_data_for_this_subplot: ax.text(0.5,0.5,"No data",ha='center',va='center',transform=ax.transAxes)
                for j_ax_hide in range(num_subplots_plot, nrows_fig * ncols_fig):
                    if j_ax_hide < len(axes_flat): fig.delaxes(axes_flat[j_ax_hide])
                if plot_successful_for_this_fig:
                    fig.suptitle(f"Calibration Curves Under Correction (PDE: {args.pde_type}, Fixed: {figure_title_fixed_var_plot})", fontsize=16) # Corrected Title
                    plt.tight_layout() # Corrected: removed rect
                    combined_plot_filename = f"calib_curves_PDE{args.pde_type}_s{args.theorem_s}_nu{args.theorem_nu}_{plot_filename_tag_plot}_{current_plot_filename_suffix}.png"
                    combined_plot_full_path = os.path.join(args.results_dir_sweep_plots, combined_plot_filename)
                    plt.savefig(combined_plot_full_path); print(f"\nPlot for {figure_title_fixed_var_plot} saved to {combined_plot_full_path}"); plt.show()
                else: plt.close(fig); print(f"  No data to plot for {figure_title_fixed_var_plot}.")
        else: # Single plot scenario (only one grf_alpha and one k_trunc_bound)
            # ... (single plot logic as before, ensuring title, labels, legend are correct) ...
            fig, ax = plt.subplots(1, 1, figsize=(6, 5.5)) 
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal Coverage')
            has_data = False; fixed_grf_alpha_plot = args.grf_alpha_values[0]; fixed_k_bound_plot = args.k_trunc_bound_values[0]
            plot_filename_suffix_specific = ""
            if args.pde_type == "phenomenological_channel":
                _parts = []; 
                if args.use_grf_for_phenom_input: _parts.append(f"grfInA{fixed_grf_alpha_plot:.1f}T{args.grf_tau:.1f}")
                if args.apply_attenuation: _parts.append(f"att{args.attenuation_loss_factor:.2f}")
                if args.apply_additive_sobolev_noise: _parts.append(f"sob{args.sobolev_noise_level_base:.3f}s{args.sobolev_order_s:.1f}")
                if args.apply_phase_noise: _parts.append(f"ph{args.phase_noise_std_rad:.2f}")
                plot_filename_suffix_specific = "_".join(_parts) if _parts else "no_noise_or_grf_input"
            elif args.pde_type == "poisson": plot_filename_suffix_specific = f"poisson_grfA{fixed_grf_alpha_plot:.1f}T{args.grf_tau:.1f}"
            for k_snn_val in sorted(args.k_snn_output_res_values):
                result_key = (k_snn_val, fixed_k_bound_plot, args.pde_type, fixed_grf_alpha_plot, plot_filename_suffix_specific)
                if result_key in all_sweep_results_dict:
                    data = all_sweep_results_dict[result_key]
                    ax.plot(data["nominal_coverages"], data["empirical_coverages_theorem"], marker='o', linestyle='-', markersize=4, label=f'$N_{{max}}$={k_snn_val}') # Corrected Legend
                    has_data = True
            ax.set_xlabel("Nominal Coverage ($1-\\alpha$)"); ax.set_ylabel("Empirical Coverage") # Corrected Y-label
            ax.set_title(f"GRF $\\alpha={fixed_grf_alpha_plot:.1f}, k_{{bound}}={fixed_k_bound_plot}$")
            ax.legend(fontsize='small', loc='lower right'); ax.grid(True); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
            if not has_data: ax.text(0.5,0.5,"No data",ha='center',va='center',transform=ax.transAxes)
            fig.suptitle(f"Calibration Curves Under Correction (PDE: {args.pde_type})", fontsize=16) # Corrected Title
            plt.tight_layout() # Corrected: removed rect
            combined_plot_filename = f"calib_curves_PDE{args.pde_type}_s{args.theorem_s}_nu{args.theorem_nu}_single_config.png"
            combined_plot_full_path = os.path.join(args.results_dir_sweep_plots, combined_plot_filename)
            plt.savefig(combined_plot_full_path); print(f"\nSingle config plot saved to {combined_plot_full_path}"); plt.show()

    print("\nSweep complete.")

