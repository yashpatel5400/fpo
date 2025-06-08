import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import argparse
from itertools import product
import multiprocessing
import time
import json

def run_script(script_name, args_list, log_prefix=""):
    """Helper function to run a python script with arguments."""
    command = ["python", script_name] + [str(arg) for arg in args_list]
    max_print_len = 2500
    command_str = ' '.join(command)
    if len(command_str) > max_print_len:
        command_str = command_str[:max_print_len-3] + "..."
    
    print(f"{log_prefix}Executing: {command_str}")
    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10800) # 3-hour timeout
        if process.stderr:
            stderr_lower = process.stderr.lower()
            if "error" in stderr_lower or "traceback" in stderr_lower or "warning" in stderr_lower:
                print(f"{log_prefix}Stderr from {script_name} (may indicate issues):")
                print(process.stderr[:1000])
        return True
    except subprocess.CalledProcessError as e:
        print(f"{log_prefix}Error running {script_name}:")
        print(f"  Return code: {e.returncode}")
        print(f"  Stdout:\n{e.stdout}") 
        print(f"  Stderr:\n{e.stderr}") 
        return False
    except FileNotFoundError:
        print(f"{log_prefix}Error: Script {script_name} not found. Check paths.")
        return False
    except subprocess.TimeoutExpired as e:
        stdout_decoded = e.stdout.decode(errors='ignore') if e.stdout else 'Timeout: No stdout'
        stderr_decoded = e.stderr.decode(errors='ignore') if e.stderr else 'Timeout: No stderr'
        print(f"{log_prefix}Error: Script {script_name} timed out after {e.timeout} seconds.")
        print(f"  Stdout:\n{stdout_decoded}") 
        print(f"  Stderr:\n{stderr_decoded}") 
        return False
    except Exception as ex:
        print(f"{log_prefix}An unexpected error occurred while trying to run {script_name}: {ex}")
        return False

def run_single_calibration_pipeline(params_tuple):
    """
    Worker function for multiprocessing. Runs data gen, SNN train, and calibration.
    """
    k_snn_output_res, args_namespace_obj, filename_suffix_for_run_arg, \
    data_gen_script, snn_train_script, conformal_calib_script, \
    exp_idx, total_exps_for_current_config_set = params_tuple
    
    args = argparse.Namespace(**vars(args_namespace_obj)) 
    args.grf_alpha = args_namespace_obj.grf_alpha

    time.sleep(np.random.uniform(0, 0.1)) 
    log_prefix = f"[Worker {os.getpid()} Exp {exp_idx+1}/{total_exps_for_current_config_set} " \
                 f"GRF_A={args.grf_alpha:.1f}, K_SNN_Out={k_snn_output_res}] "
    print(f"{log_prefix}Processing...")

    dataset_filename = f"dataset_{args.pde_type}_Nin{args.n_grid_sim_input_ds}_Nout{k_snn_output_res}_{filename_suffix_for_run_arg}.npz"
    dataset_full_path = os.path.join(args.dataset_dir, dataset_filename)

    snn_model_filename = f"snn_PDE{args.pde_type}_Kin{args.n_grid_sim_input_ds}_Kout{k_snn_output_res}_H{args.snn_hidden_channels}_L{args.snn_num_hidden_layers}_{filename_suffix_for_run_arg}.pth"
    snn_model_full_path = os.path.join(args.model_dir, snn_model_filename)
    
    snn_training_plot_dir = os.path.join(args.results_dir_sweep_plots, f"snn_training_PDE{args.pde_type}_Kin{args.n_grid_sim_input_ds}_Kout{k_snn_output_res}_{filename_suffix_for_run_arg}")

    calib_results_subdir_name = (f"PDE{args.pde_type}_NinDS{args.n_grid_sim_input_ds}_SNNres{k_snn_output_res}_"
                                 f"KfullThm{args.n_grid_sim_input_ds}_s{args.theorem_s}_nu{args.theorem_nu}_{filename_suffix_for_run_arg}")
    calib_results_subdir = os.path.join(args.results_dir_calib_base, calib_results_subdir_name)
    
    coverage_data_filename_npz = os.path.join(
        calib_results_subdir, 
        f"coverage_data_PDE{args.pde_type}_thm_s{args.theorem_s}_nu{args.theorem_nu}_d{args.theorem_d}"
        f"_Nin{args.n_grid_sim_input_ds}_SNNout{k_snn_output_res}_NfullThm{args.n_grid_sim_input_ds}"
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
        "--snn_output_res", str(k_snn_output_res), 
        "--snn_model_dir", args.model_dir, 
        "--snn_hidden_channels", str(args.snn_hidden_channels), 
        "--snn_num_hidden_layers", str(args.snn_num_hidden_layers),
        "--snn_model_filename_override", snn_model_filename, 
        "--dataset_dir", args.dataset_dir, 
        "--results_dir", calib_results_subdir, 
        "--s_theorem", str(args.theorem_s), 
        "--nu_theorem", str(args.theorem_nu),
        "--d_dimensions", str(args.theorem_d), 
        "--elliptic_PDE_const_C_sq", str(args.elliptic_PDE_const_C_sq),
        "--fiber_potential_depth", str(args.fiber_potential_depth), 
        "--L_domain", str(args.L_domain), 
        "--evolution_time_T", str(args.evolution_time_T),
        "--solver_num_steps", str(args.solver_num_steps),
        "--no_plot" 
    ]
    if not run_script(conformal_calib_script, calib_args_list, log_prefix):
        print(f"{log_prefix}Conformal calibration FAILED for SNN_Output_Res={k_snn_output_res}. Returning None.")
        return None
    
    try:
        coverage_data = np.load(coverage_data_filename_npz)
        result_key = (k_snn_output_res, args.pde_type, args.grf_alpha, filename_suffix_for_run_arg)
        print(f"{log_prefix}Successfully processed results for key: {result_key}")
        return result_key, { 
            "nominal_coverages": coverage_data["nominal_coverages"],
            "empirical_coverages_theorem": coverage_data["empirical_coverages_theorem"],
            "empirical_coverages_no_correction": coverage_data.get("empirical_coverages_no_correction") 
        }
    except Exception as e:
        print(f"{log_prefix}Error loading coverage data for SNN_Output_Res={k_snn_output_res} from {coverage_data_filename_npz}: {e}")
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sweep for conformal calibration: SNN Output Res and GRF Alpha.")
    
    parser.add_argument('--pde_type', type=str, default="step_index_fiber", 
                        choices=["poisson", "step_index_fiber", "grin_fiber", "heat_equation"], 
                        help="Type of data generation process for the dataset.")
    
    parser.add_argument('--k_snn_output_res_values', nargs='+', type=int, default=[32, 48],
                        help='List of SNN output resolutions to sweep over.')
    parser.add_argument('--grf_alpha_values', nargs='+', type=float, default=[2.5, 4.0],
                        help="List of GRF alpha values to sweep over.")
    
    parser.add_argument('--n_grid_sim_input_ds', type=int, default=64,
                        help='Resolution for full input spectrum (Nin) in dataset generation AND N_full for theorem evaluation.')
    parser.add_argument('--num_samples_dataset', type=int, default=200) 
    parser.add_argument('--k_psi0_limit_dataset', type=int, default=12,
                        help="K_psi0_band_limit for GRF base initial state.")
    
    parser.add_argument('--snn_epochs', type=int, default=20) 
    parser.add_argument('--snn_hidden_channels', type=int, default=64)
    parser.add_argument('--snn_num_hidden_layers', type=int, default=3)

    parser.add_argument('--theorem_s', type=float, default=2.0)
    parser.add_argument('--theorem_nu', type=float, default=2.0) 
    parser.add_argument('--theorem_d', type=int, default=2)
    parser.add_argument('--elliptic_PDE_const_C_sq', type=float, default=4.0)

    parser.add_argument('--grf_tau', type=float, default=1.0)   
    parser.add_argument('--grf_offset_sigma', type=float, default=0.5)

    parser.add_argument('--L_domain', type=float, default=2*np.pi)
    parser.add_argument('--fiber_core_radius_factor', type=float, default=0.2)
    parser.add_argument('--fiber_potential_depth', type=float, default=1.0) 
    parser.add_argument('--grin_strength', type=float, default=0.01)
    parser.add_argument('--viscosity_nu', type=float, default=0.01)
    parser.add_argument('--evolution_time_T', type=float, default=0.1) 
    parser.add_argument('--solver_num_steps', type=int, default=50) 

    parser.add_argument('--dataset_dir', type=str, default="datasets_sweep_final_v3")
    parser.add_argument('--model_dir', type=str, default="trained_snn_models_sweep_final_v3")
    parser.add_argument('--results_dir_calib_base', type=str, default="results_conformal_validation_sweep_final_v3")
    parser.add_argument('--results_dir_sweep_plots', type=str, default="results_calibration_sweep_plots_final_v3")
    
    parser.add_argument('--skip_data_gen', action='store_true')
    parser.add_argument('--skip_snn_train', action='store_true')
    parser.add_argument('--skip_completed_calib_runs', action='store_true') 
    parser.add_argument('--num_processes', type=int, default=min(os.cpu_count(), 4))

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
    
    param_configurations_for_pool = []
    outer_sweep_product = list(product(args.grf_alpha_values, args.k_snn_output_res_values))
    total_experiments = len(outer_sweep_product)

    for i_exp, (current_grf_alpha_sweep_val, k_snn_io_res_current) in enumerate(outer_sweep_product):
        current_iter_args = argparse.Namespace(**vars(args))
        current_iter_args.grf_alpha = current_grf_alpha_sweep_val 

        current_filename_suffix = ""
        current_base_sub_script_args = ["--pde_type", current_iter_args.pde_type]
        
        if current_iter_args.pde_type == "poisson":
            current_filename_suffix = f"poisson_grfA{current_iter_args.grf_alpha:.1f}T{current_iter_args.grf_tau:.1f}OffS{current_iter_args.grf_offset_sigma:.1f}"
            current_base_sub_script_args.extend([
                "--grf_alpha", str(current_iter_args.grf_alpha), 
                "--grf_tau", str(current_iter_args.grf_tau), 
                "--grf_offset_sigma", str(current_iter_args.grf_offset_sigma)
            ])
        elif current_iter_args.pde_type == "step_index_fiber":
            current_filename_suffix = (f"fiber_GRFinA{current_iter_args.grf_alpha:.1f}T{current_iter_args.grf_tau:.1f}_"
                                       f"coreR{current_iter_args.fiber_core_radius_factor:.1f}_V{current_iter_args.fiber_potential_depth:.1f}_"
                                       f"evoT{current_iter_args.evolution_time_T:.1e}_steps{current_iter_args.solver_num_steps}")
            current_base_sub_script_args.extend([
                "--L_domain", str(current_iter_args.L_domain), 
                "--fiber_core_radius_factor", str(current_iter_args.fiber_core_radius_factor),
                "--fiber_potential_depth", str(current_iter_args.fiber_potential_depth), 
                "--evolution_time_T", str(current_iter_args.evolution_time_T),
                "--solver_num_steps", str(current_iter_args.solver_num_steps), 
                "--grf_alpha", str(current_iter_args.grf_alpha), 
                "--grf_tau", str(current_iter_args.grf_tau)
            ])
        elif current_iter_args.pde_type == "grin_fiber":
            current_filename_suffix = (f"grinfiber_GRFinA{current_iter_args.grf_alpha:.1f}T{current_iter_args.grf_tau:.1f}_"
                                       f"strength{current_iter_args.grin_strength:.2e}_"
                                       f"evoT{current_iter_args.evolution_time_T:.1e}_steps{current_iter_args.solver_num_steps}")
            current_base_sub_script_args.extend([
                "--L_domain", str(current_iter_args.L_domain), 
                "--grin_strength", str(current_iter_args.grin_strength),
                "--evolution_time_T", str(current_iter_args.evolution_time_T),
                "--solver_num_steps", str(current_iter_args.solver_num_steps), 
                "--grf_alpha", str(current_iter_args.grf_alpha), 
                "--grf_tau", str(current_iter_args.grf_tau)
            ])
        elif current_iter_args.pde_type == "heat_equation":
            current_filename_suffix = (f"heat_GRFinA{current_iter_args.grf_alpha:.1f}T{current_iter_args.grf_tau:.1f}_"
                                       f"nu{current_iter_args.viscosity_nu:.2e}_evoT{current_iter_args.evolution_time_T:.1e}")
            current_base_sub_script_args.extend([
                "--L_domain", str(current_iter_args.L_domain),
                "--viscosity_nu", str(current_iter_args.viscosity_nu),
                "--evolution_time_T", str(current_iter_args.evolution_time_T),
                "--grf_alpha", str(current_iter_args.grf_alpha), 
                "--grf_tau", str(current_iter_args.grf_tau)
            ])
        
        current_iter_args.filename_suffix_for_this_run = current_filename_suffix
        current_iter_args.base_sub_script_args_for_current_pde = current_base_sub_script_args
        
        param_configurations_for_pool.append((
            k_snn_io_res_current, 
            current_iter_args,    
            current_iter_args.filename_suffix_for_this_run, 
            data_gen_script_name, 
            snn_train_script_name, 
            conformal_calib_script_name,
            i_exp, 
            total_experiments 
        ))
            
    print(f"Total experiment configurations to run: {len(param_configurations_for_pool)}")
    results_list_from_pool = []
    if args.num_processes > 1 and len(param_configurations_for_pool) > 0:
        print(f"Using {args.num_processes} processes for parallel execution.")
        with multiprocessing.Pool(processes=args.num_processes) as pool:
            results_list_from_pool = pool.map(run_single_calibration_pipeline, param_configurations_for_pool)
    elif len(param_configurations_for_pool) > 0:
        print("Running experiments sequentially...")
        results_list_from_pool = [run_single_calibration_pipeline(params) for params in param_configurations_for_pool]

    for result_item in results_list_from_pool:
        if result_item: 
            key, data = result_item
            all_sweep_results_dict[key] = data 
    
    # --- Plotting Section (after ALL experiments are done) ---
    print("\n\n--- Aggregating and Plotting All Calibration Curves ---")
    
    if not all_sweep_results_dict:
        print("No results to plot.")
    else:
        swept_grf_alphas = sorted(list(set(args.grf_alpha_values)))
        num_alpha_plots = len(swept_grf_alphas)

        if num_alpha_plots == 0:
             print("No GRF alpha values to plot for.")
        else:
            ncols_fig = 2 if num_alpha_plots > 1 else 1 
            nrows_fig = int(np.ceil(num_alpha_plots / ncols_fig))
            
            fig, axes = plt.subplots(nrows_fig, ncols_fig, 
                                     figsize=(ncols_fig * 6.5, nrows_fig * 5.5), 
                                     sharex=True, sharey=True, squeeze=False)
            axes_flat = axes.flatten()
            plot_successful_overall = False

            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

            for i_plot, current_grf_alpha_plot_val in enumerate(swept_grf_alphas):
                ax = axes_flat[i_plot]
                ax.plot([0, 1], [0, 1], linestyle=':', color='gray', label='Ideal Coverage') 
                
                plot_iter_filename_suffix = ""
                if args.pde_type == "poisson":
                    plot_iter_filename_suffix = f"poisson_grfA{current_grf_alpha_plot_val:.1f}T{args.grf_tau:.1f}OffS{args.grf_offset_sigma:.1f}"
                elif args.pde_type == "step_index_fiber":
                     plot_iter_filename_suffix = (f"fiber_GRFinA{current_grf_alpha_plot_val:.1f}T{args.grf_tau:.1f}_"
                                               f"coreR{args.fiber_core_radius_factor:.1f}_V{args.fiber_potential_depth:.1f}_"
                                               f"evoT{args.evolution_time_T:.1e}_steps{args.solver_num_steps}")
                elif args.pde_type == "grin_fiber":
                     plot_iter_filename_suffix = (f"grinfiber_GRFinA{current_grf_alpha_plot_val:.1f}T{args.grf_tau:.1f}_"
                                               f"strength{args.grin_strength:.2e}_"
                                               f"evoT{args.evolution_time_T:.1e}_steps{args.solver_num_steps}")
                elif args.pde_type == "heat_equation":
                    plot_iter_filename_suffix = (f"heat_GRFinA{current_grf_alpha_plot_val:.1f}T{args.grf_tau:.1f}_"
                                       f"nu{args.viscosity_nu:.2e}_evoT{args.evolution_time_T:.1e}")

                has_data_for_this_subplot = False
                for i_snn, k_snn_val in enumerate(sorted(args.k_snn_output_res_values)):
                    result_key_lookup = (k_snn_val, args.pde_type, current_grf_alpha_plot_val, plot_iter_filename_suffix)
                    
                    current_color = color_cycle[i_snn % len(color_cycle)]

                    if result_key_lookup in all_sweep_results_dict:
                        data = all_sweep_results_dict[result_key_lookup]
                        ax.plot(data["nominal_coverages"], data["empirical_coverages_theorem"], 
                                marker='o', linestyle='-', markersize=4, color=current_color,
                                label=f'$N_{{max}}$={k_snn_val // 2} (Corrected)')
                        if data.get("empirical_coverages_no_correction") is not None:
                             ax.plot(data["nominal_coverages"], data["empirical_coverages_no_correction"], 
                                     marker='x', linestyle='--', markersize=4, color=current_color,
                                     label=f'$N_{{max}}$={k_snn_val // 2} (Uncorrected)')
                        has_data_for_this_subplot = True
                        plot_successful_overall = True
                
                ax.set_xlabel("Nominal Coverage ($1-\\alpha$)")
                if i_plot % ncols_fig == 0:
                    ax.set_ylabel("Empirical Coverage")
                ax.set_title(f"GRF $\\rho = {current_grf_alpha_plot_val:.1f}$")
                ax.legend(fontsize='xx-small', loc='lower right')
                ax.grid(True)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1.05)
                if not has_data_for_this_subplot:
                    ax.text(0.5,0.5,"No data",ha='center',va='center',transform=ax.transAxes)
            
            for j_ax_hide in range(num_alpha_plots, nrows_fig * ncols_fig): 
                if j_ax_hide < len(axes_flat):
                    fig.delaxes(axes_flat[j_ax_hide])
            
            if plot_successful_overall:
                pde_type_to_title = {
                    "poisson": "Poisson Equation", 
                    "step_index_fiber": "Step Index Fiber", 
                    "grin_fiber": "GRIN Fiber", 
                    "heat_equation": "Heat Equation",
                }

                fig.suptitle(f"Calibration Curves With Correction: {pde_type_to_title[args.pde_type]}", fontsize=16)
                plt.tight_layout() 
                
                combined_plot_filename = f"calib_curves_PDE{args.pde_type}_s{args.theorem_s}_nu{args.theorem_nu}_vs_alpha.png"
                combined_plot_full_path = os.path.join(args.results_dir_sweep_plots, combined_plot_filename)
                plt.savefig(combined_plot_full_path)
                print(f"\nCombined GRF Alpha calibration curve plot saved to {combined_plot_full_path}")
                plt.show()
            else:
                plt.close(fig) 
                print(f"  No data to plot for any GRF Alpha value in this figure.")
        
    print("\nSweep complete.")
