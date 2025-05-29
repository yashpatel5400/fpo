import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess # To call other scripts
import argparse
from itertools import product

def run_script(script_name, args_list):
    """Helper function to run a python script with arguments."""
    command = ["python", script_name] + args_list
    print(f"\nExecuting: {' '.join(command)}")
    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True, timeout=3600) # Increased timeout
        if process.stderr:
            if "error" in process.stderr.lower() or "traceback" in process.stderr.lower():
                print(f"Stderr from {script_name} (potential error):")
                print(process.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print(f"Return code: {e.returncode}\nStdout: {e.stdout}\nStderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"Error: Script {script_name} not found.")
        return False
    except subprocess.TimeoutExpired as e:
        stdout_decoded = e.stdout.decode(errors='ignore') if e.stdout else 'None'
        stderr_decoded = e.stderr.decode(errors='ignore') if e.stderr else 'None'
        print(f"Error: Script {script_name} timed out after {e.timeout} seconds.")
        print(f"  Stdout: {stdout_decoded[:500]}...")
        print(f"  Stderr: {stderr_decoded[:500]}...")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a sweep over K_TRUNC_SNN and K_TRUNC_BOUND for conformal calibration, supporting different PDE types.")
    # PDE Type
    parser.add_argument('--pde_type', type=str, default="phenomenological_channel", 
                        choices=["phenomenological_channel", "poisson"], 
                        help="Type of data generation process for the dataset.")
    
    # Sweep parameters
    parser.add_argument('--k_trunc_snn_values', nargs='+', type=int, default=[64, 80, 96], 
                        help='List of K_TRUNC_SNN (N_max) values to sweep over.')
    parser.add_argument('--k_trunc_bound_values', nargs='+', type=int, default=[16, 32, 48, 64],
                        help='List of k_trunc_bound values (K_grid_size for get_mode_indices_and_weights) to sweep over.')
    
    # Fixed parameters for dataset and SNN structure
    parser.add_argument('--k_trunc_full_fixed', type=int, default=96, 
                        help='Fixed K_TRUNC_FULL_EVAL (N_full) for all runs.')
    parser.add_argument('--num_samples_dataset', type=int, default=1000)
    parser.add_argument('--n_grid_sim_dataset', type=int, default=96) # Should be >= k_trunc_full
    parser.add_argument('--k_psi0_limit_dataset', type=int, default=12, help="K_psi0_band_limit for dataset (phenom. channel) or GRF base (Poisson).")
    
    parser.add_argument('--snn_epochs', type=int, default=30)
    parser.add_argument('--snn_hidden_channels', type=int, default=64)
    parser.add_argument('--snn_num_hidden_layers', type=int, default=3)

    # Fixed theorem parameters
    parser.add_argument('--theorem_s', type=float, default=2.0)
    parser.add_argument('--theorem_nu', type=float, default=2.0) 
    parser.add_argument('--theorem_d', type=int, default=2)

    # Phenomenological Channel Noise Parameters
    parser.add_argument('--apply_attenuation', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--attenuation_loss_factor', type=float, default=0.2)
    parser.add_argument('--apply_additive_sobolev_noise', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--sobolev_noise_level_base', type=float, default=0.01)
    parser.add_argument('--sobolev_order_s', type=float, default=1.0)
    parser.add_argument('--apply_phase_noise', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--phase_noise_std_rad', type=float, default=0.05)

    # Poisson Source (GRF parameters)
    parser.add_argument('--grf_alpha', type=float, default=2.5, help="GRF alpha for Poisson source.")
    parser.add_argument('--grf_tau', type=float, default=7.0, help="GRF tau for Poisson source.")
    parser.add_argument('--grf_offset_sigma', type=float, default=0.5, help="Sigma for hierarchical offset in Poisson source.")


    # Directories and control flags
    parser.add_argument('--dataset_dir', type=str, default="datasets_sweep")
    parser.add_argument('--model_dir', type=str, default="trained_snn_models_sweep")
    parser.add_argument('--results_dir_calib_base', type=str, default="results_conformal_theorem_validation_sweep")
    parser.add_argument('--results_dir_sweep_plots', type=str, default="results_calibration_sweep_plots")
    parser.add_argument('--skip_data_gen', action='store_true', help="Skip data generation if datasets already exist.")
    parser.add_argument('--skip_snn_train', action='store_true', help="Skip SNN training if models already exist.")

    args = parser.parse_args()

    if args.n_grid_sim_dataset < args.k_trunc_full_fixed:
        print(f"Warning: n_grid_sim_dataset ({args.n_grid_sim_dataset}) is less than k_trunc_full_fixed ({args.k_trunc_full_fixed}). Adjusting n_grid_sim_dataset.")
        args.n_grid_sim_dataset = args.k_trunc_full_fixed

    os.makedirs(args.dataset_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir_calib_base, exist_ok=True)
    os.makedirs(args.results_dir_sweep_plots, exist_ok=True)

    all_sweep_results = {} 

    data_gen_script = "data.py" 
    snn_train_script = "model.py"    
    conformal_calib_script = "calibration.py" 

    # Construct filename_suffix based on pde_type and relevant params for this sweep run
    filename_suffix_for_run = ""
    common_data_gen_args_for_run = [
        "--num_samples", str(args.num_samples_dataset),
        "--n_grid_sim", str(args.n_grid_sim_dataset), 
        "--k_psi0_limit", str(args.k_psi0_limit_dataset),
        # k_trunc_snn, k_trunc_full, output_dir will be added per k_snn
    ]
    common_snn_train_args_for_run = [
        "--snn_hidden_channels", str(args.snn_hidden_channels),
        "--snn_num_hidden_layers", str(args.snn_num_hidden_layers),
        "--epochs", str(args.snn_epochs),
    ]
    common_calib_args_for_run = [
        "--snn_model_dir", args.model_dir, 
        "--snn_hidden_channels", str(args.snn_hidden_channels), 
        "--snn_num_hidden_layers", str(args.snn_num_hidden_layers),
        "--dataset_dir", args.dataset_dir, 
        "--s_theorem", str(args.theorem_s), "--nu_theorem", str(args.theorem_nu),
        "--d_dimensions", str(args.theorem_d), "--no_plot"
    ]

    if args.pde_type == "phenomenological_channel":
        noise_parts = []
        if args.apply_attenuation: noise_parts.append(f"att{args.attenuation_loss_factor:.2f}")
        if args.apply_additive_sobolev_noise: noise_parts.append(f"sob{args.sobolev_noise_level_base:.3f}s{args.sobolev_order_s:.1f}")
        if args.apply_phase_noise: noise_parts.append(f"ph{args.phase_noise_std_rad:.2f}")
        filename_suffix_for_run = "_".join(noise_parts) if noise_parts else "no_noise"
        
        # Add specific noise params to common args for sub-scripts
        common_data_gen_args_for_run.extend([
            "--pde_type", "phenomenological_channel",
            *(["--apply_attenuation"] if args.apply_attenuation else ["--no-apply_attenuation"]),
            "--attenuation_loss_factor", str(args.attenuation_loss_factor),
            *(["--apply_additive_sobolev_noise"] if args.apply_additive_sobolev_noise else ["--no-apply_additive_sobolev_noise"]),
            "--sobolev_noise_level_base", str(args.sobolev_noise_level_base),
            "--sobolev_order_s", str(args.sobolev_order_s),
            *(["--apply_phase_noise"] if args.apply_phase_noise else ["--no-apply_phase_noise"]),
            "--phase_noise_std_rad", str(args.phase_noise_std_rad)
        ])
        common_snn_train_args_for_run.extend(common_data_gen_args_for_run[-16:]) # Last 13 are PDE type + noise params
        common_calib_args_for_run.extend(common_data_gen_args_for_run[-16:])

    elif args.pde_type == "poisson":
        filename_suffix_for_run = f"poisson_grfA{args.grf_alpha:.1f}T{args.grf_tau:.1f}"
        common_data_gen_args_for_run.extend([
            "--pde_type", "poisson",
            "--grf_alpha", str(args.grf_alpha),
            "--grf_tau", str(args.grf_tau),
            "--grf_offset_sigma", str(args.grf_offset_sigma)
        ])
        common_snn_train_args_for_run.extend(common_data_gen_args_for_run[-8:]) # Last 5 are PDE type + GRF params
        common_calib_args_for_run.extend(common_data_gen_args_for_run[-8:])
    
    print(f"Using fixed PDE type '{args.pde_type}' with suffix '{filename_suffix_for_run}' for this sweep.")

    param_combinations = list(product(args.k_trunc_snn_values, args.k_trunc_bound_values))

    for k_snn, k_bound in param_combinations:
        print(f"\n\n--- Processing K_TRUNC_SNN (N_max) = {k_snn}, K_TRUNC_BOUND = {k_bound} ---")
        
        dataset_filename = f"dataset_{args.pde_type}_Nmax{k_snn}_Nfull{args.k_trunc_full_fixed}_{filename_suffix_for_run}.npz"
        dataset_full_path = os.path.join(args.dataset_dir, dataset_filename)

        if not args.skip_data_gen or not os.path.exists(dataset_full_path):
            print(f"Generating dataset: {dataset_filename}...")
            current_data_gen_args = common_data_gen_args_for_run + [
                "--k_trunc_snn", str(k_snn),
                "--k_trunc_full", str(args.k_trunc_full_fixed),
                "--output_dir", args.dataset_dir
            ]
            if not run_script(data_gen_script, current_data_gen_args):
                print(f"Data generation failed for {dataset_filename}. Skipping this K_SNN.")
                continue
        else:
            print(f"Skipping data generation, dataset found: {dataset_full_path}")

        snn_model_filename = f"snn_PDE{args.pde_type}_K{k_snn}_H{args.snn_hidden_channels}_L{args.snn_num_hidden_layers}_{filename_suffix_for_run}.pth"
        snn_model_full_path = os.path.join(args.model_dir, snn_model_filename)
        snn_training_plot_dir = os.path.join(args.results_dir_sweep_plots, f"snn_training_PDE{args.pde_type}_K{k_snn}_{filename_suffix_for_run}")

        if not args.skip_snn_train or not os.path.exists(snn_model_full_path):
            print(f"Training SNN: {snn_model_filename}...")
            os.makedirs(snn_training_plot_dir, exist_ok=True)
            current_train_args = common_snn_train_args_for_run + [
                "--k_trunc_snn", str(k_snn), "--k_trunc_full", str(args.k_trunc_full_fixed), 
                "--dataset_dir", args.dataset_dir, "--model_save_dir", args.model_dir,
                "--plot_save_dir", snn_training_plot_dir,
                # SNN arch params are already in common_snn_train_args_for_run (if added there)
                # or taken from main args if not part of common.
                # The snn_trainer script should also parse these if they are not fixed.
            ]
            print(current_train_args)
            if not run_script(snn_train_script, current_train_args):
                print(f"SNN training failed for {snn_model_filename}. Skipping this K_SNN.")
                continue
        else:
            print(f"Skipping SNN training, model found: {snn_model_full_path}")
            
        calib_results_subdir_name = (f"PDE{args.pde_type}_Ksnn{k_snn}_Kfull{args.k_trunc_full_fixed}_"
                                     f"Kbound{k_bound}_s{args.theorem_s}_nu{args.theorem_nu}_{filename_suffix_for_run}")
        calib_results_subdir = os.path.join(args.results_dir_calib_base, calib_results_subdir_name)
        os.makedirs(calib_results_subdir, exist_ok=True)

        current_calib_args = common_calib_args_for_run + [
            "--k_trunc_snn", str(k_snn), "--k_trunc_full", str(args.k_trunc_full_fixed),
            "--results_dir", calib_results_subdir, 
            "--k_trunc_bound", str(k_bound),
            "--snn_model_filename_override", snn_model_filename # Pass exact model name
        ]
        if not run_script(conformal_calib_script, current_calib_args):
            print(f"Conformal calibration failed for K_SNN={k_snn}, K_BOUND={k_bound}. Skipping.")
            continue
        
        coverage_data_filename = os.path.join(
            calib_results_subdir, 
            f"coverage_data_thm_s{args.theorem_s}_nu{args.theorem_nu}_d{args.theorem_d}_Nmax{k_snn}_Nfull{args.k_trunc_full_fixed}_Kbound{k_bound}_{filename_suffix_for_run}.npz"
        )
        try:
            coverage_data = np.load(coverage_data_filename)
            all_sweep_results[(k_snn, k_bound, args.pde_type, filename_suffix_for_run)] = { 
                "nominal_coverages": coverage_data["nominal_coverages"],
                "empirical_coverages_theorem": coverage_data["empirical_coverages_theorem"]
            }
            print(f"Successfully processed results for K_SNN={k_snn}, K_BOUND={k_bound}, PDE={args.pde_type}")
        except Exception as e:
            print(f"Error loading coverage data for K_SNN={k_snn}, K_BOUND={k_bound} from {coverage_data_filename}: {e}")

    print("\n\n--- Aggregating and Plotting Side-by-Side Calibration Curves ---")
    
    unique_k_trunc_bounds = sorted(list(set(args.k_trunc_bound_values)))
    num_bounds = len(unique_k_trunc_bounds)

    if num_bounds == 0 or not all_sweep_results:
        print("No results to plot. Exiting plotting.")
    else:
        ncols = int(np.ceil(np.sqrt(num_bounds)))
        nrows = int(np.ceil(num_bounds / ncols))
        if nrows == 0: nrows = 1; 
        if ncols == 0: ncols = 1;
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5), squeeze=False)
        axes_flat = axes.flatten()

        for i, k_bound_val in enumerate(unique_k_trunc_bounds):
            ax = axes_flat[i]
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal Coverage (y=x)')
            has_data_for_this_k_bound = False
            for k_snn_val in sorted(args.k_trunc_snn_values):
                # Construct the key used for storing results
                result_key = (k_snn_val, k_bound_val, args.pde_type, filename_suffix_for_run)
                if result_key in all_sweep_results:
                    data = all_sweep_results[result_key]
                    ax.plot(data["nominal_coverages"], data["empirical_coverages_theorem"], 
                            marker='o', linestyle='-', markersize=4,
                            label=f'$N_{{max}}$={k_snn_val}')
                    has_data_for_this_k_bound = True
            
            ax.set_xlabel("Nominal Coverage ($1-\\alpha$)")
            ax.set_ylabel("Empirical Coverage")
            ax.set_title(f"$N_{{ \\mathrm{{eff}} }} = {k_bound_val}$")
            ax.legend(fontsize='small'); ax.grid(True); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
            if not has_data_for_this_k_bound:
                ax.text(0.5,0.5,"No data",ha='center',va='center',transform=ax.transAxes)

        for j in range(num_bounds, nrows * ncols): fig.delaxes(axes_flat[j])

        fig.suptitle(f"Calibration Curves Under Correction by $N_{{ \mathrm{{eff}} }}$ (Poisson)", fontsize=14) # Corrected LaTeX
        # fig.suptitle(f"Calibration Curves Under Correction by $N_{{ \mathrm{{eff}} }}$ ($\\mu_{{\\eta}}={args.sobolev_noise_level_base:.3f}$, $\\sigma_{{\\theta}}={args.phase_noise_std_rad:.2f}$)", fontsize=14) # Corrected LaTeX
        plt.tight_layout() 
        
        combined_plot_path = os.path.join(args.results_dir_sweep_plots, 
                                          f"side_by_side_calib_PDE{args.pde_type}_s{args.theorem_s}_nu{args.theorem_nu}_{filename_suffix_for_run}.png")
        plt.savefig(combined_plot_path)
        print(f"\nCombined side-by-side calibration curve plot saved to {combined_plot_path}")
        plt.show()

    print("\nSweep complete.")

