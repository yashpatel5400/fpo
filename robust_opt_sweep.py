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
        process = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5*10800) # 3-hour timeout
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

def run_single_robust_opt(params_tuple):
    """
    Worker function for multiprocessing. Runs a single instance of robust_optimization.py.
    """
    current_grf_alpha, current_m_val, current_snn_res, args, robust_opt_script, exp_idx, total_exps = params_tuple
    
    time.sleep(np.random.uniform(0, 0.2))
    log_prefix = f"[Worker {os.getpid()} Exp {exp_idx+1}/{total_exps} " \
                 f"GRF_A={current_grf_alpha:.1f}, M={current_m_val}, SNN_Res={current_snn_res}] "
    print(f"{log_prefix}Processing configuration...")

    # --- Construct necessary filenames and arguments ---
    filename_suffix = ""
    if args.pde_type == "poisson":
        filename_suffix = f"poisson_grfA{current_grf_alpha:.1f}T{args.grf_tau:.1f}OffS{args.grf_offset_sigma:.1f}"
    elif args.pde_type == "step_index_fiber":
        filename_suffix = (f"fiber_GRFinA{current_grf_alpha:.1f}T{args.grf_tau:.1f}_"
                           f"coreR{args.fiber_core_radius_factor:.1f}_V{args.fiber_potential_depth:.1f}_"
                           f"evoT{args.evolution_time_T:.1e}_steps{args.solver_num_steps}")
    elif args.pde_type == "grin_fiber":
        filename_suffix = (f"grinfiber_GRFinA{current_grf_alpha:.1f}T{args.grf_tau:.1f}_"
                           f"strength{args.grin_strength:.2e}_"
                           f"evoT{args.evolution_time_T:.1e}_steps{args.solver_num_steps}")

    # Define the expected output file path
    output_filename = f"sweep_run_alpha{current_grf_alpha}_M{current_m_val}_SNNres{current_snn_res}_results_M{current_m_val}_alphaCalib{args.alpha_for_radius:.2f}_{filename_suffix}.json"
    output_filepath = os.path.join(args.results_dir_rob_opt, output_filename)

    run_needed = True
    if args.skip_completed_runs and os.path.exists(output_filepath):
        print(f"{log_prefix}Skipping execution, result file found: {output_filepath}")
        run_needed = False
    
    if run_needed:
        robust_opt_args = [
            "--pde_type", args.pde_type,
            "--snn_model_dir", args.snn_model_dir,
            "--n_grid_sim_input_ds", args.n_grid_sim_input_ds,
            "--snn_output_res", current_snn_res, # This is a swept parameter
            "--snn_hidden_channels", args.snn_hidden_channels,
            "--snn_num_hidden_layers", args.snn_num_hidden_layers,
            "--grf_alpha", current_grf_alpha,
            "--grf_tau", args.grf_tau,
            "--L_domain", args.L_domain,
            "--fiber_core_radius_factor", args.fiber_core_radius_factor,
            "--fiber_potential_depth", args.fiber_potential_depth,
            "--grin_strength", args.grin_strength,
            "--evolution_time_T", args.evolution_time_T,
            "--solver_num_steps", args.solver_num_steps,
            "--num_distinct_states_M", current_m_val,
            "--alpha_for_radius", args.alpha_for_radius,
            "--calibration_results_base_dir", args.calibration_results_base_dir,
            "--theorem_s", args.theorem_s,
            "--theorem_nu", args.theorem_nu,
            "--theorem_d", args.theorem_d,
            "--num_trials_per_config", args.num_trials_per_config,
            "--results_dir", args.results_dir_rob_opt,
            "--output_json_filename_tag", f"sweep_run_alpha{current_grf_alpha}_M{current_m_val}_SNNres{current_snn_res}"
        ]
        
        success = run_script(robust_opt_script, robust_opt_args, log_prefix)
        
        if not success:
            print(f"{log_prefix}Robust optimization FAILED. Returning None.")
            return None
    
    try:
        with open(output_filepath, 'r') as f:
            result_data = json.load(f)
        result_key = (current_grf_alpha, current_m_val, current_snn_res)
        print(f"{log_prefix}Successfully processed results for key: {result_key}")
        return result_key, result_data
    except Exception as e:
        print(f"{log_prefix}Error loading or processing results from {output_filepath}: {e}")
        return None

def generate_latex_table(results_dict, args):
    """Generates a LaTeX table string from the results dictionary."""
    
    latex_string = "\\begin{table}[h!]\n"
    latex_string += "\\centering\n"
    latex_string += "\\caption{Summary of Robust Optimization Sweep Results for "
    latex_string += f"{args.pde_type.replace('_', ' ').title()}"
    latex_string += "}\n"
    latex_string += f"\\label{{tab:rob_opt_summary_{args.pde_type}}}\n"
    latex_string += "\\begin{tabular}{ccc S[table-format=1.4] S[table-format=1.4] S[table-format=1.4] S[table-format=1.3] S[table-format=1.3]}\n"
    latex_string += "\\toprule\n"
    latex_string += " {$\\rho$} & {M} & {$N_{max}$} & {PGM} & {Nominal} & {Robust} & {$p(R>P)$} & {$p(R>N)$} \\\\\n"
    latex_string += "\\midrule\n"

    sorted_keys = sorted(results_dict.keys())

    for key in sorted_keys:
        data = results_dict[key]
        grf_alpha, m_val, snn_res = key

        pgm_iab = data.get('avg_pgm_IAB', float('nan'))
        nom_iab = data.get('avg_nominal_IAB', float('nan'))
        rob_iab = data.get('avg_robust_IAB', float('nan'))
        p_val_rob_gt_nom = data.get('p_value_rob_gt_nom', 1.0)
        p_val_rob_gt_pgm = data.get('p_value_rob_gt_pgm', 1.0)
        
        pgm_str = f"{pgm_iab:.4f}"
        nom_str = f"{nom_iab:.4f}"
        rob_str = f"{rob_iab:.4f}"
        
        p_rob_pgm_str = f"{p_val_rob_gt_pgm:.3f}"
        p_rob_nom_str = f"{p_val_rob_gt_nom:.3f}"
        
        if rob_iab > nom_iab:
            if p_val_rob_gt_nom < 0.05:
                 rob_str = f"\\textbf{{{rob_str}}}$^*$"
            else:
                 rob_str = f"\\textbf{{{rob_str}}}"
        else:
            nom_str = f"\\textbf{{{nom_str}}}"
            
        latex_string += f" {grf_alpha:.1f} & {m_val} & {snn_res} & {pgm_str} & {nom_str} & {rob_str} & {p_rob_pgm_str} & {p_rob_nom_str} \\\\\n"

    latex_string += "\\bottomrule\n"
    latex_string += "\\end{tabular}\n"
    latex_string += "\\end{table}\n"
    
    return latex_string

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sweep script for Robust Optimization pipeline.")
    
    parser.add_argument('--pde_type', type=str, default="grin_fiber", 
                        choices=["poisson", "step_index_fiber", "grin_fiber"], 
                        help="Primary PDE type for this sweep execution.")
    
    parser.add_argument('--grf_alpha_values', nargs='+', type=float, default=[1.5, 1.8],
                        help="List of GRF alpha (smoothness) values to sweep over.")
    parser.add_argument('--num_distinct_states_M_values', nargs='+', type=int, default=[3, 4],
                        help="List of M values (number of states) to sweep over.")
    parser.add_argument('--snn_output_res_values', nargs='+', type=int, default=[32, 48],
                        help="List of SNN output resolutions to sweep over.")
    
    parser.add_argument('--snn_model_dir', type=str, default="trained_snn_models_sweep_final_v3",
                        help="Directory where pre-trained SNN models are stored.")
    parser.add_argument('--calibration_results_base_dir', type=str, default="results_conformal_validation_sweep_final_v3",
                        help="Base directory where calibration result subdirectories are stored.")
    
    parser.add_argument('--n_grid_sim_input_ds', type=int, default=64)
    parser.add_argument('--snn_hidden_channels', type=int, default=64)
    parser.add_argument('--snn_num_hidden_layers', type=int, default=3)
    parser.add_argument('--grf_tau', type=float, default=1.0)   
    parser.add_argument('--grf_offset_sigma', type=float, default=0.5)
    parser.add_argument('--L_domain', type=float, default=2*np.pi)
    parser.add_argument('--fiber_core_radius_factor', type=float, default=0.2)
    parser.add_argument('--fiber_potential_depth', type=float, default=1.0) 
    parser.add_argument('--grin_strength', type=float, default=0.01)
    parser.add_argument('--evolution_time_T', type=float, default=0.1) 
    parser.add_argument('--solver_num_steps', type=int, default=50) 
    parser.add_argument('--alpha_for_radius', type=float, default=0.1)
    parser.add_argument('--theorem_s', type=float, default=2.0)
    parser.add_argument('--theorem_nu', type=float, default=2.0)
    parser.add_argument('--theorem_d', type=int, default=2)
    parser.add_argument('--num_trials_per_config', type=int, default=50,
                        help="Number of trials per robust optimization run.")

    parser.add_argument('--base_results_dir', type=str, default="sweep_results_rob_opt",
                        help="Base directory to store all results from this sweep.")
    parser.add_argument('--skip_completed_runs', action='store_true', help="Skip runs where the output JSON file already exists.")
    parser.add_argument('--num_processes', type=int, default=min(os.cpu_count()-1 if os.cpu_count() and os.cpu_count() > 1 else 1, 8),
                        help="Number of parallel processes to use.")

    args = parser.parse_args()

    args.results_dir_rob_opt = os.path.join(args.base_results_dir, "rob_opt_runs")
    os.makedirs(args.results_dir_rob_opt, exist_ok=True)

    robust_opt_script_name = "robust_opt.py" 
    
    param_configurations_for_pool = []
    outer_sweep_product = list(product(args.grf_alpha_values, args.num_distinct_states_M_values, args.snn_output_res_values))
    total_experiments = len(outer_sweep_product)

    print(f"--- Starting Robust Optimization Sweep for PDE Type: {args.pde_type} ---")
    print(f"Sweeping over GRF Alphas: {args.grf_alpha_values}")
    print(f"Sweeping over M values: {args.num_distinct_states_M_values}")
    print(f"Sweeping over SNN Output Resolutions: {args.snn_output_res_values}")

    for i_exp, (current_grf_alpha, current_m_val, current_snn_res) in enumerate(outer_sweep_product):
        # We need to ensure that the robust_opt script is called with the correct snn_output_res for each run
        # We can pass it as part of the tuple to the worker function
        current_iter_args = argparse.Namespace(**vars(args))
        current_iter_args.snn_output_res = current_snn_res # Set current snn_res for this iteration

        param_configurations_for_pool.append((
            current_grf_alpha,
            current_m_val,
            current_snn_res,
            current_iter_args, 
            robust_opt_script_name,
            i_exp,
            total_experiments
        ))
            
    print(f"\nTotal robust optimization configurations to run: {len(param_configurations_for_pool)}")
    
    results_list_from_pool = []
    actual_num_processes = min(args.num_processes, len(param_configurations_for_pool))
    if actual_num_processes <= 0 and len(param_configurations_for_pool) > 0:
        actual_num_processes = 1

    if actual_num_processes > 1 and len(param_configurations_for_pool) > 0:
        print(f"Using {actual_num_processes} processes for parallel execution.")
        try:
            with multiprocessing.Pool(processes=actual_num_processes) as pool:
                results_list_from_pool = pool.map(run_single_robust_opt, param_configurations_for_pool)
        except Exception as e_pool:
            print(f"Error during multiprocessing pool execution: {e_pool}")
    elif len(param_configurations_for_pool) > 0:
        print("Running experiments sequentially...")
        for config_tuple in param_configurations_for_pool:
            results_list_from_pool.append(run_single_robust_opt(config_tuple))
    else:
        print("No experiment configurations to run.")

    all_sweep_results_dict = {}
    for result_item in results_list_from_pool:
        if result_item:
            key, data = result_item
            all_sweep_results_dict[key] = data 
    
    # --- Print Summary of Results ---
    print("\n\n--- Robust Optimization Sweep Summary ---")
    print(f"PDE Type: {args.pde_type}")
    print("-" * 115)
    print(f"{'GRF Alpha':<12} {'M':<4} {'N_max':<6} {'Avg I_AB (PGM)':<18} {'Avg I_AB (Nominal)':<22} {'Avg I_AB (Robust)':<22} {'p(R>P)':<10} {'p(R>N)':<10}")
    print("-" * 115)

    sorted_keys = sorted(all_sweep_results_dict.keys())

    if not sorted_keys:
        print("No successful results to display.")
    else:
        for key in sorted_keys:
            data = all_sweep_results_dict[key]
            grf_alpha, m_val, snn_res = key
            
            pgm_iab = data.get('avg_pgm_IAB', float('nan'))
            nom_iab = data.get('avg_nominal_IAB', float('nan'))
            rob_iab = data.get('avg_robust_IAB', float('nan'))
            p_val_rob_gt_pgm = data.get('p_value_rob_gt_pgm', 1.0)
            p_val_rob_gt_nom = data.get('p_value_rob_gt_nom', 1.0)

            print(f"{grf_alpha:<12.2f} {m_val:<4} {snn_res:<6} {pgm_iab:<18.4f} {nom_iab:<22.4f} {rob_iab:<22.4f} {p_val_rob_gt_pgm:<10.3f} {p_val_rob_gt_nom:<10.3f}")
    
    print("-" * 115)
    
    # --- Generate and Save LaTeX Table ---
    if all_sweep_results_dict:
        latex_content = generate_latex_table(all_sweep_results_dict, args)
        latex_filename = f"summary_table_rob_opt_PDE_{args.pde_type}.tex"
        latex_filepath = os.path.join(args.base_results_dir, latex_filename)
        try:
            with open(latex_filepath, 'w') as f:
                f.write(latex_content)
            print(f"\nLaTeX summary table saved to: {latex_filepath}")
            print("Note: To compile, your LaTeX document may need '\\usepackage{booktabs}' and '\\usepackage{siunitx}'.")
        except Exception as e:
            print(f"\nError saving LaTeX table: {e}")

    print("\nSweep complete.")
