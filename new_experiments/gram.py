import numpy as np
import torch
import torch.optim as optim
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
from scipy import stats # For t-test
from itertools import product 
import multiprocessing # Added for parallel execution
import json # For saving individual experiment results
from scipy.linalg import dft # For DFT matrix

# --- Assume constants and basic functions are available if run externally ---
try:
    from constants import GLOBAL_HBAR, GLOBAL_M
except ImportError:
    GLOBAL_HBAR = 1.0; GLOBAL_M = 1.0 # Dummy values

# --- Core Functions for I_AB Calculation and Optimization ---

def calculate_conditional_probs_p_j_given_k_torch(phi_full_torch, x_s_torch, M_val):
    """ Calculates p(j|k; phi) matrix using PyTorch. P[k,j] = p(j|k). """
    p_j_k_matrix = torch.zeros((M_val, M_val), dtype=torch.float64, device=phi_full_torch.device)
    s_indices = torch.arange(M_val, device=phi_full_torch.device, dtype=torch.float64)
    y_s_real = x_s_torch * torch.cos(phi_full_torch)
    y_s_imag = x_s_torch * (-torch.sin(phi_full_torch))
    for k_sent_idx in range(M_val):
        for j_outcome_idx in range(M_val):
            dft_phase_term = 2 * np.pi * s_indices * (k_sent_idx - j_outcome_idx) / M_val
            cos_dft, sin_dft = torch.cos(dft_phase_term), torch.sin(dft_phase_term)
            sum_real = torch.sum(y_s_real * cos_dft - y_s_imag * sin_dft)
            sum_imag = torch.sum(y_s_real * sin_dft + y_s_imag * cos_dft)
            p_j_k_matrix[k_sent_idx, j_outcome_idx] = (1/M_val**2) * (sum_real**2 + sum_imag**2)
    row_sums = torch.sum(p_j_k_matrix, dim=1, keepdim=True)
    p_j_k_matrix = torch.where(row_sums > 1e-9, p_j_k_matrix / (row_sums + 1e-15), torch.ones_like(p_j_k_matrix) / M_val)
    return p_j_k_matrix

def calculate_I_AB_components_torch(phi_params_torch, M_val, x_s_torch, q_priors_torch, eps=1e-12):
    """ Calculates H_B, H_cond (H(B|A)), and I_AB = H_B - H_cond using PyTorch. """
    if M_val == 1: return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
    device = x_s_torch.device 
    if phi_params_torch.numel() > 0 : device = phi_params_torch.device
    if phi_params_torch.ndim == 0 and M_val > 1 and phi_params_torch.numel() == 0: 
         phi_full_torch = torch.zeros(M_val, dtype=torch.float64, device=device)
    elif M_val == 1: 
        phi_full_torch = torch.tensor([0.0], dtype=torch.float64, device=device)
    else: 
        phi_full_torch = torch.cat((torch.tensor([0.0], dtype=torch.float64, device=device), phi_params_torch))
    p_j_given_k_matrix = calculate_conditional_probs_p_j_given_k_torch(phi_full_torch, x_s_torch, M_val)
    p_j_given_0 = p_j_given_k_matrix[0, :]
    H_cond = -torch.sum(torch.where(p_j_given_0 > eps, p_j_given_0 * torch.log2(p_j_given_0), torch.tensor(0.0, device=device)))
    P_B_j_list = [torch.sum(q_priors_torch * p_j_given_k_matrix[:, j_idx]) for j_idx in range(M_val)]
    P_B = torch.stack(P_B_j_list)
    P_B_sum = torch.sum(P_B)
    P_B = torch.where(P_B_sum > 1e-9, P_B / (P_B_sum + eps), torch.ones_like(P_B) / M_val) 
    H_B = -torch.sum(torch.where(P_B > eps, P_B * torch.log2(P_B), torch.tensor(0.0, device=device)))
    I_AB = H_B - H_cond
    return I_AB, H_B, H_cond

def calculate_I_AB_numpy(phi_values_np, M_val, x_s_np, q_priors_np, eps=1e-12):
    x_torch = torch.tensor(x_s_np, dtype=torch.float64); q_torch = torch.tensor(q_priors_np, dtype=torch.float64)
    if M_val == 1: phi_params_torch = torch.tensor([], dtype=torch.float64)
    else:
        phi_params_np_slice = phi_values_np[1:] 
        if not isinstance(phi_params_np_slice, np.ndarray): phi_params_np_slice = np.array(phi_params_np_slice)
        if phi_params_np_slice.ndim == 0:
            phi_params_torch = torch.tensor([phi_params_np_slice.item()], dtype=torch.float64) if phi_params_np_slice.size >0 else torch.tensor([], dtype=torch.float64)
        else: phi_params_torch = torch.tensor(phi_params_np_slice, dtype=torch.float64)
    I_AB, _, _ = calculate_I_AB_components_torch(phi_params_torch, M_val, x_torch, q_torch, eps)
    return I_AB.item()

def inner_adversarial_objective_scipy(x_tilde_np, M_val, phi_values_rad_np, q_priors_np):
    return calculate_I_AB_numpy(phi_values_rad_np, M_val, x_tilde_np, q_priors_np)

def solve_inner_adversarial_problem_scipy(phi_values_rad_np, M_val, x_center_of_uncertainty_np, L2_ball_radius, q_priors_np):
    def l2_constraint(x): return L2_ball_radius**2 - np.sum((x - x_center_of_uncertainty_np)**2)
    constraints = [{'type': 'ineq', 'fun': l2_constraint}]; bounds = [(0.001, None) for _ in range(M_val)] 
    x0_guess = np.maximum(0.001, x_center_of_uncertainty_np.copy())
    diff_from_center = x0_guess - x_center_of_uncertainty_np; norm_diff = np.linalg.norm(diff_from_center)
    if norm_diff > L2_ball_radius and L2_ball_radius > 1e-9 : 
        x0_guess = x_center_of_uncertainty_np + diff_from_center * (L2_ball_radius / norm_diff)
        x0_guess = np.maximum(0.001, x0_guess) 
    res = minimize(inner_adversarial_objective_scipy, x0_guess, args=(M_val, phi_values_rad_np, q_priors_np),
                   method='SLSQP', bounds=bounds, constraints=constraints, options={'disp': False, 'ftol': 1e-7, 'maxiter':200})
    if res.success: return res.fun, res.x
    return inner_adversarial_objective_scipy(x_center_of_uncertainty_np, M_val, phi_values_rad_np, q_priors_np), x_center_of_uncertainty_np

def optimize_phases_pytorch(M_val, x_for_opt_torch, q_priors_torch, is_robust=False, x_center_for_robust_np=None, 
                            L2_ball_radius_for_robust=None, num_epochs=300, lr=0.01, initial_phases_np=None, trial_info=""): 
    num_phases_to_opt = M_val - 1
    if M_val == 1: 
        I_AB_val, _, _ = calculate_I_AB_components_torch(torch.tensor([],dtype=torch.float64,device=x_for_opt_torch.device),M_val,x_for_opt_torch,q_priors_torch)
        return np.array([0.0]), [I_AB_val.item()] * num_epochs 
    if initial_phases_np is None or len(initial_phases_np) != num_phases_to_opt : 
        initial_phases_np = (np.random.rand(num_phases_to_opt) - 0.5) * 0.1 if num_phases_to_opt > 0 else np.array([])
    phi_params = torch.tensor(initial_phases_np, dtype=torch.float64, requires_grad=True)
    optimizer = optim.Adam([phi_params], lr=lr) if num_phases_to_opt > 0 else None
    history_I_AB_during_opt = []
    for epoch in range(num_epochs):
        if optimizer: optimizer.zero_grad()
        current_phi_params_for_calc = phi_params if num_phases_to_opt > 0 else torch.tensor([],dtype=torch.float64,device=x_for_opt_torch.device)
        loss = torch.tensor(0.0, dtype=torch.float64, device=current_phi_params_for_calc.device)
        I_AB_epoch = 0.0
        if not is_robust:
            I_AB_val, _, _ = calculate_I_AB_components_torch(current_phi_params_for_calc, M_val, x_for_opt_torch, q_priors_torch)
            loss = -I_AB_val; I_AB_epoch = I_AB_val.item()
        else:
            phi_full_torch_current = torch.cat((torch.tensor([0.0],dtype=torch.float64,device=current_phi_params_for_calc.device), current_phi_params_for_calc))
            phi_np_for_scipy = phi_full_torch_current.detach().numpy()
            min_I_AB_inner, x_tilde_star_np = solve_inner_adversarial_problem_scipy(phi_np_for_scipy, M_val, x_center_for_robust_np, L2_ball_radius_for_robust, q_priors_torch.numpy())
            x_tilde_star_torch = torch.tensor(x_tilde_star_np, dtype=torch.float64, device=current_phi_params_for_calc.device)
            I_AB_val_at_x_tilde_star, _, _ = calculate_I_AB_components_torch(current_phi_params_for_calc, M_val, x_tilde_star_torch, q_priors_torch)
            loss = -I_AB_val_at_x_tilde_star; I_AB_epoch = I_AB_val_at_x_tilde_star.item() 
        if num_phases_to_opt > 0 and loss.requires_grad and optimizer: loss.backward(); optimizer.step()
        history_I_AB_during_opt.append(I_AB_epoch)
    final_phi_list_params = phi_params.detach().numpy().tolist() if num_phases_to_opt > 0 else []
    return np.array([0.0] + final_phi_list_params), history_I_AB_during_opt

def generate_GUS_gram_matrix_and_sqrt_eigenvalues(M_val, structured_pattern=None):
    """
    Generates a GUS-compliant Gram matrix G_true and its sqrt_eigenvalues.
    G_true is circulant, PSD, with 1s on the diagonal.
    """
    if M_val <= 0: return None, np.array([])
    
    # 1. Generate target eigenvalues g_j for G_true such that sum(g_j) = M_val
    target_g_j_values = np.zeros(M_val)
    if M_val == 1: target_g_j_values = np.array([1.0])
    elif structured_pattern == 'dominant_first' and M_val >= 2:
        target_g_j_values[0] = 0.7 * M_val 
        remaining_sum = M_val - target_g_j_values[0]
        if M_val > 1 and remaining_sum > 1e-7 : 
            target_g_j_values[1:] = np.random.dirichlet(np.ones(M_val - 1)) * remaining_sum
        elif M_val > 1: 
            target_g_j_values[1:] = 1e-6 
            if np.sum(target_g_j_values) > 1e-9: target_g_j_values = M_val * target_g_j_values / np.sum(target_g_j_values)
            else: target_g_j_values = np.full(M_val, 1.0)
    elif structured_pattern == 'uniform_like' and M_val > 0:
        target_g_j_values = np.full(M_val, 1.0) 
    else: 
        target_g_j_values = np.random.dirichlet(np.ones(M_val) * 1.0) * M_val 
    target_g_j_values = np.maximum(1e-6, target_g_j_values) # Ensure strictly positive
    target_g_j_values = np.sort(target_g_j_values)[::-1] # Sort for consistency (optional)

    # 2. Construct G_true = U diag(target_g_j) U_dagger
    # U is the DFT matrix (unitary, U_dagger = U_inv)
    # scipy.linalg.dft(M) returns the DFT matrix. U = dft(M, norm='ortho') or (1/sqrt(M))*dft(M)
    # Let's use the definition that U_dagger diagonalizes circulant matrices.
    # If G is circulant, G = U D_G U_dagger.
    # First row of G: c = (c0, c1, ..., c_{M-1}) where c0=1. Eigenvalues g_s = sum_k c_k exp(-2pi*i*k*s/M)
    # Inverse: First row c_k = (1/M) sum_s g_s exp(2pi*i*k*s/M)
    
    first_row_G_true = np.zeros(M_val, dtype=np.complex128)
    for k_idx in range(M_val):
        for s_idx in range(M_val):
            first_row_G_true[k_idx] += target_g_j_values[s_idx] * np.exp(2j * np.pi * k_idx * s_idx / M_val)
    first_row_G_true /= M_val
    
    # Construct circulant G_true from its first row
    G_true = np.array([np.roll(first_row_G_true, i) for i in range(M_val)])
    # Ensure G_true is Hermitian (it should be if target_g_j are real)
    G_true = (G_true + G_true.conj().T) / 2.0 
    # Ensure diagonal is 1 (should be if sum(target_g_j)=M)
    # Forcing it might break PSD if eigenvalues were not summing to M correctly.
    # The construction above should ensure G_kk = (1/M)sum(g_j). If sum(g_j)=M, then G_kk=1.

    # Verify eigenvalues of constructed G_true (should be close to target_g_j_values)
    # computed_eigvals = np.sort(np.linalg.eigvalsh(G_true))[::-1]
    # if not np.allclose(computed_eigvals, target_g_j_values):
    #     print(f"Warning: Eigenvalues of constructed G_true {computed_eigvals} differ from target {target_g_j_values}")

    return G_true, np.sqrt(target_g_j_values) # Return G_true and sqrt of its target eigenvalues

def run_single_experiment_config(config_tuple_with_id):
    config, experiment_idx, total_experiments, process_id = config_tuple_with_id
    print(f"[P{process_id}] Exp {experiment_idx+1}/{total_experiments}: "
          f"EstBiasFactor={config.get('estimation_bias_factor',0):.2f}, EstNoiseStd={config['estimation_noise_std']:.3f}, L2Rad={config['L2_uncertainty_ball_radius']:.3f}, M={config['M_states']}")
    
    num_trials_this_experiment = config['num_trials_per_experiment']
    pgm_IAB_on_true_trials, nominal_IAB_on_true_trials, robust_IAB_on_true_trials = [], [], []

    for i_trial in range(num_trials_this_experiment):
        # 1. Generate G_true and its sqrt_eigenvalues
        _, sqrt_g_j_true_np = generate_GUS_gram_matrix_and_sqrt_eigenvalues(
            config['M_states'], structured_pattern=config['true_g_pattern']
        )
        x_true_torch = torch.tensor(sqrt_g_j_true_np, dtype=torch.float64)
        q_priors_torch = torch.tensor(config['priors_q_j'], dtype=torch.float64)
        
        # 2. Create Estimated sqrt_g_j_estimated_np (eigenvalues of an "estimated" G_est)
        # This part simulates having an imperfect estimate of G, by perturbing its eigenvalues
        if config['estimation_error_type'] == 'fixed_bias_plus_random_noise':
            sqrt_g_j_estimated_np = sqrt_g_j_true_np + config.get('fixed_estimation_bias_offset', 0.0)
        elif config['estimation_error_type'] == 'invert_trend':
            sqrt_g_j_estimated_np = np.sort(sqrt_g_j_true_np) if np.all(np.diff(sqrt_g_j_true_np) <= 1e-9) else np.random.permutation(sqrt_g_j_true_np)
        else: 
            sqrt_g_j_estimated_np = sqrt_g_j_true_np.copy()
        sqrt_g_j_estimated_np += (np.random.rand(config['M_states']) - 0.5) * config['estimation_noise_std']
        sqrt_g_j_estimated_np = np.maximum(0.01, sqrt_g_j_estimated_np) 
        # Ensure sum of (estimated g_j) is M for consistency
        g_j_est_temp = sqrt_g_j_estimated_np**2 
        if np.sum(g_j_est_temp) > 1e-9: sqrt_g_j_estimated_np = np.sqrt(config['M_states'] * g_j_est_temp / np.sum(g_j_est_temp))
        else: sqrt_g_j_estimated_np = np.sqrt(generate_random_gram_eigenvalues(config['M_states'], structured_pattern='uniform_like', ensure_sum_M=True)[0]**2) # Fallback
        sqrt_g_j_estimated_np = np.maximum(0.01, sqrt_g_j_estimated_np)
        x_estimated_torch = torch.tensor(sqrt_g_j_estimated_np, dtype=torch.float64)

        # PGM Performance
        phi_pgm_np = np.zeros(config['M_states']) 
        I_AB_pgm_on_true = calculate_I_AB_numpy(phi_pgm_np, config['M_states'], sqrt_g_j_true_np, config['priors_q_j'])
        pgm_IAB_on_true_trials.append(I_AB_pgm_on_true) 

        # Nominal Optimization
        phi_nom_opt_np, _ = optimize_phases_pytorch(config['M_states'], x_estimated_torch, q_priors_torch, is_robust=False, num_epochs=config['max_pytorch_opt_epochs'], lr=config['pytorch_lr'], trial_info=f"P{process_id}Tr{i_trial+1}Nom")
        I_AB_phi_nom_on_true = calculate_I_AB_numpy(phi_nom_opt_np, config['M_states'], sqrt_g_j_true_np, config['priors_q_j'])
        nominal_IAB_on_true_trials.append(I_AB_phi_nom_on_true)

        # Robust Optimization
        L2_ball_rad_config = config['L2_uncertainty_ball_radius']
        phi_rob_opt_np, _ = optimize_phases_pytorch(config['M_states'], x_true_torch, q_priors_torch, is_robust=True, x_center_for_robust_np=sqrt_g_j_estimated_np, L2_ball_radius_for_robust=L2_ball_rad_config, num_epochs=config['max_pytorch_opt_epochs'], lr=config['pytorch_lr'], trial_info=f"P{process_id}Tr{i_trial+1}Rob")
        I_AB_phi_rob_on_true = calculate_I_AB_numpy(phi_rob_opt_np, config['M_states'], sqrt_g_j_true_np, config['priors_q_j'])
        robust_IAB_on_true_trials.append(I_AB_phi_rob_on_true)
        
        if (i_trial + 1) % (num_trials_this_experiment // 5 or 1) == 0 :
            print(f"    [P{process_id}] Exp {experiment_idx+1}, T{i_trial+1}: PGM={I_AB_pgm_on_true:.4f}, Nom={I_AB_phi_nom_on_true:.4f}, Rob={I_AB_phi_rob_on_true:.4f}")

    avg_pgm_IAB = np.mean(pgm_IAB_on_true_trials); avg_nominal_IAB = np.mean(nominal_IAB_on_true_trials); avg_robust_IAB = np.mean(robust_IAB_on_true_trials)
    t_statistic_rob_gt_nom, p_value_rob_gt_nom = -1, 1.0 
    t_statistic_nom_gt_pgm, p_value_nom_gt_pgm = -1, 1.0
    t_statistic_rob_gt_pgm, p_value_rob_gt_pgm = -1, 1.0
    if num_trials_this_experiment > 1:
        # (T-test logic as before, ensuring valid arrays for stats.ttest_rel)
        # Robust vs Nominal
        nom_v, rob_v = np.array(nominal_IAB_on_true_trials), np.array(robust_IAB_on_true_trials)
        mask = ~np.isnan(nom_v) & ~np.isnan(rob_v)
        if np.sum(mask) > 1: t_statistic_rob_gt_nom, p_value_rob_gt_nom = stats.ttest_rel(rob_v[mask], nom_v[mask], alternative='greater')
        # Nominal vs PGM
        pgm_v = np.array(pgm_IAB_on_true_trials)
        mask_nom_pgm = ~np.isnan(nom_v) & ~np.isnan(pgm_v)
        if np.sum(mask_nom_pgm) > 1: t_statistic_nom_gt_pgm, p_value_nom_gt_pgm = stats.ttest_rel(nom_v[mask_nom_pgm], pgm_v[mask_nom_pgm], alternative='greater')
        # Robust vs PGM
        mask_rob_pgm = ~np.isnan(rob_v) & ~np.isnan(pgm_v)
        if np.sum(mask_rob_pgm) > 1: t_statistic_rob_gt_pgm, p_value_rob_gt_pgm = stats.ttest_rel(rob_v[mask_rob_pgm], pgm_v[mask_rob_pgm], alternative='greater')

    result_summary = {"config": config, "avg_pgm_IAB": avg_pgm_IAB, "avg_nominal_IAB": avg_nominal_IAB, 
                      "avg_robust_IAB": avg_robust_IAB, 
                      "t_statistic_rob_gt_nom": t_statistic_rob_gt_nom, "p_value_rob_gt_nom": p_value_rob_gt_nom,
                      "t_statistic_nom_gt_pgm": t_statistic_nom_gt_pgm, "p_value_nom_gt_pgm": p_value_nom_gt_pgm,
                      "t_statistic_rob_gt_pgm": t_statistic_rob_gt_pgm, "p_value_rob_gt_pgm": p_value_rob_gt_pgm
                     }
    RESULTS_DIR_INDIVIDUAL = "results_beam_search_individual_targeted_Gmatrix" 
    os.makedirs(RESULTS_DIR_INDIVIDUAL, exist_ok=True)
    filename = f"exp_M{config['M_states']}_biasType{config['estimation_error_type']}_noiseStd{config['estimation_noise_std']:.3f}_L2rad{config['L2_uncertainty_ball_radius']:.3f}.json"
    filepath = os.path.join(RESULTS_DIR_INDIVIDUAL, filename)
    try:
        serializable_config = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in result_summary["config"].items()}
        serializable_summary = {key: val for key, val in result_summary.items() if key != "config"}
        serializable_summary["config_params"] = serializable_config 
        with open(filepath, 'w') as f: json.dump(serializable_summary, f, indent=4)
    except Exception as e: print(f"  [P{process_id}] Exp {experiment_idx+1} FAILED to save results: {e}")
    print(f"[P{process_id}] Finished Exp {experiment_idx+1}/{total_experiments}")
    return result_summary

if __name__ == '__main__':
    NUM_DISTINCT_STATES_M_SWEEP = [3] 
    fixed_bias_offset_values = [0.15] # Example: underestimate, no bias, overestimate
    estimation_noise_std_values = [0.01, 0.05] 
    L2_uncertainty_ball_radius_values = [0.05, 0.1] 

    config_base = {
        'num_trials_per_experiment': 30, 
        'true_g_pattern': 'dominant_first', 
        'estimation_error_type': 'fixed_bias_plus_random_noise', 
        'max_pytorch_opt_epochs': 300, 'pytorch_lr': 0.005 
    }
    print(f"--- Starting Parameter Sweep (Explicit Gram Matrix Eigenvalue Perturbation) ---")
    param_configurations_for_pool = []
    param_combinations = list(product(NUM_DISTINCT_STATES_M_SWEEP, fixed_bias_offset_values, estimation_noise_std_values, L2_uncertainty_ball_radius_values))
    total_experiments = len(param_combinations)
    for i_exp, (n_states, bias_offset, est_noise_std, l2_rad) in enumerate(param_combinations):
        current_config = config_base.copy()
        current_config['M_states'] = n_states
        current_config['fixed_estimation_bias_offset'] = bias_offset 
        if n_states == 3: current_config['priors_q_j'] = np.array([0.7, 0.15, 0.15])
        elif n_states == 5: current_config['priors_q_j'] = np.array([0.4,0.25,0.15,0.1,0.1]) # Kept for flexibility
        else: current_config['priors_q_j'] = np.array([1.0/n_states]*n_states)
        current_config['priors_q_j'] /= np.sum(current_config['priors_q_j']) 
        current_config['estimation_noise_std'] = est_noise_std
        current_config['L2_uncertainty_ball_radius'] = l2_rad
        param_configurations_for_pool.append((current_config, i_exp, total_experiments, None)) 

    num_processes = min(multiprocessing.cpu_count(), len(param_configurations_for_pool), 18) 
    print(f"Using {num_processes} processes for parallel execution.")
    all_results_summary_list = []
    if num_processes > 0 and len(param_configurations_for_pool) > 0:
        with multiprocessing.Pool(processes=num_processes) as pool:
            all_results_summary_list = pool.map(run_single_experiment_config, param_configurations_for_pool)
    elif len(param_configurations_for_pool) > 0: 
        print("Running experiments sequentially...")
        for config_tuple_val in param_configurations_for_pool:
            all_results_summary_list.append(run_single_experiment_config((*config_tuple_val[:3], os.getpid()))) # Add pid
            
    all_results_summary_list.sort(key=lambda r: (r['config']['M_states'], r['config'].get('fixed_estimation_bias_offset',0), r['config']['estimation_noise_std'], r['config']['L2_uncertainty_ball_radius']))
    print("\n\n--- Parameter Sweep Results Summary (Explicit Gram Matrix Eigenvalue Perturbation) ---")
    header = "| M | Bias | EstNoiseStd | L2 Radius | Avg PGM IAB   | AvgNominal IAB | AvgRobust IAB | t(Rob>Nom) | p(Rob>Nom) | t(Nom>PGM) | p(Nom>PGM) | t(Rob>PGM) | p(Rob>PGM) |"
    print(header); print("|" + "-"*(len(header)-2) + "|")
    for res in all_results_summary_list:
        cfg = res['config']
        print(f"| {cfg['M_states']:<1d} | {cfg.get('fixed_estimation_bias_offset',0):<4.2f} | {cfg['estimation_noise_std']:<11.3f} | {cfg['L2_uncertainty_ball_radius']:<9.3f} | "
              f"{res['avg_pgm_IAB']:<13.4f} | {res['avg_nominal_IAB']:<14.4f} | {res['avg_robust_IAB']:<13.4f} | "
              f"{res['t_statistic_rob_gt_nom']:<10.3f} | {res['p_value_rob_gt_nom']:<10.4f} | "
              f"{res['t_statistic_nom_gt_pgm']:<10.3f} | {res['p_value_nom_gt_pgm']:<10.4f} | "
              f"{res['t_statistic_rob_gt_pgm']:<10.3f} | {res['p_value_rob_gt_pgm']:<10.4f} |")

    RESULTS_DIR_SWEEP = "results_beam_search_summary_G_eigen_perturb" 
    os.makedirs(RESULTS_DIR_SWEEP, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR_SWEEP, f"param_sweep_summary_L2_Robust_G_eigen_perturb.csv")
    with open(csv_path, 'w') as f:
        f.write("M_states,FixedBias,EstNoiseStd,L2Radius,AvgPgmIAB,AvgNominalIAB,AvgRobustIAB,TRobGtNom,PRobGtNom,TNomGtPGM,PNomGtPGM,TRobGtPGM,PRobGtPGM\n")
        for res in all_results_summary_list:
            cfg = res['config']
            f.write(f"{cfg['M_states']},{cfg.get('fixed_estimation_bias_offset',0)},{cfg['estimation_noise_std']},{cfg['L2_uncertainty_ball_radius']},"
                    f"{res['avg_pgm_IAB']},{res['avg_nominal_IAB']},{res['avg_robust_IAB']},"
                    f"{res['t_statistic_rob_gt_nom']},{res['p_value_rob_gt_nom']},"
                    f"{res['t_statistic_nom_gt_pgm']},{res['p_value_nom_gt_pgm']},"
                    f"{res['t_statistic_rob_gt_pgm']},{res['p_value_rob_gt_pgm']}\n")
    print(f"\nSummary results saved to CSV: {csv_path}")

