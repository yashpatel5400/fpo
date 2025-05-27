import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
from scipy import stats # For t-test
from itertools import product 
import multiprocessing # Added for parallel execution
import json # For saving individual experiment results

# --- Assume constants are available if run externally ---
try:
    from constants import GLOBAL_HBAR, GLOBAL_M
except ImportError:
    GLOBAL_HBAR = 1.0; GLOBAL_M = 1.0 # Dummy values

# --- SNN Model Definition ---
class SimpleSpectralOperatorCNN(nn.Module):
    def __init__(self, K_trunc, hidden_channels=64, num_hidden_layers=3):
        super().__init__()
        self.K_trunc = K_trunc
        layers = []
        layers.append(nn.Conv2d(2, hidden_channels, kernel_size=3, padding='same'))
        layers.append(nn.ReLU())
        for _ in range(num_hidden_layers):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding='same'))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(hidden_channels, 2, kernel_size=3, padding='same'))
        self.model = nn.Sequential(*layers)
    def forward(self, x_spec_ch): return self.model(x_spec_ch)

def spectrum_complex_to_channels_torch(spectrum_mat_complex):
    if not isinstance(spectrum_mat_complex, torch.Tensor):
        spectrum_mat_complex = torch.from_numpy(spectrum_mat_complex)
    if not torch.is_complex(spectrum_mat_complex): 
        if spectrum_mat_complex.shape[0] == 2 and len(spectrum_mat_complex.shape) == 3: 
            return spectrum_mat_complex.float() 
        raise ValueError("Input spectrum_mat_complex must be a complex tensor or already 2-channel real.")
    return torch.stack([spectrum_mat_complex.real, spectrum_mat_complex.imag], dim=0).float()

def channels_to_spectrum_complex_torch(channels_mat_real_imag):
    if channels_mat_real_imag.shape[0] != 2:
        raise ValueError("Input must have 2 channels (real and imaginary parts).")
    return torch.complex(channels_mat_real_imag[0], channels_mat_real_imag[1])

# --- Functions for State Generation and Channel Simulation ---
def random_low_order_state(N, K_max_band=16):
    freq_array = np.zeros((N,N), dtype=np.complex128)
    def wrap_index(k_val): return k_val % N
    for kx_eff in range(-K_max_band, K_max_band + 1):
        for ky_eff in range(-K_max_band, K_max_band + 1):
            c = np.random.randn() + 1j*np.random.randn()
            freq_array[wrap_index(kx_eff), wrap_index(ky_eff)] = c
    psi0_real = np.fft.ifft2(freq_array) 
    norm_psi = np.linalg.norm(psi0_real) 
    if norm_psi > 1e-14: psi0_real /= norm_psi
    return psi0_real

def get_truncated_spectrum(psi_real_space, K_trunc):
    N_grid_psi = psi_real_space.shape[0]
    F_psi_shifted = np.fft.fftshift(np.fft.fft2(psi_real_space))
    if K_trunc > N_grid_psi: raise ValueError(f"K_trunc too large.")
    start_idx = N_grid_psi // 2 - K_trunc // 2
    end_idx = start_idx + K_trunc
    truncated_block = F_psi_shifted[start_idx:end_idx, start_idx:end_idx]
    norm_sq_block = np.sum(np.abs(truncated_block)**2)
    if norm_sq_block > 1e-14: return truncated_block / np.sqrt(norm_sq_block)
    return np.zeros_like(truncated_block)

def apply_phenomenological_noise_channel(spectrum_b_normalized_truncated, config_channel_noise):
    c_current = spectrum_b_normalized_truncated.copy(); K_trunc = c_current.shape[0]
    if K_trunc == 0: return c_current
    if K_trunc % 2 == 0: k_vals_one_dim_eff = np.arange(-K_trunc//2, K_trunc//2)
    else: k_vals_one_dim_eff = np.arange(-(K_trunc-1)//2, (K_trunc-1)//2 + 1)
    kx_eff_grid, ky_eff_grid = np.meshgrid(k_vals_one_dim_eff, k_vals_one_dim_eff, indexing='ij')
    if config_channel_noise.get('apply_attenuation', False):
        loss_factor = config_channel_noise.get('attenuation_loss_factor', 0.1) 
        norm_sq_grid = kx_eff_grid**2 + ky_eff_grid**2
        max_norm_sq_in_trunc = np.max(norm_sq_grid) if K_trunc > 1 else 1.0
        if max_norm_sq_in_trunc < 1e-9: max_norm_sq_in_trunc = 1.0 
        attenuation_profile = np.exp(-loss_factor * norm_sq_grid / max_norm_sq_in_trunc)
        c_current = c_current * attenuation_profile
    if config_channel_noise.get('apply_additive_sobolev_noise', False):
        noise_level_base = config_channel_noise.get('sobolev_noise_level_base', 0.01)
        sobolev_order_s = config_channel_noise.get('sobolev_order_s', 1.0) 
        base_complex_noise = np.random.randn(K_trunc, K_trunc) + 1j * np.random.randn(K_trunc, K_trunc)
        sobolev_denominators = (1 + kx_eff_grid**2 + ky_eff_grid**2)**(sobolev_order_s / 2.0)
        sobolev_denominators[sobolev_denominators < 1e-9] = 1.0 
        scaled_noise = noise_level_base * (base_complex_noise / sobolev_denominators)
        c_current = c_current + scaled_noise
    if config_channel_noise.get('apply_phase_noise', False):
        phase_noise_std = config_channel_noise.get('phase_noise_std_rad', 0.1) 
        random_phases = np.random.randn(K_trunc, K_trunc) * phase_noise_std 
        c_current = c_current * np.exp(1j * random_phases)
    norm_factor_sq = np.sum(np.abs(c_current)**2)
    if norm_factor_sq > 1e-12: c_current /= np.sqrt(norm_factor_sq)
    else: c_current = np.zeros_like(c_current)
    return c_current

def generate_initial_GUS_states_via_phase_ramp(M_states, N_grid, L_domain, k_gamma0_band_limit, delta_n_vector):
    initial_states_real_space = []
    gamma_0_real = random_low_order_state(N_grid, K_max_band=k_gamma0_band_limit)
    if isinstance(L_domain, (int, float)): L_domain_vec = np.array([L_domain] * gamma_0_real.ndim)
    elif len(L_domain) == gamma_0_real.ndim: L_domain_vec = np.array(L_domain)
    else: raise ValueError("L_domain mismatch.")
    coords = [(np.linspace(0,L_d,N_grid,endpoint=False)/L_d) for L_d in L_domain_vec]
    norm_coord_grids = np.meshgrid(*coords, indexing='ij')
    delta_n_dot_x_norm = sum(delta_n_vector[d_idx] * norm_coord_grids[d_idx] for d_idx in range(gamma_0_real.ndim))
    for k_state_idx in range(M_states):
        phase_factor_k = (2 * np.pi * k_state_idx / M_states) * delta_n_dot_x_norm
        gamma_k_real = gamma_0_real * np.exp(1j * phase_factor_k)
        norm_gamma_k = np.linalg.norm(gamma_k_real)
        if norm_gamma_k > 1e-14: gamma_k_real /= norm_gamma_k
        else: 
            gamma_k_real = np.zeros_like(gamma_0_real)
            if N_grid > 0: gamma_k_real.flat[0] = 1.0 / np.sqrt(N_grid**gamma_0_real.ndim)
        initial_states_real_space.append(gamma_k_real)
    return initial_states_real_space, delta_n_vector

def calculate_conditional_probs_p_j_given_k_torch(phi_full_torch, x_s_torch, M_val):
    p_j_k_matrix = torch.zeros((M_val, M_val), dtype=torch.float64, device=phi_full_torch.device)
    s_indices = torch.arange(M_val, device=phi_full_torch.device, dtype=torch.float64)
    y_s_real = x_s_torch * torch.cos(phi_full_torch); y_s_imag = x_s_torch * (-torch.sin(phi_full_torch))
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
    if M_val == 1: return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
    device = x_s_torch.device 
    if phi_params_torch.numel() > 0 : device = phi_params_torch.device
    if phi_params_torch.ndim == 0 and M_val > 1 and phi_params_torch.numel() == 0: phi_full_torch = torch.zeros(M_val, dtype=torch.float64, device=device)
    elif M_val == 1: phi_full_torch = torch.tensor([0.0], dtype=torch.float64, device=device)
    else: phi_full_torch = torch.cat((torch.tensor([0.0], dtype=torch.float64, device=device), phi_params_torch))
    p_j_given_k_matrix = calculate_conditional_probs_p_j_given_k_torch(phi_full_torch, x_s_torch, M_val)
    p_j_given_0 = p_j_given_k_matrix[0, :]; H_cond = -torch.sum(torch.where(p_j_given_0 > eps, p_j_given_0 * torch.log2(p_j_given_0), torch.tensor(0.0, device=device)))
    P_B_j_list = [torch.sum(q_priors_torch * p_j_given_k_matrix[:, j_idx]) for j_idx in range(M_val)]
    P_B = torch.stack(P_B_j_list); P_B_sum = torch.sum(P_B)
    P_B = torch.where(P_B_sum > 1e-9, P_B / (P_B_sum + eps), torch.ones_like(P_B) / M_val) 
    H_B = -torch.sum(torch.where(P_B > eps, P_B * torch.log2(P_B), torch.tensor(0.0, device=device)))
    return H_B - H_cond, H_B, H_cond

def calculate_I_AB_numpy(phi_values_np, M_val, x_s_np, q_priors_np, eps=1e-12):
    x_torch = torch.tensor(x_s_np, dtype=torch.float64); q_torch = torch.tensor(q_priors_np, dtype=torch.float64)
    if M_val == 1: phi_params_torch = torch.tensor([], dtype=torch.float64)
    else:
        phi_params_np_slice = phi_values_np[1:] 
        if not isinstance(phi_params_np_slice, np.ndarray): phi_params_np_slice = np.array(phi_params_np_slice)
        if phi_params_np_slice.ndim == 0: phi_params_torch = torch.tensor([phi_params_np_slice.item()], dtype=torch.float64) if phi_params_np_slice.size >0 else torch.tensor([], dtype=torch.float64)
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

# --- Main Experiment Function (Modified for Multiprocessing) ---
def run_single_experiment_with_snn(args_tuple): # MODIFIED: Takes a single tuple argument
    config, experiment_idx, total_experiments, snn_model_state_dict_arg, device_str_arg = args_tuple # UNPACK
    process_id = os.getpid() 
    
    # Ensure device is correctly set for this worker process
    device = torch.device(device_str_arg)

    print(f"[P{process_id}] Exp {experiment_idx+1}/{total_experiments}: "
          f"L2Rad={config['L2_uncertainty_ball_radius']:.3f}, M={config['M_states']}")
    
    snn = SimpleSpectralOperatorCNN(K_trunc=config['K_TRUNC_SNN'], 
                                    hidden_channels=config['snn_hidden_channels'],
                                    num_hidden_layers=config['snn_num_hidden_layers'])
    snn.load_state_dict(snn_model_state_dict_arg); snn.to(device); snn.eval() # Use unpacked args

    num_trials_this_experiment = config['num_trials_per_experiment']
    pgm_IAB_on_true_trials, nominal_IAB_on_true_trials, robust_IAB_on_true_trials = [], [], []

    for i_trial in range(num_trials_this_experiment):
        initial_states_psi0_real, _ = generate_initial_GUS_states_via_phase_ramp(
            config['M_states'], config['n_grid_snn_input'], config['l_domain_snn_input'], 
            config['k_gamma0_band_limit'], config['delta_n_vector_gus']   
        )
        list_gamma_b_k_spectra = [get_truncated_spectrum(psi0_real, config['K_TRUNC_SNN']) for psi0_real in initial_states_psi0_real]
        list_gamma_a_k_true_spectra = [apply_phenomenological_noise_channel(gb_spec, config['true_channel_config']) for gb_spec in list_gamma_b_k_spectra]
        
        G_true = np.zeros((config['M_states'], config['M_states']), dtype=np.complex128)
        for r_idx in range(config['M_states']):
            for c_idx in range(config['M_states']):
                G_true[r_idx, c_idx] = np.vdot(list_gamma_a_k_true_spectra[r_idx].ravel(), list_gamma_a_k_true_spectra[c_idx].ravel())
        g_j_true_vals = np.linalg.eigvalsh(G_true); g_j_true_vals = np.maximum(1e-7, g_j_true_vals) 
        sqrt_g_j_true_np = np.sqrt(np.sort(g_j_true_vals)[::-1]) 
        x_true_torch = torch.tensor(sqrt_g_j_true_np, dtype=torch.float64, device=device)
        q_priors_torch = torch.tensor(config['priors_q_j'], dtype=torch.float64, device=device)

        list_gamma_a_k_snn_pred_spectra_np = []
        with torch.no_grad():
            for gamma_b_spec in list_gamma_b_k_spectra:
                gamma_b_channels = spectrum_complex_to_channels_torch(gamma_b_spec).unsqueeze(0).to(device) 
                gamma_a_pred_channels = snn(gamma_b_channels)
                gamma_a_pred_complex = channels_to_spectrum_complex_torch(gamma_a_pred_channels.squeeze(0).cpu())
                norm_pred_sq = torch.sum(torch.abs(gamma_a_pred_complex)**2)
                if norm_pred_sq > 1e-12: gamma_a_pred_complex /= torch.sqrt(norm_pred_sq)
                else: gamma_a_pred_complex = torch.zeros_like(gamma_a_pred_complex)
                list_gamma_a_k_snn_pred_spectra_np.append(gamma_a_pred_complex.numpy())
        
        G_est = np.zeros((config['M_states'], config['M_states']), dtype=np.complex128)
        for r_idx in range(config['M_states']):
            for c_idx in range(config['M_states']):
                G_est[r_idx, c_idx] = np.vdot(list_gamma_a_k_snn_pred_spectra_np[r_idx].ravel(), list_gamma_a_k_snn_pred_spectra_np[c_idx].ravel())
        g_j_estimated_vals = np.linalg.eigvalsh(G_est); g_j_estimated_vals = np.maximum(1e-7, g_j_estimated_vals)
        sqrt_g_j_estimated_np = np.sqrt(np.sort(g_j_estimated_vals)[::-1])
        x_estimated_torch = torch.tensor(sqrt_g_j_estimated_np, dtype=torch.float64, device=device)

        phi_pgm_np = np.zeros(config['M_states']) 
        I_AB_pgm_on_true = calculate_I_AB_numpy(phi_pgm_np, config['M_states'], sqrt_g_j_true_np, config['priors_q_j'])
        pgm_IAB_on_true_trials.append(I_AB_pgm_on_true) 

        phi_nom_opt_np, _ = optimize_phases_pytorch(config['M_states'], x_estimated_torch, q_priors_torch, is_robust=False, num_epochs=config['max_pytorch_opt_epochs'], lr=config['pytorch_lr'], trial_info=f"P{process_id}Tr{i_trial+1}Nom")
        I_AB_phi_nom_on_true = calculate_I_AB_numpy(phi_nom_opt_np, config['M_states'], sqrt_g_j_true_np, config['priors_q_j'])
        nominal_IAB_on_true_trials.append(I_AB_phi_nom_on_true)

        L2_ball_rad_config = config['L2_uncertainty_ball_radius']
        phi_rob_opt_np, _ = optimize_phases_pytorch(config['M_states'], x_true_torch, q_priors_torch, is_robust=True, x_center_for_robust_np=sqrt_g_j_estimated_np, L2_ball_radius_for_robust=L2_ball_rad_config, num_epochs=config['max_pytorch_opt_epochs'], lr=config['pytorch_lr'], trial_info=f"P{process_id}Tr{i_trial+1}Rob")
        I_AB_phi_rob_on_true = calculate_I_AB_numpy(phi_rob_opt_np, config['M_states'], sqrt_g_j_true_np, config['priors_q_j'])
        robust_IAB_on_true_trials.append(I_AB_phi_rob_on_true)
        
        if (i_trial + 1) % (num_trials_this_experiment // 5 or 1) == 0 :
            print(f"    [P{process_id}] Exp {experiment_idx+1}, T{i_trial+1}: PGM={I_AB_pgm_on_true:.4f}, Nom={I_AB_phi_nom_on_true:.4f}, Rob={I_AB_phi_rob_on_true:.4f}")

    avg_pgm_IAB = np.mean(pgm_IAB_on_true_trials); avg_nominal_IAB = np.mean(nominal_IAB_on_true_trials); avg_robust_IAB = np.mean(robust_IAB_on_true_trials)
    t_statistic_rob_gt_nom, p_value_rob_gt_nom = -1.0, 1.0 # Ensure float for JSON
    t_statistic_nom_gt_pgm, p_value_nom_gt_pgm = -1.0, 1.0
    t_statistic_rob_gt_pgm, p_value_rob_gt_pgm = -1.0, 1.0
    if num_trials_this_experiment > 1:
        nom_v, rob_v, pgm_v = np.array(nominal_IAB_on_true_trials), np.array(robust_IAB_on_true_trials), np.array(pgm_IAB_on_true_trials)
        mask = ~np.isnan(nom_v) & ~np.isnan(rob_v) & ~np.isnan(pgm_v)
        nom_v, rob_v, pgm_v = nom_v[mask], rob_v[mask], pgm_v[mask]
        if len(rob_v) > 1: # Check length after masking
            t_statistic_rob_gt_nom, p_value_rob_gt_nom = stats.ttest_rel(rob_v, nom_v, alternative='greater')
            t_statistic_nom_gt_pgm, p_value_nom_gt_pgm = stats.ttest_rel(nom_v, pgm_v, alternative='greater')
            t_statistic_rob_gt_pgm, p_value_rob_gt_pgm = stats.ttest_rel(rob_v, pgm_v, alternative='greater')
            # Handle potential NaN from ttest_rel if arrays are identical or have issues
            if np.isnan(t_statistic_rob_gt_nom): t_statistic_rob_gt_nom = 0.0; p_value_rob_gt_nom = 0.5 if np.allclose(rob_v, nom_v) else 1.0
            if np.isnan(t_statistic_nom_gt_pgm): t_statistic_nom_gt_pgm = 0.0; p_value_nom_gt_pgm = 0.5 if np.allclose(nom_v, pgm_v) else 1.0
            if np.isnan(t_statistic_rob_gt_pgm): t_statistic_rob_gt_pgm = 0.0; p_value_rob_gt_pgm = 0.5 if np.allclose(rob_v, pgm_v) else 1.0

    result_summary = {"config": config, "avg_pgm_IAB": avg_pgm_IAB, "avg_nominal_IAB": avg_nominal_IAB, 
                      "avg_robust_IAB": avg_robust_IAB, 
                      "t_statistic_rob_gt_nom": float(t_statistic_rob_gt_nom), "p_value_rob_gt_nom": float(p_value_rob_gt_nom),
                      "t_statistic_nom_gt_pgm": float(t_statistic_nom_gt_pgm), "p_value_nom_gt_pgm": float(p_value_nom_gt_pgm),
                      "t_statistic_rob_gt_pgm": float(t_statistic_rob_gt_pgm), "p_value_rob_gt_pgm": float(p_value_rob_gt_pgm)
                     }
    RESULTS_DIR_INDIVIDUAL = "results_snn_robust_sweep_individual" 
    os.makedirs(RESULTS_DIR_INDIVIDUAL, exist_ok=True)
    filename = f"exp_M{config['M_states']}_L2rad{config['L2_uncertainty_ball_radius']:.3f}_SNN.json" 
    filepath = os.path.join(RESULTS_DIR_INDIVIDUAL, filename)
    try:
        serializable_config = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in result_summary["config"].items()}
        serializable_summary = {key: (float(val) if isinstance(val, (np.float32, np.float64, np.float_)) else val) for key, val in result_summary.items() if key != "config"} 
        serializable_summary["config_params"] = serializable_config 
        with open(filepath, 'w') as f: json.dump(serializable_summary, f, indent=4)
    except Exception as e: print(f"  [P{process_id}] Exp {experiment_idx+1} FAILED to save results: {e}")
    print(f"[P{process_id}] Finished Exp {experiment_idx+1}/{total_experiments}")
    return result_summary

if __name__ == '__main__':
    SNN_MODEL_PATH = "trained_snn_models/snn_K32_H64_L3.pth" 
    K_TRUNC_SNN_CONFIG = 32 
    N_GRID_SNN_INPUT_CONFIG = 64 
    K_GAMMA0_BAND_LIMIT_CONFIG = 12 
    DELTA_N_VECTOR_GUS_CONFIG = np.array([1,0]) 

    TRUE_CHANNEL_CONFIG = {
        'apply_attenuation': True, 'attenuation_loss_factor': 0.2,
        'apply_additive_sobolev_noise': True, 'sobolev_noise_level_base': 0.02, 'sobolev_order_s': 1.0,
        'apply_phase_noise': True, 'phase_noise_std_rad': 0.15
    }
    
    NUM_DISTINCT_STATES_M_SWEEP = [3, 5] 
    L2_uncertainty_ball_radius_values = [0.05, 0.1, 0.2] 

    config_base = {
        'num_trials_per_experiment': 30, 
        'K_TRUNC_SNN': K_TRUNC_SNN_CONFIG,
        'n_grid_snn_input': N_GRID_SNN_INPUT_CONFIG,
        'l_domain_snn_input': 2 * np.pi, 
        'k_gamma0_band_limit': K_GAMMA0_BAND_LIMIT_CONFIG,
        'delta_n_vector_gus': DELTA_N_VECTOR_GUS_CONFIG.tolist(), 
        'true_channel_config': TRUE_CHANNEL_CONFIG,
        'snn_hidden_channels': 64, 
        'snn_num_hidden_layers': 3, 
        'max_pytorch_opt_epochs': 300, 
        'pytorch_lr': 0.005 
    }
    print(f"--- Starting SNN-based Parameter Sweep (Non-Orthogonal GUS Initial States) ---")
    
    if not os.path.exists(SNN_MODEL_PATH):
        print(f"ERROR: SNN model file not found at {SNN_MODEL_PATH}. Please train the SNN first or provide the correct path.")
        exit()
    snn_model_state_dict_main = torch.load(SNN_MODEL_PATH, map_location=torch.device('cpu')) 

    param_configurations_for_pool = []
    param_combinations = list(product(NUM_DISTINCT_STATES_M_SWEEP, L2_uncertainty_ball_radius_values))
    total_experiments = len(param_combinations)

    for i_exp, (n_states, l2_rad) in enumerate(param_combinations):
        current_config = config_base.copy()
        current_config['M_states'] = n_states
        if n_states == 3: current_config['priors_q_j'] = np.array([0.7, 0.15, 0.15])
        elif n_states == 5: current_config['priors_q_j'] = np.array([0.4,0.25,0.15,0.1,0.1])
        else: current_config['priors_q_j'] = np.array([1.0/n_states]*n_states)
        current_config['priors_q_j'] = current_config['priors_q_j'] / np.sum(current_config['priors_q_j'])
        current_config['L2_uncertainty_ball_radius'] = l2_rad
        # Package all arguments for the worker function into a single tuple
        param_configurations_for_pool.append((current_config, i_exp, total_experiments, snn_model_state_dict_main, "cpu")) 

    num_processes = min(multiprocessing.cpu_count(), len(param_configurations_for_pool), 6) 
    print(f"Using {num_processes} processes for parallel execution.")
    
    all_results_summary_list = []
    if num_processes > 0 and len(param_configurations_for_pool) > 0:
        with multiprocessing.Pool(processes=num_processes) as pool:
            all_results_summary_list = pool.map(run_single_experiment_with_snn, param_configurations_for_pool)
    elif len(param_configurations_for_pool) > 0: 
        print("Running experiments sequentially...")
        for config_tuple_val in param_configurations_for_pool:
            # Unpack and repack for sequential call to match worker signature
            cfg, exp_idx, tot_exp, sdict, dev = config_tuple_val
            all_results_summary_list.append(run_single_experiment_with_snn((cfg, exp_idx, tot_exp, sdict, dev)))
            
    all_results_summary_list.sort(key=lambda r: (r['config']['M_states'], r['config']['L2_uncertainty_ball_radius']))

    print("\n\n--- Parameter Sweep Results Summary (SNN Surrogate, L2 Ball Robustness, Non-Orthogonal GUS) ---")
    header = "| M | L2 Radius | Avg PGM IAB   | AvgNominal IAB | AvgRobust IAB | t(Rob>Nom) | p(Rob>Nom) | t(Nom>PGM) | p(Nom>PGM) | t(Rob>PGM) | p(Rob>PGM) |"
    print(header); print("|" + "-"*(len(header)-2) + "|")
    for res in all_results_summary_list:
        cfg = res['config']
        print(f"| {cfg['M_states']:<1d} | {cfg['L2_uncertainty_ball_radius']:<9.3f} | "
              f"{res['avg_pgm_IAB']:<13.4f} | {res['avg_nominal_IAB']:<14.4f} | {res['avg_robust_IAB']:<13.4f} | "
              f"{res['t_statistic_rob_gt_nom']:<10.3f} | {res['p_value_rob_gt_nom']:<10.4f} | "
              f"{res['t_statistic_nom_gt_pgm']:<10.3f} | {res['p_value_nom_gt_pgm']:<10.4f} | "
              f"{res['t_statistic_rob_gt_pgm']:<10.3f} | {res['p_value_rob_gt_pgm']:<10.4f} |")

    RESULTS_DIR_SWEEP = "results_snn_robust_sweep_summary_nonorthoGUS" 
    os.makedirs(RESULTS_DIR_SWEEP, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR_SWEEP, f"param_sweep_summary_SNN_L2_Robust_NonOrthoGUS.csv")
    with open(csv_path, 'w') as f:
        f.write("M_states,L2Radius,AvgPgmIAB,AvgNominalIAB,AvgRobustIAB,TRobGtNom,PRobGtNom,TNomGtPGM,PNomGtPGM,TRobGtPGM,PRobGtPGM\n")
        for res in all_results_summary_list:
            cfg = res['config']
            f.write(f"{cfg['M_states']},{cfg['L2_uncertainty_ball_radius']},"
                    f"{res['avg_pgm_IAB']},{res['avg_nominal_IAB']},{res['avg_robust_IAB']},"
                    f"{res['t_statistic_rob_gt_nom']},{res['p_value_rob_gt_nom']},"
                    f"{res['t_statistic_nom_gt_pgm']},{res['p_value_nom_gt_pgm']},"
                    f"{res['t_statistic_rob_gt_pgm']},{res['p_value_rob_gt_pgm']}\n")
    print(f"\nSummary results saved to CSV: {csv_path}")

