import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize
# import matplotlib.pyplot as plt # Kept commented unless debugging
import os
from scipy import stats
import argparse 
import json 

# --- Global constants for solver ---
HBAR_CONST = 1.0
MASS_CONST = 1.0

# --- SNN Model Definition ---
class SimpleSpectralOperatorCNN(nn.Module):
    def __init__(self, K_input_resolution, K_output_resolution, hidden_channels=64, num_hidden_layers=3):
        super().__init__()
        self.K_input_resolution = K_input_resolution
        self.K_output_resolution = K_output_resolution
        
        if K_output_resolution > K_input_resolution:
            raise ValueError("K_output_resolution > K_input_resolution not supported by this SNN design.")
        
        layers = []
        layers.append(nn.Conv2d(2, hidden_channels, kernel_size=3, padding='same'))
        layers.append(nn.ReLU())
        
        for _ in range(num_hidden_layers):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding='same'))
            layers.append(nn.ReLU())
            
        layers.append(nn.Conv2d(hidden_channels, 2, kernel_size=3, padding='same'))
        self.cnn_body = nn.Sequential(*layers)

    def forward(self, x_spec_ch_full_input): 
        x_processed_full = self.cnn_body(x_spec_ch_full_input)
        if self.K_input_resolution == self.K_output_resolution:
            return x_processed_full
        else: 
            start_idx = self.K_input_resolution // 2 - self.K_output_resolution // 2
            end_idx = start_idx + self.K_output_resolution
            return x_processed_full[:, :, start_idx:end_idx, start_idx:end_idx]

# --- Data Handling & Physics Functions ---
def spectrum_complex_to_channels_torch(spectrum_mat_complex):
    if not isinstance(spectrum_mat_complex, torch.Tensor):
        spectrum_mat_complex = torch.from_numpy(spectrum_mat_complex)
    
    if not torch.is_complex(spectrum_mat_complex): 
        if spectrum_mat_complex.ndim == 3 and spectrum_mat_complex.shape[0] == 2: 
            return spectrum_mat_complex.float() 
        if spectrum_mat_complex.ndim == 2 and not torch.is_complex(spectrum_mat_complex):
             raise ValueError(f"Input spectrum_mat_complex is real [K,K] but expected complex or [2,K,K] real.")
        raise ValueError(f"Input spectrum_mat_complex has shape {spectrum_mat_complex.shape}. Expected complex [K,K] or real [2,K,K].")
        
    return torch.stack([spectrum_mat_complex.real, spectrum_mat_complex.imag], dim=0).float()

def channels_to_spectrum_complex_torch(channels_mat_real_imag):
    if channels_mat_real_imag.ndim != 3 or channels_mat_real_imag.shape[0] != 2: 
        raise ValueError(f"Input must have 2 channels as the first dimension, got shape {channels_mat_real_imag.shape}")
    return torch.complex(channels_mat_real_imag[0], channels_mat_real_imag[1])

class GaussianRF(object): 
    def __init__(self, dim, size, alpha=2, tau=3, sigma=None, device=None):
        self.dim = dim
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cpu")

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))

        k_max = size // 2
        if dim == 2:
            import math
            if k_max > 0:
                k_range = torch.arange(start=0, end=k_max, step=1, device=self.device)
                wavenumbers_half = torch.cat((k_range, torch.arange(start=-k_max, end=0, step=1, device=self.device)), 0)
            else:
                wavenumbers_half = torch.tensor([0], device=self.device)
            
            wavenumbers = wavenumbers_half.repeat(size,1)

            k_x = wavenumbers.transpose(0, 1)
            k_y = wavenumbers
            
            self.sqrt_eig = ((size**2)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2+k_y**2)+tau**2)**(-alpha/2.0)))
            if self.sqrt_eig.numel() > 0 and k_max > 0 :
                self.sqrt_eig[0,0] = 0.0
        else:
            raise ValueError("Dimension must be 2 for this GRF in robust_opt.")
        self.size_tuple = tuple([size]*dim)

    def sample(self, N_samples):
        if self.sqrt_eig.numel() == 0 and self.size_tuple[0] > 0 : 
            return torch.zeros(N_samples, *self.size_tuple, dtype=torch.float32) 
            
        coeff = torch.randn(N_samples, *self.size_tuple, dtype=torch.cfloat, device=self.device)
        if self.sqrt_eig.numel() > 0 :
            coeff = self.sqrt_eig * coeff 
            
        return torch.fft.ifftn(coeff, dim=list(range(-self.dim, 0))).real

def get_full_centered_spectrum(psi_real_space):
    N_grid = psi_real_space.shape[0]
    if N_grid == 0:
        return np.array([], dtype=np.complex64)
    return np.fft.fftshift(np.fft.fft2(psi_real_space))

def normalize_spectrum(spectrum_complex):
    if spectrum_complex.size == 0:
        return spectrum_complex
    norm_sq = np.sum(np.abs(spectrum_complex)**2)
    if norm_sq > 1e-14:
        return spectrum_complex / np.sqrt(norm_sq)
    return np.zeros_like(spectrum_complex)

def extract_center_block_np(full_spectrum_centered_np, K_extract):
    N_full = full_spectrum_centered_np.shape[0]
    if K_extract == N_full:
        return full_spectrum_centered_np
    if K_extract > N_full: 
        padded_block = np.zeros((K_extract, K_extract), dtype=full_spectrum_centered_np.dtype)
        start_pad_x = K_extract//2 - N_full//2
        end_pad_x = start_pad_x + N_full
        start_pad_y = K_extract//2 - N_full//2
        end_pad_y = start_pad_y + N_full
        valid_src_s_x=max(0,-start_pad_x)
        valid_src_e_x=N_full-max(0,end_pad_x-K_extract)
        valid_dst_s_x=max(0,start_pad_x)
        valid_dst_e_x=K_extract-max(0,K_extract-end_pad_x)
        valid_src_s_y=max(0,-start_pad_y)
        valid_src_e_y=N_full-max(0,end_pad_y-K_extract)
        valid_dst_s_y=max(0,start_pad_y)
        valid_dst_e_y=K_extract-max(0,K_extract-end_pad_y)
        if valid_src_e_x > valid_src_s_x and valid_src_e_y > valid_src_s_y and \
           valid_dst_e_x > valid_dst_s_x and valid_dst_e_y > valid_dst_s_y:
             padded_block[valid_dst_s_x:valid_dst_e_x, valid_dst_s_y:valid_dst_e_y] = \
                 full_spectrum_centered_np[valid_src_s_x:valid_src_e_x, valid_src_s_y:valid_src_e_y]
        return padded_block
    if K_extract <= 0:
        return np.array([], dtype=np.complex64)
    start_idx = N_full//2 - K_extract//2
    end_idx = start_idx + K_extract
    return full_spectrum_centered_np[start_idx:end_idx, start_idx:end_idx]

def generate_initial_GUS_states_via_phase_ramp(M_states, N_grid, L_domain, grf_alpha, grf_tau, delta_n_vector):
    grf_gen_for_base = GaussianRF(dim=2, size=N_grid, alpha=grf_alpha, tau=grf_tau)
    gamma_0_real_unnorm = grf_gen_for_base.sample(1).cpu().numpy().squeeze()
    norm_gamma0 = np.linalg.norm(gamma_0_real_unnorm)
    if norm_gamma0 > 1e-14:
        gamma_0_real = gamma_0_real_unnorm / norm_gamma0
    else:
        gamma_0_real = np.zeros_like(gamma_0_real_unnorm)
        if N_grid > 0: 
            gamma_0_real.flat[0]=1.0
            gamma_0_real = gamma_0_real / np.linalg.norm(gamma_0_real) 
    
    initial_states_real_space = []
    coords = [(np.linspace(0, L_domain_dim, N_grid, endpoint=False) / L_domain_dim) for L_domain_dim in [L_domain, L_domain]] 
    norm_coord_grids = np.meshgrid(*coords, indexing='ij')
    delta_n_dot_x_norm = sum(delta_n_vector[d_idx] * norm_coord_grids[d_idx] for d_idx in range(2)) 
    
    for k_state_idx in range(M_states):
        phase_factor_k = (2 * np.pi * k_state_idx / M_states) * delta_n_dot_x_norm
        gamma_k_real = gamma_0_real * np.exp(1j * phase_factor_k)
        norm_gamma_k = np.linalg.norm(gamma_k_real) 
        if norm_gamma_k > 1e-14:
            gamma_k_real = gamma_k_real / norm_gamma_k
        initial_states_real_space.append(gamma_k_real)
    return initial_states_real_space

def get_step_index_fiber_potential(N_grid, L_domain, core_radius_factor, potential_depth):
    actual_core_radius = core_radius_factor * (L_domain / 2.0)
    coords1d_edge = np.linspace(-L_domain / 2.0, L_domain / 2.0, N_grid, endpoint=False)
    dx = L_domain / N_grid
    coords1d_center = coords1d_edge + dx / 2.0 
    x_mg, y_mg = np.meshgrid(coords1d_center, coords1d_center, indexing='ij')
    
    rr = np.sqrt(x_mg**2 + y_mg**2)
    potential = np.zeros((N_grid, N_grid), dtype=float)
    potential[rr < actual_core_radius] = -potential_depth 
    return potential

def split_step_solver_2d(V_grid, psi0, N, dx, T, num_steps, hbar_val, m_val):
    dt = T / num_steps
    psi = psi0.astype(np.complex128).copy() 
    
    V_op = np.exp(-0.5j * dt / hbar_val * V_grid)
    
    k_vec = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    kx, ky = np.meshgrid(k_vec, k_vec, indexing='ij')
    k2 = kx**2 + ky**2
    K_op = np.exp(-0.5j * hbar_val * dt / m_val * k2) 

    for _ in range(num_steps):
        psi = psi * V_op 
        psi_k = np.fft.fft2(psi)
        psi_k = psi_k * K_op 
        psi = np.fft.ifft2(psi_k)
        psi = psi * V_op 
    return psi

def solver_main(V_potential, psi0_real_space, N_grid, L_domain, T_evolution, num_solver_steps, hbar, m):
    dx = L_domain / N_grid
    
    norm_psi0 = np.linalg.norm(psi0_real_space)
    if norm_psi0 > 1e-14:
        psi0_normalized = psi0_real_space / norm_psi0
    else: 
        psi0_normalized = np.zeros_like(psi0_real_space)
        if N_grid > 0: 
            psi0_normalized.flat[0] = 1.0 
            psi0_normalized = psi0_normalized / np.linalg.norm(psi0_normalized) 

    psi_T = split_step_solver_2d(V_potential, psi0_normalized, N_grid, dx, T_evolution, num_solver_steps, hbar, m)
    
    norm_psi_T = np.linalg.norm(psi_T)
    if norm_psi_T > 1e-14:
        psi_T = psi_T / norm_psi_T
    return psi_T

# --- Core I_AB Calculation and Optimization Functions ---
def calculate_conditional_probs_p_j_given_k_torch(phi_full_torch, x_s_torch, M_val):
    target_device = x_s_torch.device
    phi_full_torch = phi_full_torch.to(target_device)

    p_j_k_matrix = torch.zeros((M_val, M_val), dtype=torch.float64, device=target_device)
    s_indices = torch.arange(M_val, device=target_device, dtype=torch.float64)
    
    y_s_real = x_s_torch * torch.cos(phi_full_torch) 
    y_s_imag = x_s_torch * (-torch.sin(phi_full_torch))

    for k_sent_idx in range(M_val):
        for j_outcome_idx in range(M_val):
            dft_phase_term = 2 * np.pi * s_indices * (k_sent_idx - j_outcome_idx) / M_val
            cos_dft = torch.cos(dft_phase_term)
            sin_dft = torch.sin(dft_phase_term)
            sum_real = torch.sum(y_s_real * cos_dft - y_s_imag * sin_dft)
            sum_imag = torch.sum(y_s_real * sin_dft + y_s_imag * cos_dft)
            p_j_k_matrix[k_sent_idx, j_outcome_idx] = (1/M_val**2) * (sum_real**2 + sum_imag**2)
            
    row_sums = torch.sum(p_j_k_matrix, dim=1, keepdim=True)
    p_j_k_matrix = torch.where(row_sums > 1e-9, p_j_k_matrix / (row_sums + 1e-15), torch.ones_like(p_j_k_matrix) / M_val)
    return p_j_k_matrix

def calculate_I_AB_components_torch(phi_params_torch, M_val, x_s_torch, q_priors_torch, eps=1e-12):
    target_device = x_s_torch.device
    
    if M_val == 1:
        return torch.tensor(0.0, device=target_device), \
               torch.tensor(0.0, device=target_device), \
               torch.tensor(0.0, device=target_device)
    
    if phi_params_torch.numel() == 0: 
         phi_full_torch = torch.zeros(M_val, dtype=torch.float64, device=target_device) 
    else:
        phi_full_torch = torch.cat((torch.tensor([0.0], dtype=torch.float64, device=target_device), 
                                    phi_params_torch.to(target_device)))
    
    p_j_given_k_matrix = calculate_conditional_probs_p_j_given_k_torch(phi_full_torch, x_s_torch.to(target_device), M_val)
    p_j_given_0 = p_j_given_k_matrix[0, :]
    
    H_cond_terms = p_j_given_0 * torch.log2(p_j_given_0 + eps) 
    H_cond = -torch.sum(torch.where(p_j_given_0 > eps, H_cond_terms, torch.tensor(0.0, device=target_device)))
    
    P_B_j_list = []
    for j_idx in range(M_val):
        P_B_j_list.append(torch.sum(q_priors_torch.to(target_device) * p_j_given_k_matrix[:, j_idx]))
    P_B = torch.stack(P_B_j_list)
    
    P_B_sum = torch.sum(P_B)
    if not torch.isclose(P_B_sum, torch.tensor(1.0, device=target_device, dtype=torch.float64)):
        if P_B_sum > eps:
            P_B = P_B / P_B_sum
        else: 
            P_B = torch.ones_like(P_B, device=target_device) / M_val

    H_B_terms = P_B * torch.log2(P_B + eps) 
    H_B = -torch.sum(torch.where(P_B > eps, H_B_terms, torch.tensor(0.0, device=target_device)))
    
    I_AB = H_B - H_cond
    return I_AB, H_B, H_cond

def calculate_I_AB_numpy(phi_values_np, M_val, x_s_np, q_priors_np, target_device_str="cpu", eps=1e-12):
    device = torch.device(target_device_str)
    x_torch = torch.tensor(x_s_np, dtype=torch.float64).to(device)
    q_torch = torch.tensor(q_priors_np, dtype=torch.float64).to(device)
    
    phi_params_torch = torch.tensor([], dtype=torch.float64).to(device)
    if M_val > 1:
        phi_params_np_slice = phi_values_np[1:] 
        if not isinstance(phi_params_np_slice, np.ndarray):
            phi_params_np_slice = np.array(phi_params_np_slice)
        if phi_params_np_slice.ndim == 0:
            if phi_params_np_slice.size > 0:
                phi_params_torch = torch.tensor([phi_params_np_slice.item()], dtype=torch.float64).to(device)
        else:
            if phi_params_np_slice.size > 0:
                 phi_params_torch = torch.tensor(phi_params_np_slice, dtype=torch.float64).to(device)
            
    I_AB, _, _ = calculate_I_AB_components_torch(phi_params_torch, M_val, x_torch, q_torch, eps)
    return I_AB.item()

def inner_adversarial_objective_scipy(x_tilde_np, M_val, phi_values_rad_np, q_priors_np, target_device_str):
    return calculate_I_AB_numpy(phi_values_rad_np, M_val, x_tilde_np, q_priors_np, target_device_str=target_device_str)

def solve_inner_adversarial_problem_scipy(phi_values_rad_np, M_val, x_center_of_uncertainty_np, 
                                          L2_ball_radius, q_priors_np, target_device_str):
    def l2_constraint(x_adv):
        return L2_ball_radius**2 - np.sum((x_adv - x_center_of_uncertainty_np)**2)
        
    constraints = [{'type':'ineq','fun': l2_constraint}]
    bounds = [(0.001, None) for _ in range(M_val)] 
    
    x0_guess = np.maximum(0.001, x_center_of_uncertainty_np.copy())
    diff_from_center = x0_guess - x_center_of_uncertainty_np
    norm_diff = np.linalg.norm(diff_from_center)
    
    if norm_diff > L2_ball_radius and L2_ball_radius > 1e-9 :
        x0_guess = x_center_of_uncertainty_np + diff_from_center * (L2_ball_radius / norm_diff)
        x0_guess = np.maximum(0.001, x0_guess) 
        
    res = minimize(inner_adversarial_objective_scipy, x0_guess, 
                   args=(M_val, phi_values_rad_np, q_priors_np, target_device_str), 
                   method='SLSQP', bounds=bounds, constraints=constraints, 
                   options={'disp': False, 'ftol': 1e-7, 'maxiter': 200})
    
    if res.success:
        return res.fun, res.x
    else:
        I_AB_at_center = inner_adversarial_objective_scipy(
            x_center_of_uncertainty_np, M_val, phi_values_rad_np, q_priors_np, target_device_str
        )
        return I_AB_at_center, x_center_of_uncertainty_np

def optimize_phases_pytorch(M_val, x_for_opt_torch, q_priors_torch, target_device_obj, 
                            is_robust=False, x_center_for_robust_np=None, 
                            L2_ball_radius_for_robust=None, 
                            num_epochs=300, lr=0.01, initial_phases_np=None, trial_info=""): 
    
    x_for_opt_torch = x_for_opt_torch.to(target_device_obj)
    q_priors_torch = q_priors_torch.to(target_device_obj)

    num_phases_to_opt = M_val - 1
    if M_val == 1: 
        I_AB_val, _, _ = calculate_I_AB_components_torch(
            torch.tensor([], dtype=torch.float64, device=target_device_obj),
            M_val, x_for_opt_torch, q_priors_torch
        )
        return np.array([0.0]), [I_AB_val.item()] * num_epochs 
    
    if initial_phases_np is None or len(initial_phases_np) != num_phases_to_opt : 
        if num_phases_to_opt > 0:
            initial_phases_np = (np.random.rand(num_phases_to_opt) - 0.5) * 0.1
        else:
            initial_phases_np = np.array([])
            
    phi_params = torch.tensor(initial_phases_np, dtype=torch.float64, device=target_device_obj, requires_grad=True)
    
    optimizer = None
    if num_phases_to_opt > 0:
        optimizer = optim.Adam([phi_params], lr=lr)
    
    history_I_AB_during_opt = []
    for epoch in range(num_epochs):
        if optimizer:
            optimizer.zero_grad()
            
        current_phi_params_for_calc = phi_params
        if num_phases_to_opt == 0:
             current_phi_params_for_calc = torch.tensor([], dtype=torch.float64, device=target_device_obj)
        
        loss = torch.tensor(0.0, dtype=torch.float64, device=target_device_obj)
        I_AB_epoch = 0.0
        
        if not is_robust:
            I_AB_val, _, _ = calculate_I_AB_components_torch(current_phi_params_for_calc, M_val, x_for_opt_torch, q_priors_torch)
            loss = -I_AB_val
            I_AB_epoch = I_AB_val.item()
        else:
            phi_full_torch_current = torch.cat((torch.tensor([0.0],dtype=torch.float64,device=target_device_obj), current_phi_params_for_calc))
            phi_np_for_scipy = phi_full_torch_current.cpu().detach().numpy() 
            
            min_I_AB_inner, x_tilde_star_np = solve_inner_adversarial_problem_scipy(
                phi_np_for_scipy, M_val, x_center_for_robust_np, 
                L2_ball_radius_for_robust, q_priors_torch.cpu().numpy(), str(target_device_obj)
            )
            x_tilde_star_torch = torch.tensor(x_tilde_star_np, dtype=torch.float64, device=target_device_obj)
            I_AB_val_at_x_tilde_star, _, _ = calculate_I_AB_components_torch(current_phi_params_for_calc, M_val, x_tilde_star_torch, q_priors_torch)
            loss = -I_AB_val_at_x_tilde_star
            I_AB_epoch = I_AB_val_at_x_tilde_star.item() 
            
        if num_phases_to_opt > 0 and loss.requires_grad and optimizer:
            loss.backward()
            optimizer.step()
            
        history_I_AB_during_opt.append(I_AB_epoch)
            
    final_phi_list_params = []
    if num_phases_to_opt > 0:
        final_phi_list_params = phi_params.cpu().detach().numpy().tolist()
        
    return np.array([0.0] + final_phi_list_params), history_I_AB_during_opt

# --- Main Experiment Function ---
def main(args):
    print(f"--- Running SNN-based Discrimination for M={args.num_distinct_states_M} States ---")
    print(f"PDE Type: {args.pde_type}")
    print(f"SNN Model Path: {args.snn_model_path}")
    if args.pde_type == "step_index_fiber":
        print(f"  Fiber Params: L_domain={args.L_domain}, CoreFactor={args.fiber_core_radius_factor}, Depth={args.fiber_potential_depth}, T_evo={args.evolution_time_T}")
    
    # --- Construct Calibration Data File Path ---
    # Filename suffix for calibration data (based on how calibration.py saves it)
    calib_filename_suffix = ""
    if args.pde_type == "poisson":
        calib_filename_suffix = f"poisson_grfA{args.grf_alpha:.1f}T{args.grf_tau:.1f}OffS{args.grf_offset_sigma:.1f}"
    elif args.pde_type == "step_index_fiber":
        calib_filename_suffix = (f"fiber_GRFinA{args.grf_alpha:.1f}T{args.grf_tau:.1f}_"
                                 f"coreR{args.fiber_core_radius_factor:.1f}_V{args.fiber_potential_depth:.1f}_"
                                 f"evoT{args.evolution_time_T:.1e}_steps{args.solver_num_steps}")
    
    # Subdirectory name for calibration results
    calib_results_subdir_name = (f"PDE{args.pde_type}_NinDS{args.n_grid_sim_input_ds}_SNNres{args.snn_output_res}_" # SNNres changed to SNNres
                                 f"KfullThm{args.n_grid_sim_input_ds}_s{args.theorem_s}_nu{args.theorem_nu}_{calib_filename_suffix}")
    
    # Actual .npz filename from calibration.py
    coverage_data_filename_npz = os.path.join(
        args.calibration_results_base_dir, 
        calib_results_subdir_name,
        f"coverage_data_PDE{args.pde_type}_thm_s{args.theorem_s}_nu{args.theorem_nu}_d{args.theorem_d}"
        f"_Nin{args.n_grid_sim_input_ds}_SNNout{args.snn_output_res}_NfullThm{args.n_grid_sim_input_ds}" # No KB0factor in filename itself
        f"_{calib_filename_suffix}.npz"
    )
    print(f"Attempting to load calibration data from: {coverage_data_filename_npz}")

    L2_ball_radius_calculated = 0.1 # Default
    try:
        calib_data = np.load(coverage_data_filename_npz)
        # Assuming 'nominal_coverages' are 1-alpha and 'avg_R_bounds_for_alpha' is q_star for that 1-alpha
        # We need to find the index corresponding to args.alpha_for_radius
        
        # Option 1: If 'alpha_values_for_quantiles' is saved (ideal)
        if 'alpha_values_for_quantiles' in calib_data:
            alpha_values_from_calib = calib_data['alpha_values_for_quantiles']
            closest_alpha_idx = np.argmin(np.abs(alpha_values_from_calib - args.alpha_for_radius))
            if np.abs(alpha_values_from_calib[closest_alpha_idx] - args.alpha_for_radius) > 1e-3:
                print(f"Warning: Specified alpha_for_radius {args.alpha_for_radius} not found exactly. Using closest: {alpha_values_from_calib[closest_alpha_idx]:.3f}")
            actual_alpha_used = alpha_values_from_calib[closest_alpha_idx]
        # Option 2: Derive from 'nominal_coverages' (1-alpha)
        elif 'nominal_coverages' in calib_data:
            calib_1_minus_alphas = calib_data['nominal_coverages']
            target_1_minus_alpha = 1.0 - args.alpha_for_radius
            closest_alpha_idx = np.argmin(np.abs(calib_1_minus_alphas - target_1_minus_alpha))
            actual_alpha_used = 1.0 - calib_1_minus_alphas[closest_alpha_idx]
            if np.abs(actual_alpha_used - args.alpha_for_radius) > 1e-3:
                print(f"Warning: Specified alpha_for_radius {args.alpha_for_radius} (1-alpha={target_1_minus_alpha:.3f}) "
                      f"not found exactly. Using closest: {actual_alpha_used:.3f} (1-alpha={calib_1_minus_alphas[closest_alpha_idx]:.3f})")
        else:
            raise KeyError("Neither 'alpha_values_for_quantiles' nor 'nominal_coverages' found in calibration data.")

        q_star_val = calib_data['avg_R_bounds_for_alpha'][closest_alpha_idx] 
        
        if q_star_val < 0:
            print(f"Warning: q_star_val from calibration data is negative ({q_star_val:.4e}). Using absolute value for sqrt.")
            q_star_val_for_sqrt = abs(q_star_val)
        else:
            q_star_val_for_sqrt = q_star_val

        L2_ball_radius_calculated_sq = (args.num_distinct_states_M**2) * (2 * np.sqrt(q_star_val_for_sqrt) + q_star_val)
        L2_ball_radius_calculated = np.sqrt(L2_ball_radius_calculated_sq)
        L2_ball_radius_calculated = 0.1
        print(f"Loaded calibration data. For target alpha~{args.alpha_for_radius:.3f} (used {actual_alpha_used:.3f}), found q*={q_star_val:.4e}.")
        print(f"Calculated L2 Uncertainty Ball Radius for Robust Opt: {L2_ball_radius_calculated:.4e}")

    except FileNotFoundError:
        print(f"ERROR: Calibration data file not found at {coverage_data_filename_npz}. Using default radius {L2_ball_radius_calculated}.")
    except KeyError as e:
        print(f"ERROR: Key {e} not found in calibration data file {coverage_data_filename_npz}. Using default radius {L2_ball_radius_calculated}.")
    except Exception as e:
        print(f"ERROR loading or processing calibration data: {e}. Using default radius {L2_ball_radius_calculated}.")

    print(f"Priors: {args.priors_q_j}")
    print(f"Number of trials for this config: {args.num_trials_per_config}")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(args.snn_model_path):
        print(f"ERROR: SNN model file not found at {args.snn_model_path}.")
        return None
        
    snn_model_state_dict = torch.load(args.snn_model_path, map_location=DEVICE)
    snn = SimpleSpectralOperatorCNN(K_input_resolution=args.n_grid_sim_input_ds, 
                                    K_output_resolution=args.snn_output_res, 
                                    hidden_channels=args.snn_hidden_channels,
                                    num_hidden_layers=args.snn_num_hidden_layers)
    snn.load_state_dict(snn_model_state_dict)
    snn.to(DEVICE)
    snn.eval()
    print("SNN model loaded successfully.")

    pgm_IAB_on_true_trials = []
    nominal_IAB_on_true_trials = []
    robust_IAB_on_true_trials = []
    
    delta_n_vector_gus_np = np.array(args.delta_n_vector_gus_components)

    for i_trial in range(args.num_trials_per_config):
        if (i_trial + 1) % (args.num_trials_per_config // 10 or 1) == 0 or args.num_trials_per_config==1 or i_trial==0:
            print(f"  Starting Trial {i_trial + 1}/{args.num_trials_per_config}...")
        
        initial_states_psi0_real = generate_initial_GUS_states_via_phase_ramp(
            args.num_distinct_states_M, args.n_grid_sim_input_ds, args.L_domain, 
            args.grf_alpha, args.grf_tau, 
            delta_n_vector_gus_np
        )
        
        list_gamma_b_k_full_input_spec = []
        for psi0_real_k in initial_states_psi0_real:
            spec_unnorm = get_full_centered_spectrum(psi0_real_k)
            list_gamma_b_k_full_input_spec.append(normalize_spectrum(spec_unnorm))

        list_gamma_a_k_true_Nout_spec = []
        if args.pde_type == "step_index_fiber":
            potential_V = get_step_index_fiber_potential(args.n_grid_sim_input_ds, 
                                                         args.L_domain, 
                                                         args.fiber_core_radius_factor, 
                                                         args.fiber_potential_depth)
            for psi0_real_k_norm in initial_states_psi0_real: 
                psi_T_real = solver_main(potential_V, psi0_real_k_norm, 
                                     N_grid=args.n_grid_sim_input_ds, L_domain=args.L_domain, 
                                     T_evolution=args.evolution_time_T, num_solver_steps=args.solver_num_steps,
                                     hbar=args.hbar_val, m=args.mass_val)
                gamma_a_true_full_spec = normalize_spectrum(get_full_centered_spectrum(psi_T_real))
                gamma_a_true_Nout_spec = normalize_spectrum(extract_center_block_np(gamma_a_true_full_spec, args.snn_output_res))
                list_gamma_a_k_true_Nout_spec.append(gamma_a_true_Nout_spec)
        elif args.pde_type == "poisson": 
            print("Warning: Poisson PDE for robust opt not fully fleshed out here, using placeholder for true output.")
            list_gamma_a_k_true_Nout_spec = [normalize_spectrum(extract_center_block_np(s, args.snn_output_res)) for s in list_gamma_b_k_full_input_spec] 
        else:
            raise ValueError(f"PDE type {args.pde_type} not supported for true evolution in robust opt script.")

        G_true = np.zeros((args.num_distinct_states_M, args.num_distinct_states_M), dtype=np.complex128)
        for r_idx in range(args.num_distinct_states_M):
            for c_idx in range(args.num_distinct_states_M):
                G_true[r_idx, c_idx] = np.vdot(list_gamma_a_k_true_Nout_spec[r_idx].ravel(), list_gamma_a_k_true_Nout_spec[c_idx].ravel())
        
        g_j_true_vals = np.linalg.eigvalsh(G_true)
        g_j_true_vals = np.maximum(1e-7, g_j_true_vals) 
        sqrt_g_j_true_np = np.sqrt(np.sort(g_j_true_vals)[::-1]) 
        x_true_torch = torch.tensor(sqrt_g_j_true_np, dtype=torch.float64).to(DEVICE)
        q_priors_torch = torch.tensor(args.priors_q_j, dtype=torch.float64).to(DEVICE)

        list_gamma_a_k_snn_pred_Nout_spec_normalized = []
        with torch.no_grad():
            for gamma_b_spec_full_input in list_gamma_b_k_full_input_spec:
                gamma_b_channels = spectrum_complex_to_channels_torch(gamma_b_spec_full_input).unsqueeze(0).to(DEVICE) 
                gamma_a_pred_channels_Nout = snn(gamma_b_channels) 
                gamma_a_pred_complex_Nout = channels_to_spectrum_complex_torch(gamma_a_pred_channels_Nout.squeeze(0).cpu())
                list_gamma_a_k_snn_pred_Nout_spec_normalized.append(normalize_spectrum(gamma_a_pred_complex_Nout.numpy()))
        
        G_est = np.zeros((args.num_distinct_states_M, args.num_distinct_states_M), dtype=np.complex128)
        for r_idx in range(args.num_distinct_states_M):
            for c_idx in range(args.num_distinct_states_M):
                G_est[r_idx,c_idx] = np.vdot(list_gamma_a_k_snn_pred_Nout_spec_normalized[r_idx].ravel(), list_gamma_a_k_snn_pred_Nout_spec_normalized[c_idx].ravel())
        
        g_j_estimated_vals = np.linalg.eigvalsh(G_est)
        g_j_estimated_vals = np.maximum(1e-7, g_j_estimated_vals)
        sqrt_g_j_estimated_np = np.sqrt(np.sort(g_j_estimated_vals)[::-1])
        x_estimated_torch = torch.tensor(sqrt_g_j_estimated_np, dtype=torch.float64).to(DEVICE)

        phi_pgm_np = np.zeros(args.num_distinct_states_M) 
        I_AB_pgm_on_true = calculate_I_AB_numpy(phi_pgm_np, args.num_distinct_states_M, sqrt_g_j_true_np, args.priors_q_j, target_device_str=str(DEVICE))
        pgm_IAB_on_true_trials.append(I_AB_pgm_on_true) 

        phi_nom_opt_np, _ = optimize_phases_pytorch(args.num_distinct_states_M, x_estimated_torch, q_priors_torch, target_device_obj=DEVICE, is_robust=False, num_epochs=args.max_pytorch_opt_epochs, lr=args.pytorch_lr)
        I_AB_phi_nom_on_true = calculate_I_AB_numpy(phi_nom_opt_np, args.num_distinct_states_M, sqrt_g_j_true_np, args.priors_q_j, target_device_str=str(DEVICE))
        nominal_IAB_on_true_trials.append(I_AB_phi_nom_on_true)

        phi_rob_opt_np, _ = optimize_phases_pytorch(args.num_distinct_states_M, x_true_torch, q_priors_torch, target_device_obj=DEVICE, is_robust=True, x_center_for_robust_np=sqrt_g_j_estimated_np, L2_ball_radius_for_robust=L2_ball_radius_calculated, num_epochs=args.max_pytorch_opt_epochs, lr=args.pytorch_lr) 
        I_AB_phi_rob_on_true = calculate_I_AB_numpy(phi_rob_opt_np, args.num_distinct_states_M, sqrt_g_j_true_np, args.priors_q_j, target_device_str=str(DEVICE))
        robust_IAB_on_true_trials.append(I_AB_phi_rob_on_true)
        
    avg_pgm_IAB = np.mean(pgm_IAB_on_true_trials)
    avg_nominal_IAB = np.mean(nominal_IAB_on_true_trials)
    avg_robust_IAB = np.mean(robust_IAB_on_true_trials)
    
    t_rob_nom, p_rob_nom = -1.0, 1.0 
    t_nom_pgm, p_nom_pgm = -1.0, 1.0
    t_rob_pgm, p_rob_pgm = -1.0, 1.0

    if args.num_trials_per_config > 1:
        nom_v = np.array(nominal_IAB_on_true_trials)
        rob_v = np.array(robust_IAB_on_true_trials)
        pgm_v = np.array(pgm_IAB_on_true_trials)
        
        mask = ~np.isnan(nom_v) & ~np.isnan(rob_v) & ~np.isnan(pgm_v)
        nom_v, rob_v, pgm_v = nom_v[mask], rob_v[mask], pgm_v[mask]
        
        if len(rob_v) > 1: 
            if not np.allclose(rob_v, nom_v):
                t_rob_nom, p_rob_nom = stats.ttest_rel(rob_v, nom_v, alternative='greater')
            else:
                t_rob_nom, p_rob_nom = 0.0, 0.5
            
            if not np.allclose(nom_v, pgm_v):
                t_nom_pgm, p_nom_pgm = stats.ttest_rel(nom_v, pgm_v, alternative='greater')
            else:
                t_nom_pgm, p_nom_pgm = 0.0, 0.5

            if not np.allclose(rob_v, pgm_v):
                t_rob_pgm, p_rob_pgm = stats.ttest_rel(rob_v, pgm_v, alternative='greater')
            else:
                t_rob_pgm, p_rob_pgm = 0.0, 0.5
    
    results = {
        "params": vars(args).copy(), 
        "avg_pgm_IAB": avg_pgm_IAB, 
        "avg_nominal_IAB": avg_nominal_IAB, 
        "avg_robust_IAB": avg_robust_IAB, 
        "t_statistic_rob_gt_nom": float(t_rob_nom), 
        "p_value_rob_gt_nom": float(p_rob_nom),
        "t_statistic_nom_gt_pgm": float(t_nom_pgm), 
        "p_value_nom_gt_pgm": float(p_nom_pgm),
        "t_statistic_rob_gt_pgm": float(t_rob_pgm), 
        "p_value_rob_gt_pgm": float(p_rob_pgm),
        "L2_ball_radius_used_for_robust_opt": L2_ball_radius_calculated 
    }
    
    if 'delta_n_vector_gus_components' in results['params']: 
        results['params']['delta_n_vector_gus'] = list(results['params']['delta_n_vector_gus_components'])
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    filename_suffix_run = ""
    if args.pde_type == "poisson":
        filename_suffix_run = f"poisson_grfA{args.grf_alpha:.1f}T{args.grf_tau:.1f}OffS{args.grf_offset_sigma:.1f}"
    elif args.pde_type == "step_index_fiber":
        filename_suffix_run = (f"fiber_GRFinA{args.grf_alpha:.1f}T{args.grf_tau:.1f}_"
                               f"coreR{args.fiber_core_radius_factor:.1f}_V{args.fiber_potential_depth:.1f}_"
                               f"evoT{args.evolution_time_T:.1e}_steps{args.solver_num_steps}")

    output_filename = f"results_M{args.num_distinct_states_M}_alphaCalib{args.alpha_for_radius:.2f}_{filename_suffix_run}.json" 
    if args.output_json_filename_tag:
        output_filename = f"{args.output_json_filename_tag}_{output_filename}"
        
    output_filepath = os.path.join(args.results_dir, output_filename)
    try:
        serializable_params = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k,v in results["params"].items()}
        results_to_save = {key: val for key, val in results.items() if key != "params"}
        results_to_save["params_used_for_run"] = serializable_params

        with open(output_filepath, 'w') as f:
            json.dump(results_to_save, f, indent=4)
        print(f"\nResults for this run saved to: {output_filepath}")
    except Exception as e:
        print(f"Error saving results to {output_filepath}: {e}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run SNN-based state discrimination with robust POVM optimization using calibrated radius.")
    
    # --- PDE Type ---
    parser.add_argument('--pde_type', type=str, default="step_index_fiber", 
                        choices=["poisson", "step_index_fiber"],
                        help="Type of PDE evolution for the 'true' channel.")
    
    # --- SNN Model Parameters ---
    parser.add_argument('--snn_model_dir', type=str, default="trained_snn_models_sweep_final_v3", 
                        help='Directory containing SNN models.')
    parser.add_argument('--n_grid_sim_input_ds', type=int, default=64, 
                        help='SNN input resolution (Nin of dataset).')
    parser.add_argument('--snn_output_res', type=int, default=32, 
                        help='SNN output resolution (Nout of dataset target).')
    parser.add_argument('--snn_hidden_channels', type=int, default=64)
    parser.add_argument('--snn_num_hidden_layers', type=int, default=3)
    
    # --- Initial State Generation Parameters (GRF for base state) ---
    parser.add_argument('--grf_alpha', type=float, default=4.0) 
    parser.add_argument('--grf_tau', type=float, default=1.0)   
    parser.add_argument('--grf_offset_sigma', type=float, default=0.5, 
                        help="Sigma for hierarchical offset in Poisson source (f term).") 
    parser.add_argument('--k_psi0_limit_dataset', type=int, default=12, 
                        help="Not directly used if GRF is source for initial state, but kept for data_gen script compatibility.")

    # --- Step-Index Fiber Evolution Parameters (if pde_type is step_index_fiber) ---
    parser.add_argument('--L_domain', type=float, default=2*np.pi)
    parser.add_argument('--fiber_core_radius_factor', type=float, default=0.2)
    parser.add_argument('--fiber_potential_depth', type=float, default=1.0) 
    parser.add_argument('--evolution_time_T', type=float, default=0.1) 
    parser.add_argument('--solver_num_steps', type=int, default=50) 
    parser.add_argument('--hbar_val', type=float, default=HBAR_CONST) 
    parser.add_argument('--mass_val', type=float, default=MASS_CONST) 

    # --- Discrimination and Optimization Parameters ---
    parser.add_argument('--num_distinct_states_M', type=int, default=3)
    parser.add_argument('--delta_n_vector_gus_components', nargs='+', type=int, default=[1, 0], 
                        help='Components of delta_n for GUS phase ramp.')
    parser.add_argument('--alpha_for_radius', type=float, default=0.1,
                        help="Alpha value from calibration to determine the robust radius.")
    # --calibration_data_file removed, will be constructed
    parser.add_argument('--calibration_results_base_dir', type=str, required=True,
                        help="Base directory where calibration result subdirectories are stored.")
    parser.add_argument('--theorem_s', type=float, default=2.0, help="Theorem s parameter (for calib filename).")
    parser.add_argument('--theorem_nu', type=float, default=2.0, help="Theorem nu parameter (for calib filename).")
    parser.add_argument('--theorem_d', type=int, default=2, help="Theorem d parameter (for calib filename).")
    parser.add_argument('--k_trunc_bound_b0_factor', type=int, default=0, 
                        help="K_trunc for B0 factor in calib file (for calib filename). If 0, uses N_full.")
                        
    parser.add_argument('--num_trials_per_config', type=int, default=30) 
    parser.add_argument('--max_pytorch_opt_epochs', type=int, default=300)
    parser.add_argument('--pytorch_lr', type=float, default=0.005)
    parser.add_argument('--priors_q_j', nargs='+', type=float, default=None) 
    
    # --- Output Control ---
    parser.add_argument('--results_dir', type=str, default="results_robust_opt_calibrated_radius")
    parser.add_argument('--output_json_filename_tag', type=str, default="", 
                        help="Optional tag to prepend to output JSON filename.")
    
    args = parser.parse_args()

    if args.snn_output_res > args.n_grid_sim_input_ds:
        raise ValueError("SNN output resolution (--snn_output_res) cannot be greater than SNN input resolution (--n_grid_sim_input_ds).")

    if args.priors_q_j is None:
        if args.num_distinct_states_M == 3:
            args.priors_q_j = [0.7, 0.15, 0.15]
        elif args.num_distinct_states_M == 4:
            args.priors_q_j = [0.6, .2, .1, .1]
        elif args.num_distinct_states_M == 5:
            args.priors_q_j = [0.4,0.25,0.15,0.1,0.1]
        else:
            args.priors_q_j = [1.0/args.num_distinct_states_M]*args.num_distinct_states_M
    args.priors_q_j = (np.array(args.priors_q_j) / np.sum(args.priors_q_j)).tolist() 

    # --- Construct SNN Model Path ---
    snn_filename_suffix = ""
    if args.pde_type == "poisson":
        snn_filename_suffix = f"poisson_grfA{args.grf_alpha:.1f}T{args.grf_tau:.1f}OffS{args.grf_offset_sigma:.1f}"
    elif args.pde_type == "step_index_fiber":
        snn_filename_suffix = (f"fiber_GRFinA{args.grf_alpha:.1f}T{args.grf_tau:.1f}_"
                               f"coreR{args.fiber_core_radius_factor:.1f}_V{args.fiber_potential_depth:.1f}_"
                               f"evoT{args.evolution_time_T:.1e}_steps{args.solver_num_steps}")
    
    args.snn_model_path = os.path.join(
        args.snn_model_dir,
        f"snn_PDE{args.pde_type}_Kin{args.n_grid_sim_input_ds}_Kout{args.snn_output_res}_H{args.snn_hidden_channels}_L{args.snn_num_hidden_layers}_{snn_filename_suffix}.pth"
    )

    main(args)
