import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import argparse 

# --- SNN Model Definition (should match the one used for training) ---
class SimpleSpectralOperatorCNN(nn.Module):
    def __init__(self, K_input_resolution, K_output_resolution, hidden_channels=64, num_hidden_layers=3):
        super().__init__()
        self.K_input_resolution = K_input_resolution
        self.K_output_resolution = K_output_resolution
        
        if K_output_resolution > K_input_resolution:
            raise ValueError("K_output_resolution cannot be greater than K_input_resolution for this SNN design (cropping output).")
        
        layers = []
        # CNN body operates at K_input_resolution
        layers.append(nn.Conv2d(2, hidden_channels, kernel_size=3, padding='same'))
        layers.append(nn.ReLU())
        
        for _ in range(num_hidden_layers):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding='same'))
            layers.append(nn.ReLU())
            
        layers.append(nn.Conv2d(hidden_channels, 2, kernel_size=3, padding='same'))
        self.cnn_body = nn.Sequential(*layers)

    def forward(self, x_spec_ch_full_input): # x_spec_ch_full_input: (batch, 2, K_input, K_input)
        # CNN body processes at full input resolution
        x_processed_full = self.cnn_body(x_spec_ch_full_input) # Output: (batch, 2, K_input, K_input)
        
        # Truncate/crop the output to K_output_resolution x K_output_resolution
        if self.K_input_resolution == self.K_output_resolution:
            return x_processed_full
        else: 
            # K_input_resolution > K_output_resolution
            start_idx = self.K_input_resolution // 2 - self.K_output_resolution // 2
            end_idx = start_idx + self.K_output_resolution
            return x_processed_full[:, :, start_idx:end_idx, start_idx:end_idx]

# --- Data Handling ---
def spectrum_complex_to_channels_torch(spectrum_mat_complex):
    if not isinstance(spectrum_mat_complex, torch.Tensor):
        spectrum_mat_complex = torch.from_numpy(spectrum_mat_complex)
    
    if not torch.is_complex(spectrum_mat_complex): 
        if spectrum_mat_complex.ndim == 3 and spectrum_mat_complex.shape[0] == 2: # Already [2,K,K] real tensor
            return spectrum_mat_complex.float() 
        if spectrum_mat_complex.ndim == 2 and not torch.is_complex(spectrum_mat_complex): # Real [K,K] tensor
             raise ValueError(f"Input spectrum_mat_complex is real [K,K] but expected complex or [2,K,K] real.")
        # Other unexpected real tensor shapes
        raise ValueError(f"Input spectrum_mat_complex has shape {spectrum_mat_complex.shape}. Expected complex [K,K] or real [2,K,K].")
        
    return torch.stack([spectrum_mat_complex.real, spectrum_mat_complex.imag], dim=0).float()

def channels_to_spectrum_complex_torch(channels_mat_real_imag):
    if channels_mat_real_imag.ndim != 3 or channels_mat_real_imag.shape[0] != 2: 
        raise ValueError(f"Input must have 2 channels as the first dimension, got shape {channels_mat_real_imag.shape}")
    return torch.complex(channels_mat_real_imag[0], channels_mat_real_imag[1])

def get_mode_indices_and_weights(K_grid_size, d_dimensions, s_coeff, nu_coeff):
    """
    Generates mode indices (nx, ny for d=2) for a K_grid_size x K_grid_size
    and their corresponding Sobolev weights (1 + ||n||_2^2)^(s_coeff - nu_coeff).
    Assumes centered spectrum for K_grid_size.
    """
    if K_grid_size == 0: 
        return [], np.array([]) 
        
    if K_grid_size % 2 == 0: 
        k_vals_one_dim_eff = np.arange(-K_grid_size//2, K_grid_size//2)
    else: 
        k_vals_one_dim_eff = np.arange(-(K_grid_size-1)//2, (K_grid_size-1)//2 + 1)
    
    if d_dimensions == 1:
        n_grids_list = [k_vals_one_dim_eff.reshape(-1,1)]
    elif d_dimensions == 2:
        nx_eff_grid, ny_eff_grid = np.meshgrid(k_vals_one_dim_eff, k_vals_one_dim_eff, indexing='ij')
        n_grids_list = [nx_eff_grid, ny_eff_grid]
    else:
        raise ValueError("d_dimensions must be 1 or 2 for this implementation.")

    norm_n_L2_sq = np.zeros_like(n_grids_list[0], dtype=float)
    for i in range(d_dimensions):
        norm_n_L2_sq += n_grids_list[i]**2 
    
    term_inside_power = 1 + norm_n_L2_sq # Standard Sobolev base (1 + ||n||_2^2)
    exponent = s_coeff - nu_coeff 
    
    if np.isclose(exponent, 0.0): 
        sobolev_weights = np.ones_like(term_inside_power)
    else:
        sobolev_weights = (term_inside_power + 1e-9)**(exponent) 
    
    if np.any(np.isnan(sobolev_weights)) or np.any(np.isinf(sobolev_weights)):
        print(f"Warning: NaN or Inf encountered in Sobolev weights. Exponent: {exponent}. Min term_inside_power: {np.min(term_inside_power)}")
        sobolev_weights = np.nan_to_num(sobolev_weights, nan=1.0, posinf=1e9, neginf=1e-9) 
        if np.isclose(exponent, 0.0):
            sobolev_weights = np.ones_like(term_inside_power)

    return n_grids_list, sobolev_weights

def extract_center_block_np(full_spectrum_centered_np, K_extract):
    """ Extracts central K_extract x K_extract block from a NumPy array. DOES NOT NORMALIZE. """
    N_full = full_spectrum_centered_np.shape[0]

    if K_extract == N_full:
        return full_spectrum_centered_np
        
    if K_extract > N_full: 
        padded_block = np.zeros((K_extract, K_extract), dtype=full_spectrum_centered_np.dtype)
        start_pad_x = K_extract // 2 - N_full // 2
        end_pad_x = start_pad_x + N_full
        start_pad_y = K_extract // 2 - N_full // 2
        end_pad_y = start_pad_y + N_full
        
        valid_src_start_x = max(0, -start_pad_x)
        valid_src_end_x = N_full - max(0, end_pad_x - K_extract)
        valid_dst_start_x = max(0, start_pad_x)
        valid_dst_end_x = K_extract - max(0, K_extract - end_pad_x)

        valid_src_start_y = max(0, -start_pad_y)
        valid_src_end_y = N_full - max(0, end_pad_y - K_extract)
        valid_dst_start_y = max(0, start_pad_y)
        valid_dst_end_y = K_extract - max(0, K_extract - end_pad_y)

        if valid_src_end_x > valid_src_start_x and \
           valid_src_end_y > valid_src_start_y and \
           valid_dst_end_x > valid_dst_start_x and \
           valid_dst_end_y > valid_dst_start_y:
             padded_block[valid_dst_start_x:valid_dst_end_x, valid_dst_start_y:valid_dst_end_y] = \
                 full_spectrum_centered_np[valid_src_start_x:valid_src_end_x, valid_src_start_y:valid_src_end_y]
        return padded_block
        
    if K_extract <= 0:
        return np.array([], dtype=np.complex64)
    
    start_idx = N_full // 2 - K_extract // 2
    end_idx = start_idx + K_extract
    return full_spectrum_centered_np[start_idx:end_idx, start_idx:end_idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Conformal Prediction for SNN Error (Theorem Validation).")
    
    # --- PDE Type and Dataset Parameters ---
    parser.add_argument('--pde_type', type=str, default="step_index_fiber", 
                        choices=["poisson", "step_index_fiber"], 
                        help="Type of data generation process the dataset corresponds to.")
    parser.add_argument('--n_grid_sim_input_ds', type=int, default=64, 
                        help='Nin: Resolution of gamma_b_full_input and gamma_a_true_full_output in dataset. SNN input resolution.')
    parser.add_argument('--snn_output_res', type=int, default=32, 
                        help='Nout: Resolution of gamma_a_snn_target in dataset AND SNN output resolution.')
    parser.add_argument('--dataset_dir', type=str, default="datasets", 
                        help="Directory where datasets are stored.")
    
    # --- SNN Architecture Parameters ---
    parser.add_argument('--snn_model_dir', type=str, default="trained_snn_models",
                        help="Directory containing trained SNN models.")
    parser.add_argument('--snn_hidden_channels', type=int, default=64,
                        help="Hidden channels in SNN.")
    parser.add_argument('--snn_num_hidden_layers', type=int, default=3,
                        help="Number of hidden layers in SNN.")
    parser.add_argument('--snn_model_filename_override', type=str, default=None,
                        help="Exact SNN model filename to load, bypassing construction from other args.")

    # --- Theorem and Weighting Parameters ---
    parser.add_argument('--s_theorem', type=float, default=2.0, help="Theorem parameter 's'.")
    parser.add_argument('--nu_theorem', type=float, default=2.0, help="Theorem parameter 'nu'.") 
    parser.add_argument('--d_dimensions', type=int, default=2, choices=[1,2], 
                        help="Spatial dimension d for Sobolev weights.")
    parser.add_argument('--k_trunc_bound', type=int, default=48, 
                        help="K_grid_size for get_mode_indices_and_weights, used for default B_sq bound.")
    parser.add_argument('--elliptic_PDE_const_C_sq', type=float, default=4.0,
                        help="Constant C^2 for Poisson elliptic PDE bound (e.g., 4 for ||u||_Hs <= C||f||_{Hs-2}).")

    # --- Conformal Prediction Parameters ---
    parser.add_argument('--calib_split_ratio', type=float, default=0.5,
                        help="Ratio of data for calibration set.")
    parser.add_argument('--random_seed', type=int, default=42,
                        help="Random seed for data splitting.")
    
    # --- Source/Evolution Parameters (for constructing correct filenames and for step_index_fiber B_sq) ---
    # GRF parameters (used for Poisson source f, OR for step_index_fiber initial state)
    parser.add_argument('--grf_alpha', type=float, default=4.0) 
    parser.add_argument('--grf_tau', type=float, default=1.0)   
    parser.add_argument('--grf_offset_sigma', type=float, default=0.5,
                        help="Sigma for hierarchical offset in Poisson source (f term).")
    # Step-Index Fiber Parameters
    parser.add_argument('--L_domain', type=float, default=2*np.pi) # Used by data_gen for step_index
    parser.add_argument('--fiber_core_radius_factor', type=float, default=0.2)
    parser.add_argument('--fiber_potential_depth', type=float, default=10.0) # This is V0 for the bound
    parser.add_argument('--evolution_time_T', type=float, default=0.1) 
    parser.add_argument('--solver_num_steps', type=int, default=50) 
    
    # --- Output Control ---
    parser.add_argument('--results_dir', type=str, default="results_conformal_theorem_validation", 
                        help="Directory to save output plots and data for this run.")
    parser.add_argument('--no_plot', action='store_true', help="Suppress displaying the plot.")

    args = parser.parse_args()

    snn_input_res_val = args.n_grid_sim_input_ds
    snn_output_res_val = args.snn_output_res 
    N_full_for_theorem = args.n_grid_sim_input_ds 

    print(f"SNN Input Resolution (N_in from dataset): {snn_input_res_val}")
    print(f"SNN Output Resolution (N_out from dataset): {snn_output_res_val}")
    print(f"N_full for theorem evaluation: {N_full_for_theorem}")

    filename_suffix = ""
    if args.pde_type == "poisson":
        filename_suffix = f"poisson_grfA{args.grf_alpha:.1f}T{args.grf_tau:.1f}OffS{args.grf_offset_sigma:.1f}"
    elif args.pde_type == "step_index_fiber":
        filename_suffix = (f"fiber_GRFinA{args.grf_alpha:.1f}T{args.grf_tau:.1f}_"
                           f"coreR{args.fiber_core_radius_factor:.1f}_V{args.fiber_potential_depth:.1f}_"
                           f"evoT{args.evolution_time_T:.1e}_steps{args.solver_num_steps}")
    
    if args.snn_model_filename_override:
        SNN_MODEL_FILENAME = args.snn_model_filename_override
    else:
        SNN_MODEL_FILENAME = f"snn_PDE{args.pde_type}_Kin{snn_input_res_val}_Kout{snn_output_res_val}_H{args.snn_hidden_channels}_L{args.snn_num_hidden_layers}_{filename_suffix}.pth"
    SNN_MODEL_PATH = os.path.join(args.snn_model_dir, SNN_MODEL_FILENAME)
    
    DATASET_FILENAME = f"dataset_{args.pde_type}_Nin{args.n_grid_sim_input_ds}_Nout{args.snn_output_res}_{filename_suffix}.npz"
    DATASET_FILE_PATH = os.path.join(args.dataset_dir, DATASET_FILENAME)
    
    output_filename_suffix_calib = (f"_PDE{args.pde_type}_thm_s{args.s_theorem}_nu{args.nu_theorem}_d{args.d_dimensions}"
                                    f"_Nin{args.n_grid_sim_input_ds}_SNNout{snn_output_res_val}_NfullThm{N_full_for_theorem}_KboundB{args.k_trunc_bound}"
                                    f"_{filename_suffix}") 
    
    scenario_title_suffix = (f"(PDE: {args.pde_type}, Thm: $s={args.s_theorem}, \\nu={args.nu_theorem}, d={args.d_dimensions}\n"
                             f"$N_{{in}}={args.n_grid_sim_input_ds}, N_{{SNNout}}={snn_output_res_val}, N_{{fullThm}}={N_full_for_theorem}, K_{{B}}={args.k_trunc_bound}$, "
                             f"Params: {filename_suffix}$)")
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.results_dir, exist_ok=True) 
    
    print(f"--- Running Conformal Prediction: Theorem Test ---")
    print(f"Scenario: {scenario_title_suffix}")
    print(f"SNN Model Path: {SNN_MODEL_PATH}")
    print(f"Dataset Path: {DATASET_FILE_PATH}")

    B_sq_bound_default = 1.0 
    if args.k_trunc_bound > 0: 
        _, H_s_norm_weights_for_B_default = get_mode_indices_and_weights(args.k_trunc_bound, args.d_dimensions, args.s_theorem, 0) 
        if H_s_norm_weights_for_B_default.size > 0:
            B_sq_bound_default = np.max(H_s_norm_weights_for_B_default) 
    print(f"Default B0 factor (max H^s weight on K_bound={args.k_trunc_bound} grid): {B_sq_bound_default:.4e}")
    
    if not os.path.exists(SNN_MODEL_PATH):
        print(f"ERROR: SNN model not found: {SNN_MODEL_PATH}")
        exit()
    snn_model = SimpleSpectralOperatorCNN(K_input_resolution=snn_input_res_val, 
                                          K_output_resolution=snn_output_res_val, 
                                          hidden_channels=args.snn_hidden_channels, 
                                          num_hidden_layers=args.snn_num_hidden_layers)
    snn_model.load_state_dict(torch.load(SNN_MODEL_PATH, map_location=DEVICE))
    snn_model.to(DEVICE)
    snn_model.eval()
    print("SNN model loaded.")
    
    try:
        data = np.load(DATASET_FILE_PATH)
        gamma_b_full_all = data['gamma_b_full_input']         
        gamma_a_snn_target_all = data['gamma_a_snn_target'] 
        gamma_a_true_full_all = data['gamma_a_true_full_output'] 
    except Exception as e:
        print(f"Error loading dataset {DATASET_FILE_PATH}: {e}")
        exit()
    
    if not (gamma_b_full_all.ndim > 2 and gamma_b_full_all.shape[1] == args.n_grid_sim_input_ds and \
            gamma_a_snn_target_all.ndim > 2 and gamma_a_snn_target_all.shape[1] == args.snn_output_res and \
            gamma_a_true_full_all.ndim > 2 and gamma_a_true_full_all.shape[1] == args.n_grid_sim_input_ds):
        print(f"Error: Loaded dataset dimensions mismatch with script arguments or expected structure.")
        exit()
    if N_full_for_theorem < snn_output_res_val : 
        print(f"Error: N_full_for_theorem ({N_full_for_theorem}) must be >= SNN output res ({snn_output_res_val}).")
        exit()
    
    num_total_samples = gamma_b_full_all.shape[0]
    indices = np.arange(num_total_samples)
    if not (0 < args.calib_split_ratio < 1):
        args.calib_split_ratio = 0.5
    test_size_float = 1 - args.calib_split_ratio
    if int(num_total_samples * test_size_float) < 1 and num_total_samples > 1:
        test_size_float = 1.0 / num_total_samples
    if int(num_total_samples * (1-test_size_float)) < 1 :
        print("Error: Not enough samples for calibration.")
        exit()
        
    cal_indices, test_indices = train_test_split(indices, test_size=test_size_float, random_state=args.random_seed, shuffle=True)
    gamma_b_full_cal = gamma_b_full_all[cal_indices]
    gamma_a_snn_target_cal = gamma_a_snn_target_all[cal_indices] 
    gamma_b_full_test = gamma_b_full_all[test_indices]
    gamma_a_true_full_test = gamma_a_true_full_all[test_indices] 
    
    print(f"Cal set: {len(gamma_b_full_cal)}, Test set: {len(gamma_b_full_test)}")
    if not (len(gamma_b_full_cal) > 0 and len(gamma_b_full_test) > 0):
        print("Error: Cal or test set empty.")
        exit()

    nonconformity_scores_cal = [] 
    print("\nCalculating nonconformity scores (L2 norm on SNN output resolution)...")
    with torch.no_grad():
        for i in range(len(gamma_b_full_cal)):
            gb_full_complex = gamma_b_full_cal[i]       
            ga_snn_target_complex = gamma_a_snn_target_cal[i] 
            
            gb_cal_channels = spectrum_complex_to_channels_torch(gb_full_complex).unsqueeze(0).to(DEVICE)
            ga_cal_pred_snn_output_channels = snn_model(gb_cal_channels) 
            ga_cal_pred_snn_output_complex = channels_to_spectrum_complex_torch(ga_cal_pred_snn_output_channels.squeeze(0).cpu()).numpy()
            
            score = np.sum(np.abs(ga_cal_pred_snn_output_complex - ga_snn_target_complex)**2) 
            nonconformity_scores_cal.append(score) 
    nonconformity_scores_cal = np.array(nonconformity_scores_cal)
    print(f"Calculated {len(nonconformity_scores_cal)} calibration scores. Avg: {np.mean(nonconformity_scores_cal):.4e}")

    quantiles_q_hat_nu = []
    alpha_values_for_quantiles = np.round(np.arange(0.05, 1.0, 0.05), 2)
    nominal_coverages_1_minus_alpha = 1 - alpha_values_for_quantiles
    n_cal = len(nonconformity_scores_cal)
    for alpha_q in alpha_values_for_quantiles:
        quantile_idx = min(max(0, int(np.ceil((n_cal + 1) * (1 - alpha_q))) -1 ), n_cal -1) 
        q_hat = np.sort(nonconformity_scores_cal)[quantile_idx]
        quantiles_q_hat_nu.append(q_hat)
    
    empirical_coverages_theorem = []
    avg_R_bounds_for_alpha_if_sample_dependent = []
    _, sobolev_weights_LHS_sum_Nfull = get_mode_indices_and_weights(N_full_for_theorem, args.d_dimensions, args.s_theorem, args.nu_theorem)
    print(f"\nUsing Sobolev weights for THEOREM LHS sum (N_full={N_full_for_theorem} grid) with exponent (s-nu) = {args.s_theorem - args.nu_theorem:.2f}")
    
    # Pre-calculate weights for H^{s-2} (for Poisson f) and H^s (for Fiber gamma_b) if needed for sample-dependent B
    weights_f_Hsm2_Nfull = None
    weights_gamma_b_Hs_Nfull = None
    if args.pde_type == "poisson":
        _, weights_f_Hsm2_Nfull = get_mode_indices_and_weights(N_full_for_theorem, args.d_dimensions, args.s_theorem - 2, 0)
    if args.pde_type == "step_index_fiber":
        _, weights_gamma_b_Hs_Nfull = get_mode_indices_and_weights(N_full_for_theorem, args.d_dimensions, args.s_theorem, 0)


    print("\nCalculating empirical coverage on test set using theorem's bound...")
    with torch.no_grad():
        for q_idx, q_hat_nu_val in enumerate(quantiles_q_hat_nu):
            covered_count_theorem = 0
            current_alpha_R_bounds = [] 
            for i in range(len(gamma_b_full_test)):
                gb_full_test_complex = gamma_b_full_test[i] # This is f_full_spec for Poisson, or gamma_b_full_spec for Fiber
                ga_true_full_output_complex = gamma_a_true_full_test[i] 
                
                gb_test_channels = spectrum_complex_to_channels_torch(gb_full_test_complex).unsqueeze(0).to(DEVICE) 
                ga_test_pred_snn_output_channels = snn_model(gb_test_channels) 
                ga_test_pred_snn_output_complex = channels_to_spectrum_complex_torch(ga_test_pred_snn_output_channels.squeeze(0).cpu()).numpy() 
                
                snn_pred_Nfull_complex = extract_center_block_np(ga_test_pred_snn_output_complex, N_full_for_theorem) 
                
                diff_full_spectrum = snn_pred_Nfull_complex - ga_true_full_output_complex 
                error_theorem_sum = np.sum(sobolev_weights_LHS_sum_Nfull * np.abs(diff_full_spectrum)**2)

                B_value_this_sample = 0.0
                if args.pde_type == "poisson":
                    f_i_Nfull_coeffs = gb_full_test_complex 
                    if weights_f_Hsm2_Nfull is not None and weights_f_Hsm2_Nfull.size > 0:
                        norm_f_Hsm2_sq = np.sum(weights_f_Hsm2_Nfull * np.abs(f_i_Nfull_coeffs)**2)
                        B_value_this_sample = args.elliptic_PDE_const_C_sq * norm_f_Hsm2_sq
                    else:
                        B_value_this_sample = B_sq_bound_default 
                elif args.pde_type == "step_index_fiber":
                    gamma_b_i_Nfull_coeffs = gb_full_test_complex
                    if weights_gamma_b_Hs_Nfull is not None and weights_gamma_b_Hs_Nfull.size > 0:
                        norm_gamma_b_Hs_sq = np.sum(weights_gamma_b_Hs_Nfull * np.abs(gamma_b_i_Nfull_coeffs)**2)
                        factor_V0_s = 2 * (1 + 2 * args.fiber_potential_depth**2)**args.s_theorem
                        B_value_this_sample = norm_gamma_b_Hs_sq
                    else:
                        B_value_this_sample = B_sq_bound_default
                else: # Fallback for other PDE types or if phenomenological was re-added
                    B_value_this_sample = B_sq_bound_default 
                
                correction_term_this_sample = 0.0
                if np.isclose(args.nu_theorem, 0.0):
                    correction_term_this_sample = B_value_this_sample 
                else: 
                    if snn_output_res_val > 0: 
                        scaling_factor = (snn_output_res_val**(-2 * args.nu_theorem)) 
                        correction_term_this_sample = B_value_this_sample * scaling_factor
                    else: 
                        correction_term_this_sample = float('inf') if B_value_this_sample > 1e-9 else 0.0
                
                R_bound_this_sample = q_hat_nu_val + correction_term_this_sample
                current_alpha_R_bounds.append(R_bound_this_sample)
                if error_theorem_sum <= R_bound_this_sample:
                    covered_count_theorem += 1
            
            empirical_coverages_theorem.append(covered_count_theorem / len(gamma_b_full_test))
            if current_alpha_R_bounds: 
                avg_R_bounds_for_alpha_if_sample_dependent.append(np.mean(current_alpha_R_bounds))
            else: 
                avg_R_bounds_for_alpha_if_sample_dependent.append(q_hat_nu_val) 


    print("\n--- Theorem Coverage Results ---") 
    for i, alpha_q in enumerate(alpha_values_for_quantiles): 
        r_bound_to_print = avg_R_bounds_for_alpha_if_sample_dependent[i]
        print(f"  alpha={alpha_q:.2f}, Nom.Cov={1-alpha_q:.2f}, Emp.Cov (Thm)={empirical_coverages_theorem[i]:.4f}, q_hat_nu={quantiles_q_hat_nu[i]:.3e}, Avg_R_bound={r_bound_to_print:.3e}")

    if not args.no_plot: 
        plt.figure(figsize=(8,6))
        plt.plot(1-alpha_values_for_quantiles, empirical_coverages_theorem, marker='s',label='Empirical Coverage (Theorem)')
        plt.plot([0,1],[0,1],linestyle='--',color='gray',label='Ideal')
        plt.xlabel("Nominal Coverage ($1-\\alpha$)")
        plt.ylabel("Empirical Coverage")
        plt.title(f"Conformal Prediction Theorem Validation {scenario_title_suffix}")
        plt.legend()
        plt.grid(True)
        plt.xlim(0,1)
        plt.ylim(0,1.05)
        theorem_coverage_plot_path = os.path.join(args.results_dir, f"conformal_theorem_coverage{output_filename_suffix_calib}.png")
        plt.savefig(theorem_coverage_plot_path)
        print(f"\nTheorem coverage plot saved to {theorem_coverage_plot_path}")
        plt.show()
        plt.close() # Close the figure
    
    save_B_sq_info_str = f"DefaultMaxWeightB_{B_sq_bound_default:.3e}"
    if args.pde_type == "poisson": 
        save_B_sq_info_str = f"Poisson_sample_dependent_Csq_{args.elliptic_PDE_const_C_sq}"
        if not np.isclose(args.nu_theorem, 0.0):
             save_B_sq_info_str += f"_scaled_by_NmaxNu"
    elif args.pde_type == "step_index_fiber":
        save_B_sq_info_str = f"Fiber_sample_dependent_V0_{args.fiber_potential_depth}_s_{args.s_theorem}"
        if not np.isclose(args.nu_theorem, 0.0):
             save_B_sq_info_str += f"_scaled_by_NmaxNu"

    coverage_data_filename = os.path.join(args.results_dir, f"coverage_data{output_filename_suffix_calib}.npz") 
    np.savez_compressed(coverage_data_filename, 
                        nominal_coverages=1-alpha_values_for_quantiles, 
                        empirical_coverages_theorem=np.array(empirical_coverages_theorem),
                        quantiles_q_hat_nu=np.array(quantiles_q_hat_nu), 
                        avg_R_bounds_for_alpha=np.array(avg_R_bounds_for_alpha_if_sample_dependent),
                        B_sq_info=str(save_B_sq_info_str), 
                        snn_output_res=snn_output_res_val, 
                        k_trunc_full_theorem_eval=N_full_for_theorem, 
                        s_theorem=args.s_theorem, 
                        nu_theorem=args.nu_theorem,
                        d_dimensions=args.d_dimensions, 
                        k_trunc_bound_for_B_calc=args.k_trunc_bound)
    print(f"Coverage data saved to {coverage_data_filename}")

