import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import argparse 

# --- SNN Model Definition (should match the one used for training) ---
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
        if spectrum_mat_complex.ndim == 3 and spectrum_mat_complex.shape[0] == 2:
            return spectrum_mat_complex.float() 
        raise ValueError(f"Input spectrum_mat_complex has shape {spectrum_mat_complex.shape} and is real. Expected complex [K,K] or real [2,K,K].")
    return torch.stack([spectrum_mat_complex.real, spectrum_mat_complex.imag], dim=0).float()

def channels_to_spectrum_complex_torch(channels_mat_real_imag):
    if channels_mat_real_imag.ndim != 3 or channels_mat_real_imag.shape[0] != 2:
        raise ValueError(f"Input must have 2 channels as the first dimension, got shape {channels_mat_real_imag.shape}")
    return torch.complex(channels_mat_real_imag[0], channels_mat_real_imag[1])

def get_mode_indices_and_weights(K_grid_size, d_dimensions, s_coeff, nu_coeff):
    """
    Generates mode indices (nx, ny for d=2) for a K_grid_size x K_grid_size
    and their corresponding Sobolev weights (1 + ||n||_2^(2d))^(s_coeff - nu_coeff).
    Assumes centered spectrum.
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
    elif d_dimensions == 3:
        nx_eff_grid, ny_eff_grid, nz_eff_grid = np.meshgrid(k_vals_one_dim_eff, k_vals_one_dim_eff, k_vals_one_dim_eff, indexing='ij')
        n_grids_list = [nx_eff_grid, ny_eff_grid, nz_eff_grid]
    else:
        raise ValueError("d_dimensions must be 1, 2, or 3 for this implementation.")

    norm_n_L2_sq = np.zeros_like(n_grids_list[0], dtype=float)
    for i in range(d_dimensions):
        norm_n_L2_sq += n_grids_list[i]**2
    
    term_inside_power = 1 + norm_n_L2_sq**d_dimensions 
    exponent = s_coeff - nu_coeff
    
    if np.isclose(exponent, 0.0): 
        sobolev_weights = np.ones_like(term_inside_power)
    else:
        sobolev_weights = term_inside_power**(exponent)
    
    if np.any(np.isnan(sobolev_weights)) or np.any(np.isinf(sobolev_weights)):
        print(f"Warning: NaN or Inf encountered in Sobolev weights. Exponent: {exponent}. Min term_inside_power: {np.min(term_inside_power)}")
        sobolev_weights = np.nan_to_num(sobolev_weights, nan=1.0, posinf=1e9, neginf=1e-9)
        if np.isclose(exponent, 0.0): sobolev_weights = np.ones_like(term_inside_power)

    return n_grids_list, sobolev_weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Conformal Prediction for SNN Error (Theorem Validation).")
    # SNN and Dataset structure
    parser.add_argument('--k_trunc_snn', type=int, default=32, help='N_max: Truncation SNN was trained for and for calibration scores.')
    parser.add_argument('--k_trunc_full', type=int, default=64, help='N_full_max: Full resolution for theorem evaluation.')
    parser.add_argument('--snn_model_dir', type=str, default="trained_snn_models", help="Directory containing trained SNN models.")
    parser.add_argument('--snn_hidden_channels', type=int, default=64, help="Hidden channels in SNN.")
    parser.add_argument('--snn_num_hidden_layers', type=int, default=3, help="Number of hidden layers in SNN.")
    parser.add_argument('--dataset_dir', type=str, default="datasets", help="Directory where datasets are stored.")
    parser.add_argument('--results_dir', type=str, default="results_conformal_theorem_validation", help="Directory to save output plots and data.")
    
    # Theorem and Weighting Parameters
    parser.add_argument('--s_theorem', type=float, default=2.0, help="Theorem parameter 's'.")
    parser.add_argument('--nu_theorem', type=float, default=2.0, help="Theorem parameter 'nu'.") # Changed default to 0 for L2 error
    parser.add_argument('--d_dimensions', type=int, default=2, choices=[1,2,3], help="Spatial dimension d for Sobolev weights.")
    parser.add_argument('--k_trunc_bound', type=int, default=48, help="K_grid_size for get_mode_indices_and_weights, used for B_sq bound.")

    # Conformal Prediction Parameters
    parser.add_argument('--calib_split_ratio', type=float, default=0.5, help="Ratio of data for calibration set.")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed for data splitting.")
    parser.add_argument('--no_plot', action='store_true', help="Suppress displaying the plot.")
    parser.add_argument('--num_states_gram_example', type=int, default=3, help="Number of states for example Gram matrix (not used in this script's core logic).")

    # Noise Channel Parameters (to construct correct filenames for dataset and model)
    parser.add_argument('--apply_attenuation', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--attenuation_loss_factor', type=float, default=0.2)
    parser.add_argument('--apply_additive_sobolev_noise', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--sobolev_noise_level_base', type=float, default=0.01)
    parser.add_argument('--sobolev_order_s', type=float, default=1.0)
    parser.add_argument('--apply_phase_noise', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--phase_noise_std_rad', type=float, default=0.05)
    # Argument to override SNN model filename if needed (passed by sweep script)
    parser.add_argument('--snn_model_filename_override', type=str, default=None, help="Exact SNN model filename to load.")


    args = parser.parse_args()

    # Construct noise_config_str from args to match dataset and model naming convention
    noise_parts = []
    if args.apply_attenuation: noise_parts.append(f"att{args.attenuation_loss_factor:.2f}")
    if args.apply_additive_sobolev_noise: noise_parts.append(f"sob{args.sobolev_noise_level_base:.3f}s{args.sobolev_order_s:.1f}")
    if args.apply_phase_noise: noise_parts.append(f"ph{args.phase_noise_std_rad:.2f}")
    noise_config_str_filename = "_".join(noise_parts) if noise_parts else "no_noise"

    if args.snn_model_filename_override:
        SNN_MODEL_FILENAME = args.snn_model_filename_override
    else:
        SNN_MODEL_FILENAME = f"snn_K{args.k_trunc_snn}_H{args.snn_hidden_channels}_L{args.snn_num_hidden_layers}_{noise_config_str_filename}.pth"
    SNN_MODEL_PATH = os.path.join(args.snn_model_dir, SNN_MODEL_FILENAME)
    
    DATASET_FILENAME = f"phenomenological_channel_dataset_Nmax{args.k_trunc_snn}_Nfull{args.k_trunc_full}_{noise_config_str_filename}.npz"
    DATASET_FILE_PATH = os.path.join(args.dataset_dir, DATASET_FILENAME)
    
    scenario_title_suffix = (f"(Thm: $s={args.s_theorem}, \\nu={args.nu_theorem}, d={args.d_dimensions}, "
                             f"N_{{max}}={args.k_trunc_snn}, N_{{full}}={args.k_trunc_full}, K_{{bound}}={args.k_trunc_bound}, "
                             f"Noise: {noise_config_str_filename}$)")
    plot_filename_suffix = (f"_thm_s{args.s_theorem}_nu{args.nu_theorem}_d{args.d_dimensions}"
                            f"_Nmax{args.k_trunc_snn}_Nfull{args.k_trunc_full}_Kbound{args.k_trunc_bound}_{noise_config_str_filename}")
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    alpha_values_for_quantiles = np.round(np.arange(0.05, 1.0, 0.05), 2) 
    os.makedirs(args.results_dir, exist_ok=True) # results_dir is now the unique subdir passed by sweep

    print(f"--- Running Conformal Prediction: Theorem Test {scenario_title_suffix} ---")
    print(f"SNN Model: {SNN_MODEL_PATH}")
    print(f"Dataset: {DATASET_FILE_PATH}")
    print(f"k_trunc_bound for B_sq calculation: {args.k_trunc_bound}")

    # --- Calculate B_sq_bound_theorem for the theorem ---
    # This uses args.k_trunc_bound, args.d_dimensions, args.s_theorem, and nu=0 for B_sq
    _, H_s_norm_weights_for_B_calc = get_mode_indices_and_weights(
        args.k_trunc_bound, # Use the passed k_trunc_bound
        args.d_dimensions, 
        args.s_theorem,  # s_coeff for B_sq calculation
        0                # nu_coeff is 0 for B_sq calculation as per original script
    )
    if H_s_norm_weights_for_B_calc.size > 0:
        B_sq_bound_theorem = np.max(H_s_norm_weights_for_B_calc) 
    else: 
        B_sq_bound_theorem = 1.0 
        print(f"Warning: Could not calculate B_sq_bound_theorem using k_trunc_bound={args.k_trunc_bound}, H_s_norm_weights_for_B_calc is empty.")
    print(f"Calculated B_sq (bound on ||U'||_H^s_B^2) for theorem: {B_sq_bound_theorem:.4e} (using s_B={args.s_theorem}, k_trunc_bound={args.k_trunc_bound})")

    # --- Load SNN and Data ---
    if not os.path.exists(SNN_MODEL_PATH): print(f"ERROR: SNN model not found: {SNN_MODEL_PATH}"); exit()
    snn_model = SimpleSpectralOperatorCNN(args.k_trunc_snn, args.snn_hidden_channels, args.snn_num_hidden_layers)
    snn_model.load_state_dict(torch.load(SNN_MODEL_PATH, map_location=DEVICE)); snn_model.to(DEVICE); snn_model.eval()
    print("SNN model loaded.")
    try:
        data = np.load(DATASET_FILE_PATH)
        gamma_b_Nmax_all = data['gamma_b_Nmax']; gamma_a_Nmax_true_all = data['gamma_a_Nmax_true'] 
        gamma_a_Nfull_true_all = data['gamma_a_Nfull_true'] 
    except Exception as e: print(f"Error loading dataset {DATASET_FILE_PATH}: {e}"); exit()
    
    # Validate dimensions
    if not (gamma_b_Nmax_all.ndim > 2 and gamma_b_Nmax_all.shape[1] == args.k_trunc_snn and \
            gamma_a_Nmax_true_all.ndim > 2 and gamma_a_Nmax_true_all.shape[1] == args.k_trunc_snn and \
            gamma_a_Nfull_true_all.ndim > 2 and gamma_a_Nfull_true_all.shape[1] == args.k_trunc_full):
        print(f"Error: K_TRUNC parameters in script do not match loaded data dimensions or data is not batch of 2D arrays."); exit()
    if args.k_trunc_full < args.k_trunc_snn : print("Error: k_trunc_full must be >= k_trunc_snn"); exit()
    
    num_total_samples = gamma_b_Nmax_all.shape[0]
    indices = np.arange(num_total_samples)
    # Ensure calib_split_ratio makes sense
    if not (0 < args.calib_split_ratio < 1):
        print("Warning: calib_split_ratio should be between 0 and 1. Using 0.5.")
        args.calib_split_ratio = 0.5

    # Ensure there are enough samples for split
    if num_total_samples < 2 :
        print("Error: Not enough samples in dataset for train/calib split. Need at least 2.")
        exit()
    
    # Ensure test_size is at least 1 if num_total_samples * (1-args.calib_split_ratio) < 1
    test_size_float = 1 - args.calib_split_ratio
    if int(num_total_samples * test_size_float) < 1 and num_total_samples > 1: # Ensure at least 1 test sample if possible
        test_size_float = 1.0 / num_total_samples
        print(f"Warning: calib_split_ratio too high for small dataset. Adjusting test split to ensure 1 test sample.")
    if int(num_total_samples * (1-test_size_float)) < 1 : # Ensure at least 1 calib sample
         print("Error: Not enough samples for calibration after split adjustment.")
         exit()


    cal_indices, test_indices = train_test_split(indices, test_size=test_size_float, random_state=args.random_seed, shuffle=True)
    gamma_b_Nmax_cal, gamma_a_Nmax_true_cal = gamma_b_Nmax_all[cal_indices], gamma_a_Nmax_true_all[cal_indices]
    gamma_b_Nmax_test = gamma_b_Nmax_all[test_indices]
    gamma_a_Nfull_true_test = gamma_a_Nfull_true_all[test_indices] 
    print(f"Cal set: {len(gamma_b_Nmax_cal)}, Test set: {len(gamma_b_Nmax_test)}")
    if not (len(gamma_b_Nmax_cal) > 0 and len(gamma_b_Nmax_test) > 0): print("Error: Cal or test set empty."); exit()

    # --- Calibration ---
    nonconformity_scores_cal = [] 
    print("\nCalculating nonconformity scores (UNWEIGHTED squared L2 norm on N_max spectra)...")
    with torch.no_grad():
        for i in range(len(gamma_b_Nmax_cal)):
            gb_cal_complex = gamma_b_Nmax_cal[i]; ga_cal_true_complex = gamma_a_Nmax_true_cal[i] 
            gb_cal_channels = spectrum_complex_to_channels_torch(gb_cal_complex).unsqueeze(0).to(DEVICE)
            ga_cal_pred_channels = snn_model(gb_cal_channels) 
            ga_cal_pred_complex = channels_to_spectrum_complex_torch(ga_cal_pred_channels.squeeze(0).cpu()).numpy()
            score = np.sum(np.abs(ga_cal_pred_complex - ga_cal_true_complex)**2)
            nonconformity_scores_cal.append(score) 
    nonconformity_scores_cal = np.array(nonconformity_scores_cal)
    print(f"Calculated {len(nonconformity_scores_cal)} calibration scores. Avg: {np.mean(nonconformity_scores_cal):.4e}")

    # --- Quantile Calculation (q_hat_nu from theorem) ---
    quantiles_q_hat_nu = []
    nominal_coverages_1_minus_alpha = 1 - alpha_values_for_quantiles
    n_cal = len(nonconformity_scores_cal)
    for alpha_q in alpha_values_for_quantiles:
        quantile_idx = int(np.ceil((n_cal + 1) * (1 - alpha_q))) -1 # 0-indexed
        quantile_idx = min(max(0, quantile_idx), n_cal -1) # Ensure valid index
        q_hat = np.sort(nonconformity_scores_cal)[quantile_idx]
        quantiles_q_hat_nu.append(q_hat)

    # --- Coverage Test ---
    empirical_coverages_theorem = []
    _, sobolev_weights_LHS_sum_Nfull = get_mode_indices_and_weights(
        args.k_trunc_full, args.d_dimensions, args.s_theorem, args.nu_theorem 
    )
    print(f"\nUsing Sobolev weights for THEOREM LHS sum (N_full grid) with exponent (s-nu) = {args.s_theorem - args.nu_theorem:.2f}")
    if args.k_trunc_full > 0 and sobolev_weights_LHS_sum_Nfull.size > 0:
        center_idx = args.k_trunc_full//2
        print(f"Sample LHS N_full weights (center, corner): W_0={sobolev_weights_LHS_sum_Nfull[center_idx, center_idx]:.2e}, W_corner={sobolev_weights_LHS_sum_Nfull[0,0]:.2e}")
    
    print("\nCalculating empirical coverage on test set using theorem's bound...")
    R_bounds_for_alpha = [] 
    with torch.no_grad():
        for q_idx, q_hat_nu_val in enumerate(quantiles_q_hat_nu):
            if args.k_trunc_snn == 0 and args.nu_theorem != 0 and args.d_dimensions !=0 : correction_term = float('inf')
            elif args.nu_theorem == 0: correction_term = B_sq_bound_theorem 
            else:
                 correction_term = B_sq_bound_theorem * (args.k_trunc_snn**(-2 * args.d_dimensions * args.nu_theorem))
            R_bound = q_hat_nu_val + correction_term
            R_bounds_for_alpha.append(R_bound) 
            covered_count_theorem = 0
            for i in range(len(gamma_b_Nmax_test)):
                gb_test_Nmax_complex = gamma_b_Nmax_test[i]
                ga_test_Nfull_true_complex = gamma_a_Nfull_true_all[test_indices[i]] 
                gb_test_channels = spectrum_complex_to_channels_torch(gb_test_Nmax_complex).unsqueeze(0).to(DEVICE)
                ga_test_pred_Nmax_channels = snn_model(gb_test_channels)
                ga_test_pred_Nmax_complex = channels_to_spectrum_complex_torch(ga_test_pred_Nmax_channels.squeeze(0).cpu()).numpy()
                snn_pred_Nfull_complex = np.zeros((args.k_trunc_full, args.k_trunc_full), dtype=np.complex128)
                start_idx_snn = args.k_trunc_full // 2 - args.k_trunc_snn // 2
                end_idx_snn = start_idx_snn + args.k_trunc_snn
                snn_pred_Nfull_complex[start_idx_snn:end_idx_snn, start_idx_snn:end_idx_snn] = ga_test_pred_Nmax_complex
                diff_full_spectrum = snn_pred_Nfull_complex - ga_test_Nfull_true_complex
                error_theorem_sum = np.sum(sobolev_weights_LHS_sum_Nfull * np.abs(diff_full_spectrum)**2)
                if error_theorem_sum <= R_bound:
                    covered_count_theorem += 1
            empirical_coverages_theorem.append(covered_count_theorem / len(gamma_b_Nmax_test))
    
    print("\n--- Theorem Coverage Results ---")
    for i, alpha_q in enumerate(alpha_values_for_quantiles):
        R_b_print = R_bounds_for_alpha[i] 
        print(f"  alpha={alpha_q:.2f}, Nom.Cov (1-a)={1-alpha_q:.2f}, Emp.Cov (Thm)={empirical_coverages_theorem[i]:.4f}, q_hat_nu={quantiles_q_hat_nu[i]:.3e}, R_bound={R_b_print:.3e}")

    if not args.no_plot: 
        plt.figure(figsize=(8, 6))
        plt.plot(1 - alpha_values_for_quantiles, empirical_coverages_theorem, marker='s', linestyle='-', label='Empirical Coverage (Theorem)')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal Coverage (y=x)')
        plt.xlabel("Nominal Coverage ($1-\\alpha$)")
        plt.ylabel("Empirical Coverage")
        plt.title(f"Conformal Prediction Theorem Validation {scenario_title_suffix}")
        plt.legend(); plt.grid(True); plt.xlim(0, 1); plt.ylim(0, 1.05)
        theorem_coverage_plot_path = os.path.join(args.results_dir, f"conformal_theorem_coverage{plot_filename_suffix}.png")
        plt.savefig(theorem_coverage_plot_path)
        print(f"\nTheorem coverage plot saved to {theorem_coverage_plot_path}")
        plt.show()
    else:
        print("\nPlotting suppressed by --no_plot argument.")
    
    coverage_data_filename = os.path.join(args.results_dir, f"coverage_data{plot_filename_suffix}.npz") 
    np.savez_compressed(coverage_data_filename,
                        nominal_coverages=1 - alpha_values_for_quantiles,
                        empirical_coverages_theorem=np.array(empirical_coverages_theorem),
                        quantiles_q_hat_nu=np.array(quantiles_q_hat_nu),
                        R_bounds_for_alpha=np.array(R_bounds_for_alpha), 
                        B_sq_bound_theorem=B_sq_bound_theorem,
                        k_trunc_snn=args.k_trunc_snn, k_trunc_full=args.k_trunc_full,
                        s_theorem=args.s_theorem, nu_theorem=args.nu_theorem,
                        d_dimensions=args.d_dimensions, k_trunc_bound_for_B=args.k_trunc_bound # Save k_trunc_bound used for B
                        )
    print(f"Coverage data saved to {coverage_data_filename}")
