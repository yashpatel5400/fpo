import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import argparse 

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
    parser.add_argument('--k_trunc_snn', type=int, default=32, help='N_max: Truncation for SNN input/output and calibration scores.')
    parser.add_argument('--k_trunc_full', type=int, default=64, help='N_full_max: Full resolution for theorem evaluation.')
    parser.add_argument('--k_trunc_bound', type=int, default=48, help='N_eff: Effective cutoff for theorem evaluation.')
    parser.add_argument('--snn_model_dir', type=str, default="trained_snn_models", help="Directory containing trained SNN models.")
    parser.add_argument('--snn_hidden_channels', type=int, default=64, help="Hidden channels in SNN (must match trained model).")
    parser.add_argument('--snn_num_hidden_layers', type=int, default=3, help="Number of hidden layers in SNN (must match trained model).")
    parser.add_argument('--dataset_dir', type=str, default="datasets", help="Directory where datasets are stored.")
    parser.add_argument('--results_dir', type=str, default="results_conformal_theorem_validation", help="Directory to save output plots and data.")
    
    parser.add_argument('--s_theorem', type=float, default=2.0, help="Theorem parameter 's' (for H^s norm of U' and in LHS sum).")
    parser.add_argument('--nu_theorem', type=float, default=2.0, help="Theorem parameter 'nu' (in LHS sum and N_max exponent).")
    parser.add_argument('--d_dimensions', type=int, default=2, choices=[1,2,3], help="Spatial dimension d for Sobolev weights.")
    
    parser.add_argument('--calib_split_ratio', type=float, default=0.5, help="Ratio of data for calibration set.")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed for data splitting.")
    parser.add_argument('--no_plot', action='store_true', help="Suppress displaying the plot.")
    parser.add_argument('--num_states_gram_example', type=int, default=3, help="Number of states to use for example Gram matrix calculation.")


    args = parser.parse_args()

    SNN_MODEL_FILENAME = f"snn_K{args.k_trunc_snn}_H{args.snn_hidden_channels}_L{args.snn_num_hidden_layers}_multires.pth"
    SNN_MODEL_PATH = os.path.join(args.snn_model_dir, SNN_MODEL_FILENAME)
    DATASET_FILENAME = f"phenomenological_channel_dataset_Nmax{args.k_trunc_snn}_Nfull{args.k_trunc_full}.npz"
    DATASET_FILE_PATH = os.path.join(args.dataset_dir, DATASET_FILENAME)
    
    scenario_title_suffix = f"(Thm: $s={args.s_theorem}, \\nu={args.nu_theorem}, d={args.d_dimensions}, N_{{max}}={args.k_trunc_snn}, N_{{full}}={args.k_trunc_full}$)"
    plot_filename_suffix = f"_thm_s{args.s_theorem}_nu{args.nu_theorem}_d{args.d_dimensions}_Nmax{args.k_trunc_snn}_Nfull{args.k_trunc_full}"
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    alpha_values_for_quantiles = np.round(np.arange(0.05, 1.0, 0.05), 2) 
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"--- Running Conformal Prediction: Theorem Test {scenario_title_suffix} ---")
    print(f"SNN Model: {SNN_MODEL_PATH}")
    print(f"Dataset: {DATASET_FILE_PATH}")

    # --- Calculate B_sq_bound_theorem for the theorem ---
    _, H_s_norm_weights_for_B_calc = get_mode_indices_and_weights(
        args.k_trunc_bound, 
        args.d_dimensions, 
        args.s_theorem,  
        0                
    )
    if H_s_norm_weights_for_B_calc.size > 0:
        B_sq_bound_theorem = np.max(H_s_norm_weights_for_B_calc) 
    else: 
        B_sq_bound_theorem = 1.0 
        print("Warning: Could not calculate B_sq_bound_theorem, H_s_norm_weights_for_B_calc is empty.")
    print(f"Calculated B_sq (bound on ||U'||_H^s_B^2) for theorem: {B_sq_bound_theorem:.4e} (using s_B={args.s_theorem})")

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
    if not (gamma_b_Nmax_all.shape[1] == args.k_trunc_snn and \
            gamma_a_Nmax_true_all.shape[1] == args.k_trunc_snn and \
            gamma_a_Nfull_true_all.shape[1] == args.k_trunc_full):
        print(f"Error: K_TRUNC parameters in script do not match loaded data dimensions."); exit()
    if args.k_trunc_full < args.k_trunc_snn : print("Error: k_trunc_full must be >= k_trunc_snn"); exit()
    
    num_total_samples = gamma_b_Nmax_all.shape[0]
    indices = np.arange(num_total_samples)
    cal_indices, test_indices = train_test_split(indices, test_size=(1-args.calib_split_ratio), random_state=args.random_seed, shuffle=True)
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
        quantile_level = min(np.ceil((n_cal + 1) * (1 - alpha_q)) / n_cal, 1.0) 
        q_hat = np.quantile(nonconformity_scores_cal, quantile_level, method='higher' if quantile_level < 1.0 else 'linear') 
        quantiles_q_hat_nu.append(q_hat)

    # --- Coverage Test ---
    empirical_coverages_theorem = []
    _, sobolev_weights_LHS_sum_Nfull = get_mode_indices_and_weights(
        args.k_trunc_full, args.d_dimensions, args.s_theorem, args.nu_theorem 
    )
    print(f"\nUsing Sobolev weights for THEOREM LHS sum (N_full grid) with exponent (s-nu) = {args.s_theorem - args.nu_theorem:.2f}")
    if args.k_trunc_full > 0:
        print(f"Sample LHS N_full weights (center, corner): W_0={sobolev_weights_LHS_sum_Nfull[args.k_trunc_full//2, args.k_trunc_full//2]:.2e}, W_corner={sobolev_weights_LHS_sum_Nfull[0,0]:.2e}")
    
    print("\nCalculating empirical coverage on test set using theorem's bound...")
    R_bounds_for_alpha = [] # Store R_bound for each alpha
    with torch.no_grad():
        for q_idx, q_hat_nu_val in enumerate(quantiles_q_hat_nu):
            if args.k_trunc_snn == 0 and args.nu_theorem != 0 and args.d_dimensions !=0 : correction_term = float('inf')
            elif args.nu_theorem == 0: correction_term = B_sq_bound_theorem 
            else:
                 correction_term = B_sq_bound_theorem * (args.k_trunc_snn**(-2 * args.d_dimensions * args.nu_theorem))
            R_bound = q_hat_nu_val + correction_term
            R_bounds_for_alpha.append(R_bound) # Store this R_bound
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
        R_b_print = R_bounds_for_alpha[i] # Use stored R_bound
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
                        R_bounds_for_alpha=np.array(R_bounds_for_alpha), # Save the R_bounds used
                        B_sq_bound_theorem=B_sq_bound_theorem,
                        k_trunc_snn=args.k_trunc_snn,
                        k_trunc_full=args.k_trunc_full,
                        s_theorem=args.s_theorem,
                        nu_theorem=args.nu_theorem,
                        d_dimensions=args.d_dimensions
                        )
    print(f"Coverage data saved to {coverage_data_filename}")

    # --- Estimating Frobenius norm difference for Gram Matrix ---
    target_alpha_for_gram = 0.10 
    target_nominal_coverage_gram = 1 - target_alpha_for_gram
    try:
        idx_for_gram_q_hat = -1
        for i_alpha, val_alpha in enumerate(np.round(alpha_values_for_quantiles,2)):
            if np.isclose(val_alpha, target_alpha_for_gram):
                idx_for_gram_q_hat = i_alpha
                break
        if idx_for_gram_q_hat == -1: raise ValueError(f"Target alpha {target_alpha_for_gram} not found.")

        # R_bound_for_gram is the R_bound corresponding to target_alpha_for_gram
        # This R_bound is for the Sobolev-weighted sum of squared errors on the N_full grid.
        R_bound_for_gram_sobolev_weighted = R_bounds_for_alpha[idx_for_gram_q_hat]
            
        print(f"\n--- Estimating Gram Matrix Frobenius Norm Error Bound (using theorem's R_bound) ---")
        print(f"For nominal coverage {target_nominal_coverage_gram:.2f} (alpha={target_alpha_for_gram:.2f}):")
        print(f"  R_bound for Sobolev-weighted sum (LHS of theorem, s-nu={args.s_theorem - args.nu_theorem:.2f}): {R_bound_for_gram_sobolev_weighted:.4e}")

        # If the LHS Sobolev weights were 1 (i.e., s_theorem - nu_theorem = 0),
        # then R_bound_for_gram_sobolev_weighted would be an upper bound on sum_Nfull |error_n|^2.
        # And sqrt(R_bound_for_gram_sobolev_weighted) would be an L2 error bound.
        if np.isclose(args.s_theorem - args.nu_theorem, 0.0):
            q_L2_error_full_sq_conformal = R_bound_for_gram_sobolev_weighted
            q_L2_error_full_conformal = np.sqrt(q_L2_error_full_sq_conformal)
            
            NUM_STATES_FOR_GRAM_M_example = args.num_states_gram_example 
            q_G_Frobenius_estimated = NUM_STATES_FOR_GRAM_M_example * (2 * q_L2_error_full_conformal + q_L2_error_full_conformal**2)
            
            print(f"  Since s-nu=0 for LHS, R_bound is for unweighted L2 sum of sq errors on N_full.")
            print(f"    Conformal bound on L2 norm of FULL error (sqrt(R_bound)) = {q_L2_error_full_conformal:.4e}")
            print(f"    Estimated Frobenius norm bound for ||G_est_NmaxPadded - G_true_Nfull||_F (for M={NUM_STATES_FOR_GRAM_M_example} states) = {q_G_Frobenius_estimated:.4e}")
        else:
            print(f"  The R_bound ({R_bound_for_gram_sobolev_weighted:.4e}) is for a Sobolev-weighted error sum (s-nu = {args.s_theorem - args.nu_theorem:.2f}).")
            print(f"  To use the M(2q+q^2) formula for Gram matrix error, an L2 error quantile (from s-nu=0 calibration) is needed.")
            print(f"  The direct calculation of ||G_true_Nfull - G_est_NmaxPadded||_F below uses actual SNN predictions.")

        # Direct calculation of Gram matrix difference for example
        M_for_gram_direct = args.num_states_gram_example
        if len(gamma_b_Nmax_test) >= M_for_gram_direct and M_for_gram_direct > 1:
            gamma_b_gram_Nmax = gamma_b_Nmax_test[:M_for_gram_direct]
            gamma_a_gram_Nmax_pred_list = []
            with torch.no_grad():
                for i in range(M_for_gram_direct):
                    gb_channels = spectrum_complex_to_channels_torch(gamma_b_gram_Nmax[i]).unsqueeze(0).to(DEVICE)
                    ga_pred_channels = snn_model(gb_channels)
                    ga_pred_complex = channels_to_spectrum_complex_torch(ga_pred_channels.squeeze(0).cpu()).numpy()
                    gamma_a_gram_Nmax_pred_list.append(ga_pred_complex)
            gamma_a_gram_Nfull_true_list = gamma_a_Nfull_true_all[test_indices[:M_for_gram_direct]]
            G_true_example = np.zeros((M_for_gram_direct, M_for_gram_direct), dtype=np.complex128)
            for r_idx in range(M_for_gram_direct):
                for c_idx in range(M_for_gram_direct):
                    G_true_example[r_idx, c_idx] = np.vdot(gamma_a_gram_Nfull_true_list[r_idx].ravel(), gamma_a_gram_Nfull_true_list[c_idx].ravel())
            G_est_example = np.zeros((M_for_gram_direct, M_for_gram_direct), dtype=np.complex128)
            for r_idx in range(M_for_gram_direct):
                for c_idx in range(M_for_gram_direct):
                    G_est_example[r_idx, c_idx] = np.vdot(gamma_a_gram_Nmax_pred_list[r_idx].ravel(), gamma_a_gram_Nmax_pred_list[c_idx].ravel())
            gram_diff_F_norm_example = np.linalg.norm(G_true_example - G_est_example, 'fro')
            print(f"  Directly calculated ||G_true_Nfull - G_est_Nmax||_F for first {M_for_gram_direct} test samples: {gram_diff_F_norm_example:.4e}")

    except (ValueError, IndexError) as e:
        print(f"\nError during Gram matrix error estimation/reporting: {e}")

