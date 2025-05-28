import numpy as np
import os
import matplotlib.pyplot as plt # For optional visualization
import argparse # Added for command-line arguments

# --- Assume random_low_order_state is available (e.g., from dataset.py) ---
# For standalone running, let's define it here.
def random_low_order_state(N, K_max_band=16):
    """
    Construct a random wavefunction whose fft2 is nonzero only in 
    the band -K_max_band..K_max_band for both axes. Return it in real space.
    Normalized such that sum(|psi|^2) = 1 over the N x N real-space grid.
    """
    freq_array = np.zeros((N,N), dtype=np.complex128)
    def wrap_index(k_val): return k_val % N
    
    for kx_eff in range(-K_max_band, K_max_band + 1):
        for ky_eff in range(-K_max_band, K_max_band + 1):
            amp_real = np.random.randn()
            amp_imag = np.random.randn()
            c = amp_real + 1j*amp_imag
            kx = wrap_index(kx_eff)
            ky = wrap_index(ky_eff)
            freq_array[kx, ky] = c
            
    psi0_real = np.fft.ifft2(freq_array) 
    norm_psi = np.linalg.norm(psi0_real) # sqrt(sum(|psi_ij|^2))
    if norm_psi > 1e-14:
        psi0_real /= norm_psi
    return psi0_real

def get_truncated_spectrum(psi_real_space, K_trunc):
    """
    Computes the 2D FFT of psi_real_space, shifts it so DC is at center,
    truncates it to a K_trunc x K_trunc central block, AND NORMALIZES THIS BLOCK
    such that sum(|coeffs_in_block|^2) = 1.
    """
    if psi_real_space.ndim != 2 or psi_real_space.shape[0] != psi_real_space.shape[1]:
        raise ValueError("Input psi_real_space must be a square 2D array.")
    N_grid_psi = psi_real_space.shape[0]

    F_psi_unshifted = np.fft.fft2(psi_real_space) 
    F_psi_shifted = np.fft.fftshift(F_psi_unshifted)

    if K_trunc > N_grid_psi:
        print(f"Warning: K_trunc ({K_trunc}) is larger than N_grid_psi ({N_grid_psi}). Resulting spectrum will be from full FFT.")
        K_trunc = N_grid_psi 
    
    if K_trunc <= 0 : # Handle invalid K_trunc
        print(f"Warning: K_trunc ({K_trunc}) is invalid. Returning empty array.")
        return np.array([], dtype=np.complex64)

    start_idx = N_grid_psi // 2 - K_trunc // 2
    end_idx = start_idx + K_trunc
    
    truncated_block_unnormalized = F_psi_shifted[start_idx:end_idx, start_idx:end_idx]
    
    norm_sq_block = np.sum(np.abs(truncated_block_unnormalized)**2)
    if norm_sq_block > 1e-14: 
        return truncated_block_unnormalized / np.sqrt(norm_sq_block)
    else:
        return np.zeros_like(truncated_block_unnormalized)


def apply_phenomenological_noise_channel(spectrum_b_normalized_truncated, config_channel_noise):
    """
    Applies several types of phenomenological noise to a TRUNCATED AND NORMALIZED spectrum.
    The input spectrum_b_normalized_truncated is a K_trunc x K_trunc complex array
    with sum(|coeffs|^2) = 1.
    The output spectrum_a_centered_truncated is also K_trunc x K_trunc and
    is re-normalized such that sum(|coeffs|^2) = 1 over the K_trunc x K_trunc block.
    """
    c_current = spectrum_b_normalized_truncated.copy()
    K_trunc = c_current.shape[0]
    if K_trunc == 0: return c_current 

    if K_trunc % 2 == 0: 
        k_vals_one_dim_eff = np.arange(-K_trunc//2, K_trunc//2)
    else: 
        k_vals_one_dim_eff = np.arange(-(K_trunc-1)//2, (K_trunc-1)//2 + 1)
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
    if norm_factor_sq > 1e-12: 
        c_current /= np.sqrt(norm_factor_sq)
    else: 
        c_current = np.zeros_like(c_current)
    return c_current

def extract_center_block_and_normalize(full_spectrum_centered, K_extract):
    """
    Extracts the central K_extract x K_extract block from a larger centered spectrum
    and normalizes this extracted block so sum(|coeffs_in_block|^2) = 1.
    """
    N_full = full_spectrum_centered.shape[0]
    if K_extract > N_full:
        raise ValueError(f"K_extract ({K_extract}) cannot be larger than full_spectrum size ({N_full}).")
    if K_extract == 0: return np.array([]) 
    
    start_idx = N_full // 2 - K_extract // 2
    end_idx = start_idx + K_extract
    extracted_block = full_spectrum_centered[start_idx:end_idx, start_idx:end_idx]
    
    norm_sq_block = np.sum(np.abs(extracted_block)**2)
    if norm_sq_block > 1e-14:
        return extracted_block / np.sqrt(norm_sq_block)
    return np.zeros_like(extracted_block)


def generate_multires_dataset(num_samples, N_grid_simulation, K_psi0_band_limit, 
                              K_trunc_snn, K_trunc_full_eval, 
                              channel_config,
                              save_path_template="datasets/phenomenological_channel_dataset_Nmax{Nmax}_Nfull{Nfull}_{noise_str}.npz",
                              noise_config_str="no_noise"): # Added noise_config_str
    """
    Generates a dataset with spectra at two resolutions.
    - gamma_b_Nmax: SNN input (K_trunc_snn x K_trunc_snn), normalized
    - gamma_a_Nmax_true: SNN target (K_trunc_snn x K_trunc_snn), normalized
    - gamma_a_Nfull_true: Full true output (K_trunc_full_eval x K_trunc_full_eval), normalized
    """
    dataset_gamma_b_Nmax = []
    dataset_gamma_a_Nmax_true = []
    dataset_gamma_a_Nfull_true = []
    
    print(f"Generating {num_samples} multi-resolution samples...")
    for i in range(num_samples):
        if (i+1) % (num_samples // 10 or 1) == 0:
            print(f"  Generating sample {i+1}/{num_samples}")

        psi_b_real = random_low_order_state(N_grid_simulation, K_max_band=K_psi0_band_limit)
        gamma_b_Nmax_spec = get_truncated_spectrum(psi_b_real, K_trunc_snn) 
        gamma_a_Nmax_noisy_spec = apply_phenomenological_noise_channel(gamma_b_Nmax_spec, channel_config)
        
        dataset_gamma_b_Nmax.append(gamma_b_Nmax_spec)
        dataset_gamma_a_Nmax_true.append(gamma_a_Nmax_noisy_spec)

        if K_trunc_full_eval != K_trunc_snn:
            gamma_b_Nfull_spec = get_truncated_spectrum(psi_b_real, K_trunc_full_eval)
            gamma_a_Nfull_noisy_spec = apply_phenomenological_noise_channel(gamma_b_Nfull_spec, channel_config)
            dataset_gamma_a_Nfull_true.append(gamma_a_Nfull_noisy_spec)
        else:
            dataset_gamma_a_Nfull_true.append(gamma_a_Nmax_noisy_spec)

    gamma_b_Nmax_all = np.array(dataset_gamma_b_Nmax, dtype=np.complex64)
    gamma_a_Nmax_true_all = np.array(dataset_gamma_a_Nmax_true, dtype=np.complex64)
    gamma_a_Nfull_true_all = np.array(dataset_gamma_a_Nfull_true, dtype=np.complex64)
    
    # Use the passed noise_config_str in the filename
    current_save_path = save_path_template.format(Nmax=K_trunc_snn, Nfull=K_trunc_full_eval, noise_str=noise_config_str)
    os.makedirs(os.path.dirname(current_save_path) or '.', exist_ok=True)
    np.savez_compressed(current_save_path, 
                        gamma_b_Nmax=gamma_b_Nmax_all, 
                        gamma_a_Nmax_true=gamma_a_Nmax_true_all, 
                        gamma_a_Nfull_true=gamma_a_Nfull_true_all)
    print(f"Dataset saved to {current_save_path}")
    print(f"Shapes: gamma_b (Nmax): {gamma_b_Nmax_all.shape}, gamma_a (Nmax_true): {gamma_a_Nmax_true_all.shape}, gamma_a (Nfull_true): {gamma_a_Nfull_true_all.shape}")
    
    return gamma_b_Nmax_all, gamma_a_Nmax_true_all, gamma_a_Nfull_true_all

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate dataset with phenomenological noise for SNN training.")
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate.')
    parser.add_argument('--n_grid_sim', type=int, default=64, help='Grid size for initial real-space wavefunction (must be >= k_trunc_snn).')
    parser.add_argument('--k_psi0_limit', type=int, default=12, help='Max k for random_low_order_state (-K..K band).')
    parser.add_argument('--k_trunc_snn', type=int, default=32, help='N_max: Truncation for SNN input/output spectra.')
    parser.add_argument('--k_trunc_full', type=int, default=32, help='N_full_max: Higher resolution spectrum (can be same as k_trunc_snn).') 
    parser.add_argument('--output_dir', type=str, default="datasets", help='Directory to save the dataset.')

    # Noise Channel Parameters
    parser.add_argument('--apply_attenuation', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--attenuation_loss_factor', type=float, default=0.2)
    
    parser.add_argument('--apply_additive_sobolev_noise', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--sobolev_noise_level_base', type=float, default=0.01)
    parser.add_argument('--sobolev_order_s', type=float, default=1.0)
    
    parser.add_argument('--apply_phase_noise', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--phase_noise_std_rad', type=float, default=0.05)
    
    args = parser.parse_args()

    if args.k_trunc_snn > args.n_grid_sim:
        raise ValueError("K_TRUNC_SNN cannot be larger than N_GRID_SIM.")
    if args.k_trunc_full < args.k_trunc_snn:
         print(f"Warning: k_trunc_full ({args.k_trunc_full}) is less than k_trunc_snn ({args.k_trunc_snn}). Setting k_trunc_full = k_trunc_snn.")
         args.k_trunc_full = args.k_trunc_snn
    if args.n_grid_sim < args.k_trunc_full:
        raise ValueError("N_GRID_SIM must be >= K_TRUNC_FULL.")

    channel_noise_config = {
        'apply_attenuation': args.apply_attenuation,
        'attenuation_loss_factor': args.attenuation_loss_factor,
        'apply_additive_sobolev_noise': args.apply_additive_sobolev_noise,
        'sobolev_noise_level_base': args.sobolev_noise_level_base,
        'sobolev_order_s': args.sobolev_order_s,
        'apply_phase_noise': args.apply_phase_noise,
        'phase_noise_std_rad': args.phase_noise_std_rad
    }
    
    # Construct noise_config_str for filename
    noise_parts = []
    if args.apply_attenuation: noise_parts.append(f"att{args.attenuation_loss_factor:.2f}")
    if args.apply_additive_sobolev_noise: noise_parts.append(f"sob{args.sobolev_noise_level_base:.3f}s{args.sobolev_order_s:.1f}")
    if args.apply_phase_noise: noise_parts.append(f"ph{args.phase_noise_std_rad:.2f}")
    noise_config_str_filename = "_".join(noise_parts) if noise_parts else "no_noise"

    print("--- Phenomenological Channel Dataset Generation (Parameterized) ---")
    print(f"Noise configuration string for filename: {noise_config_str_filename}")
    # ... (other print statements as before) ...

    filename_template = os.path.join(args.output_dir, "phenomenological_channel_dataset_Nmax{Nmax}_Nfull{Nfull}_" + noise_config_str_filename + ".npz")

    gamma_b_data, gamma_a_Nmax_data, gamma_a_Nfull_data = generate_multires_dataset(
        args.num_samples,
        args.n_grid_sim,
        args.k_psi0_limit,
        args.k_trunc_snn,
        args.k_trunc_full,
        channel_noise_config,
        save_path_template=filename_template, # Pass the full template string
        noise_config_str=noise_config_str_filename # This argument is actually not used by generate_multires_dataset anymore
                                                   # as the full template is passed. Keeping for clarity if needed.
    )

    if args.num_samples > 0 and gamma_b_data.size > 0:
        # ... (visualization logic as before, ensure filenames for plots also use noise_config_str_filename) ...
        sample_idx = 0
        gb_Nmax_sample = gamma_b_data[sample_idx]
        ga_Nmax_true_sample = gamma_a_Nmax_data[sample_idx]
        ga_Nfull_true_sample = gamma_a_Nfull_data[sample_idx]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        im0 = axes[0].imshow(np.abs(gb_Nmax_sample)); axes[0].set_title(f"Sample $\gamma_b$ ($N_{{max}}={args.k_trunc_snn}$)"); plt.colorbar(im0, ax=axes[0])
        im1 = axes[1].imshow(np.abs(ga_Nmax_true_sample)); axes[1].set_title(f"Sample $\gamma_a$ ($N_{{max}}={args.k_trunc_snn}$)"); plt.colorbar(im1, ax=axes[1])
        im2 = axes[2].imshow(np.abs(ga_Nfull_true_sample)); axes[2].set_title(f"Sample $\gamma_a$ ($N_{{full}}={args.k_trunc_full}$)"); plt.colorbar(im2, ax=axes[2])
        plt.tight_layout()
        vis_dir = "results_dataset_gen_multires_cmd"
        os.makedirs(vis_dir, exist_ok=True)
        save_fig_path = os.path.join(vis_dir, f"sample_spectra_Nmax{args.k_trunc_snn}_Nfull{args.k_trunc_full}_{noise_config_str_filename}.png")
        plt.savefig(save_fig_path)
        print(f"\nSample spectra visualization saved to {save_fig_path}")
        # plt.show() 
    else:
        print("No samples generated or data is empty, skipping visualization.")
