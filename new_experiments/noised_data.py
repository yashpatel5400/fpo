import numpy as np
import os
import matplotlib.pyplot as plt # For optional visualization

# --- Assume random_low_order_state is available (e.g., from dataset.py) ---
# For standalone running, let's define it here.
def random_low_order_state(N, K_max_band=16):
    """
    Construct a random wavefunction whose fft2 is nonzero only in 
    the band -K_max_band..K_max_band for both axes. Return it in real space.
    Normalized such that sum(|psi|^2) = 1.
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
    and truncates it to a K_trunc x K_trunc central block.
    """
    if psi_real_space.ndim != 2 or psi_real_space.shape[0] != psi_real_space.shape[1]:
        raise ValueError("Input psi_real_space must be a square 2D array.")
    N_grid_psi = psi_real_space.shape[0]

    F_psi_shifted = np.fft.fftshift(np.fft.fft2(psi_real_space))

    if K_trunc > N_grid_psi:
        raise ValueError(f"K_trunc ({K_trunc}) cannot be larger than N_grid_psi ({N_grid_psi}).")
    
    start_idx = N_grid_psi // 2 - K_trunc // 2
    end_idx = start_idx + K_trunc
    
    # The coefficients F_psi_shifted are such that if sum_xy |psi(x,y)|^2 = 1, 
    # then sum_k |F_psi_unshifted_k|^2 = N_grid_psi^2.
    # And sum_k |F_psi_shifted_k|^2 = N_grid_psi^2.
    # The SNN often works with these scaled coefficients.
    # If the SNN expects coefficients c_k such that sum |c_k|^2 = 1 (in truncated space),
    # then normalization is needed after truncation or after noise.
    return F_psi_shifted[start_idx:end_idx, start_idx:end_idx]


def apply_phenomenological_noise_channel(spectrum_b_centered_truncated, config):
    """
    Applies several types of phenomenological noise to a truncated spectrum.
    The input spectrum_b_centered_truncated is a K_trunc x K_trunc complex array.
    The output spectrum_a_centered_truncated is also K_trunc x K_trunc and
    is normalized such that sum(|coeffs|^2) = 1 over the K_trunc x K_trunc block.
    """
    c_current = spectrum_b_centered_truncated.copy()
    K_trunc = c_current.shape[0]

    # Create meshgrid of effective mode indices for this K_trunc x K_trunc spectrum
    # If K_trunc=32, effective indices k_eff run from -16 to 15.
    # np.fft.fftfreq(K_trunc) * K_trunc gives these integer indices but in FFT order.
    # np.fft.ifftshift applied to np.arange will center it.
    if K_trunc % 2 == 0: # Even K_trunc, e.g., 32 -> -16 to 15
        k_vals_one_dim_eff = np.arange(-K_trunc//2, K_trunc//2)
    else: # Odd K_trunc, e.g., 31 -> -15 to 15
        k_vals_one_dim_eff = np.arange(-(K_trunc-1)//2, (K_trunc-1)//2 + 1)
        
    kx_eff_grid, ky_eff_grid = np.meshgrid(k_vals_one_dim_eff, k_vals_one_dim_eff, indexing='ij')

    # 1. Mode-Dependent Attenuation
    if config.get('apply_attenuation', False):
        loss_factor = config.get('attenuation_loss_factor', 0.1) 
        norm_sq_grid = kx_eff_grid**2 + ky_eff_grid**2
        # Normalize norm_sq_grid to roughly [0,1] for effective loss_factor application
        max_norm_sq_in_trunc = np.max(norm_sq_grid) if K_trunc > 1 else 1.0
        if max_norm_sq_in_trunc == 0: max_norm_sq_in_trunc = 1.0 # Avoid division by zero for K_trunc=1 (only DC)
        
        attenuation_profile = np.exp(-loss_factor * norm_sq_grid / max_norm_sq_in_trunc)
        c_current = c_current * attenuation_profile

    # 2. Additive Sobolev-Weighted Noise
    if config.get('apply_additive_sobolev_noise', False):
        noise_level_base = config.get('sobolev_noise_level_base', 0.01)
        sobolev_order_s = config.get('sobolev_order_s', 1.0) 

        # Generate noise with std=1 for the entire K_trunc x K_trunc block
        base_complex_noise = np.random.randn(K_trunc, K_trunc) + 1j * np.random.randn(K_trunc, K_trunc)
        
        # Create Sobolev weighting factor matrix
        # (1 + ||n||_2^2) for standard Sobolev, ||n||_2^2 = kx_eff^2 + ky_eff^2
        sobolev_denominators = (1 + kx_eff_grid**2 + ky_eff_grid**2)**(sobolev_order_s / 2.0)
        # Prevent division by zero for the denominator (though 1 + ||n||^2 is always >= 1)
        sobolev_denominators[sobolev_denominators < 1e-9] = 1.0 
        
        scaled_noise = noise_level_base * (base_complex_noise / sobolev_denominators)
        c_current = c_current + scaled_noise

    # 3. Phase Noise
    if config.get('apply_phase_noise', False):
        phase_noise_std = config.get('phase_noise_std_rad', 0.1) # In radians
        random_phases = np.random.randn(K_trunc, K_trunc) * phase_noise_std # Sample phases
        c_current = c_current * np.exp(1j * random_phases)

    # Normalize the final K_trunc x K_trunc spectrum so sum(|c_n|^2) = 1 over these modes.
    # This treats the truncated spectrum as the state vector in a truncated Hilbert space.
    norm_factor_sq = np.sum(np.abs(c_current)**2)
    if norm_factor_sq > 1e-12: # Avoid division by zero if spectrum is all zero
        c_current /= np.sqrt(norm_factor_sq)
    else: # If spectrum is zero (e.g. due to extreme attenuation), keep it zero
        c_current = np.zeros_like(c_current)
            
    return c_current


def generate_dataset(num_samples, N_grid_simulation, K_psi0_band_limit, K_trunc_spectrum, channel_config,
                     save_path="phenomenological_dataset.npz"):
    """
    Generates a dataset of (gamma_b_spec, gamma_a_spec) pairs.
    gamma_b_spec: Truncated spectrum of the initial state.
    gamma_a_spec: Truncated spectrum of the state after passing through the phenomenological noise channel.
    Both are K_trunc_spectrum x K_trunc_spectrum complex arrays.
    """
    dataset_gamma_b = []
    dataset_gamma_a = []
    
    print(f"Generating {num_samples} samples...")
    for i in range(num_samples):
        if (i+1) % (num_samples // 10 or 1) == 0:
            print(f"  Generating sample {i+1}/{num_samples}")

        # 1. Generate initial state psi_b
        psi_b_real = random_low_order_state(N_grid_simulation, K_max_band=K_psi0_band_limit)
        
        # 2. Get its truncated spectrum (this is gamma_b for the SNN)
        # This is already normalized in real space, sum(|psi_b_real|^2)=1
        # The FFT coefficients from get_truncated_spectrum are scaled by N_grid_simulation.
        # If SNN expects sum(|c_n|^2)=1 for input spectrum, we need to handle this.
        # Let's define gamma_b as the K_trunc x K_trunc block normalized such that sum(|coeffs|^2)=1.
        
        gamma_b_spec_unnorm = get_truncated_spectrum(psi_b_real, K_trunc_spectrum)
        norm_b_sq = np.sum(np.abs(gamma_b_spec_unnorm)**2)
        if norm_b_sq > 1e-12:
            gamma_b_spec = gamma_b_spec_unnorm / np.sqrt(norm_b_sq)
        else: # Should not happen if psi_b_real is normalized and K_trunc > 0
            gamma_b_spec = np.zeros_like(gamma_b_spec_unnorm)


        # 3. Apply phenomenological noise channel to this spectrum
        gamma_a_spec = apply_phenomenological_noise_channel(gamma_b_spec, channel_config)
        
        dataset_gamma_b.append(gamma_b_spec)
        dataset_gamma_a.append(gamma_a_spec)

    dataset_gamma_b_np = np.array(dataset_gamma_b, dtype=np.complex64)
    dataset_gamma_a_np = np.array(dataset_gamma_a, dtype=np.complex64)
    
    # Save the dataset
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    np.savez_compressed(save_path, gamma_b=dataset_gamma_b_np, gamma_a=dataset_gamma_a_np)
    print(f"Dataset saved to {save_path}")
    print(f"Shapes: gamma_b: {dataset_gamma_b_np.shape}, gamma_a: {dataset_gamma_a_np.shape}")
    
    return dataset_gamma_b_np, dataset_gamma_a_np

if __name__ == '__main__':
    # --- Configuration ---
    NUM_SAMPLES_TO_GENERATE = 1000
    N_GRID_FOR_SIMULATION = 64  # Grid size for initial real-space wavefunction
    K_PSI0_BAND_LIMIT = 12      # Max k for random_low_order_state (-K..K band)
    K_TRUNC_FOR_SPECTRA = 32    # Size of the KxK spectral block (e.g., 32x32)
                                # Ensure K_TRUNC_FOR_SPECTRA >= 2*K_PSI0_BAND_LIMIT + 1 ideally
                                # And K_TRUNC_FOR_SPECTRA <= N_GRID_FOR_SIMULATION

    # Noise Channel Configuration
    channel_noise_config = {
        'apply_attenuation': True,
        'attenuation_loss_factor': 0.2, # Higher means more loss for high-k

        'apply_additive_sobolev_noise': True,
        'sobolev_noise_level_base': 0.02, # Base std dev for k=0 mode noise
        'sobolev_order_s': 1.0,           # s=0: white noise, s>0: less high-freq noise

        'apply_phase_noise': True,
        'phase_noise_std_rad': 0.15       # Std dev of random phase shifts in radians
    }
    
    print("--- Phenomenological Channel Dataset Generation ---")
    print(f"Number of samples: {NUM_SAMPLES_TO_GENERATE}")
    print(f"Simulation Grid N: {N_GRID_FOR_SIMULATION}")
    print(f"Initial State K_band: {K_PSI0_BAND_LIMIT}")
    print(f"Truncated Spectrum K_trunc: {K_TRUNC_FOR_SPECTRA}")
    print("Channel Noise Configuration:")
    for key, value in channel_noise_config.items():
        print(f"  {key}: {value}")

    gamma_b_data, gamma_a_data = generate_dataset(
        NUM_SAMPLES_TO_GENERATE,
        N_GRID_FOR_SIMULATION,
        K_PSI0_BAND_LIMIT,
        K_TRUNC_FOR_SPECTRA,
        channel_noise_config,
        save_path="datasets/phenomenological_channel_dataset.npz"
    )

    # Optional: Visualize one sample
    if NUM_SAMPLES_TO_GENERATE > 0:
        sample_idx = 0
        gb_sample = gamma_b_data[sample_idx]
        ga_sample = gamma_a_data[sample_idx]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        im0 = axes[0].imshow(np.abs(gb_sample))
        axes[0].set_title(f"Sample $\gamma_b$ Spectrum (Mag) - Sample {sample_idx}")
        axes[0].set_xlabel("$k_y$ index (centered)")
        axes[0].set_ylabel("$k_x$ index (centered)")
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(np.abs(ga_sample))
        axes[1].set_title(f"Sample $\gamma_a$ Spectrum (Mag) - Sample {sample_idx}")
        axes[1].set_xlabel("$k_y$ index (centered)")
        axes[1].set_ylabel("$k_x$ index (centered)")
        plt.colorbar(im1, ax=axes[1])
        
        plt.tight_layout()
        os.makedirs("results_dataset_gen", exist_ok=True)
        plt.savefig("results_dataset_gen/sample_spectra_comparison.png")
        print("\nSample spectra visualization saved to results_dataset_gen/sample_spectra_comparison.png")
        plt.show()

