import numpy as np
import os
import matplotlib.pyplot as plt 
import argparse 
import torch # For GaussianRF

# --- GaussianRF Class (from user) ---
class GaussianRF(object):
    def __init__(self, dim, size, alpha=2, tau=3, sigma=None, boundary="periodic", device=None):
        self.dim = dim
        self.device = device if device is not None else torch.device("cpu")

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))

        k_max = size // 2

        if dim == 1:
            import math
            k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=self.device),
                           torch.arange(start=-k_max, end=0, step=1, device=self.device)), 0)
            self.sqrt_eig = (size * math.sqrt(2.0) * sigma *
                             ((4 * (math.pi**2) * (k**2) + tau**2) ** (-alpha/2.0)))
            if self.sqrt_eig.numel() > 0: self.sqrt_eig[0] = 0.0
        elif dim == 2:
            import math
            wavenumbers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=self.device),
                                      torch.arange(start=-k_max, end=0, step=1, device=self.device)), 0).repeat(size, 1)
            k_x = wavenumbers.transpose(0, 1)
            k_y = wavenumbers
            self.sqrt_eig = ((size**2) * math.sqrt(2.0) * sigma *
                             ((4*(math.pi**2)*(k_x**2 + k_y**2) + tau**2) ** (-alpha/2.0)))
            if self.sqrt_eig.numel() > 0 : self.sqrt_eig[0,0] = 0.0
        elif dim == 3:
            import math
            wavenumbers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=self.device),
                                      torch.arange(start=-k_max, end=0, step=1, device=self.device)), 0).repeat(size, size, 1)
            k_x = wavenumbers.transpose(1, 2); k_y = wavenumbers; k_z = wavenumbers.transpose(0, 2)
            self.sqrt_eig = ((size**3) * math.sqrt(2.0) * sigma *
                             ((4*(math.pi**2)*(k_x**2 + k_y**2 + k_z**2) + tau**2) ** (-alpha/2.0)))
            if self.sqrt_eig.numel() > 0: self.sqrt_eig[0,0,0] = 0.0
        else:
            raise ValueError("Dimension must be 1, 2, or 3.")
        self.size_tuple = tuple([size]*dim) # Renamed to avoid conflict with size argument

    def sample(self, N_samples):
        coeff = torch.randn(N_samples, *self.size_tuple, dtype=torch.cfloat, device=self.device)
        coeff = self.sqrt_eig * coeff
        return torch.fft.ifftn(coeff, dim=list(range(-self.dim, 0))).real

# --- Poisson Solver Functions (from user) ---
def solve_poisson_2d_rfft(f_hat_r: np.ndarray, enforce_zero_mean: bool = True) -> np.ndarray:
    N1, M = f_hat_r.shape  # M = N2//2 + 1
    u_hat_r = np.zeros_like(f_hat_r, dtype=np.complex128)
    if enforce_zero_mean and abs(f_hat_r[0, 0]) > 1e-14: f_hat_r[0, 0] = 0.0
    for k1_idx in range(N1):
        k1_shifted = k1_idx if k1_idx <= N1 // 2 else k1_idx - N1
        for k2_idx in range(M):
            k2_shifted = k2_idx 
            if k1_shifted == 0 and k2_shifted == 0: u_hat_r[k1_idx, k2_idx] = 0.0
            else:
                k_sq = k1_shifted**2 + k2_shifted**2
                if k_sq == 0: # Should be caught by above, but defensive
                    u_hat_r[k1_idx, k2_idx] = 0.0 
                else:
                    u_hat_r[k1_idx, k2_idx] = - f_hat_r[k1_idx, k2_idx] / k_sq
    return u_hat_r

def add_hierarchical_offset_2d(f: np.ndarray, K_max: int = 6, sigma: float = 0.5, rng: np.random.Generator = None) -> np.ndarray:
    if rng is None: rng = np.random.default_rng()
    Nx, Ny = f.shape
    x_vals = np.linspace(0, 2*np.pi, Nx, endpoint=False); y_vals = np.linspace(0, 2*np.pi, Ny, endpoint=False)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    n_points = 6; radius = np.pi / 2
    angles = [k * 2 * np.pi / n_points for k in range(n_points)]
    c_list = np.array([np.pi + radius * np.array([np.cos(angle), np.sin(angle)]) for angle in angles])
    center_indices = [rng.integers(low=0, high=len(c_list))] # Use rng.integers
    mu = np.zeros((Nx, Ny), dtype=float)
    for idx in center_indices:
        c_x, c_y = c_list[idx]
        dist_sq = (X - c_x)**2 + (Y - c_y)**2 # This assumes domain is [0, 2pi] for centers
        mu += np.exp(- dist_sq / (2*sigma**2)) * 0.5 # Amplitude of offset
    return f + mu

# --- Existing Helper Functions ---
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
    if K_trunc > N_grid_psi or K_trunc <= 0: raise ValueError(f"K_trunc {K_trunc} invalid for N_grid_psi {N_grid_psi}.")
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

def generate_dataset_main(num_samples, N_grid_simulation, 
                          K_psi0_band_limit, # For random_low_order_state or GRF base for Poisson
                          K_trunc_snn, K_trunc_full_eval, 
                          pde_type,
                          channel_config, # Used if pde_type is 'phenomenological_channel'
                          grf_config,     # Used if pde_type is 'poisson'
                          save_path_template,
                          filename_suffix_str):
    dataset_gamma_b_Nmax = []
    dataset_gamma_a_Nmax_true = []
    dataset_gamma_a_Nfull_true = []
    
    print(f"Generating {num_samples} samples for PDE type: {pde_type}...")
    if pde_type == "poisson":
        grf_generator = GaussianRF(dim=2, size=N_grid_simulation, 
                                   alpha=grf_config['alpha'], tau=grf_config['tau'], 
                                   device=torch.device("cpu")) # GRF on CPU

    for i in range(num_samples):
        if (i+1) % (num_samples // 10 or 1) == 0:
            print(f"  Generating sample {i+1}/{num_samples}")

        if pde_type == "phenomenological_channel":
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

        elif pde_type == "poisson":
            # 1. Generate source term f_real
            f_sample_grf = grf_generator.sample(1).cpu().numpy().squeeze()
            # add_hierarchical_offset_2d assumes domain [0, 2pi] for centers.
            # If GRF is on a different domain, this might need adjustment or f_sample_grf rescaling.
            # For now, assume N_grid_simulation corresponds to a [0, 2pi]^2 domain for offset.
            f_real_with_offset = add_hierarchical_offset_2d(f_sample_grf, 
                                                            sigma=grf_config.get('offset_sigma', 0.5))
            f_real = f_real_with_offset - np.mean(f_real_with_offset) # Enforce zero mean

            # 2. gamma_b is from f_real
            gamma_b_Nmax_spec = get_truncated_spectrum(f_real, K_trunc_snn)
            dataset_gamma_b_Nmax.append(gamma_b_Nmax_spec)

            # 3. Solve Poisson: u from f
            f_hat_r = np.fft.rfft2(f_real)
            u_hat_r = solve_poisson_2d_rfft(f_hat_r, enforce_zero_mean=True)
            u_real = np.fft.irfft2(u_hat_r, s=f_real.shape) # Ensure original shape

            # 4. gamma_a is from u_real
            gamma_a_Nmax_spec = get_truncated_spectrum(u_real, K_trunc_snn)
            dataset_gamma_a_Nmax_true.append(gamma_a_Nmax_spec)

            if K_trunc_full_eval != K_trunc_snn:
                gamma_a_Nfull_spec = get_truncated_spectrum(u_real, K_trunc_full_eval)
                dataset_gamma_a_Nfull_true.append(gamma_a_Nfull_spec)
            else:
                dataset_gamma_a_Nfull_true.append(gamma_a_Nmax_spec)
        else:
            raise ValueError(f"Unknown pde_type: {pde_type}")

    gamma_b_Nmax_all = np.array(dataset_gamma_b_Nmax, dtype=np.complex64)
    gamma_a_Nmax_true_all = np.array(dataset_gamma_a_Nmax_true, dtype=np.complex64)
    gamma_a_Nfull_true_all = np.array(dataset_gamma_a_Nfull_true, dtype=np.complex64)
    
    current_save_path = save_path_template.format(Nmax=K_trunc_snn, Nfull=K_trunc_full_eval, pde_type=pde_type, noise_str=filename_suffix_str)
    os.makedirs(os.path.dirname(current_save_path) or '.', exist_ok=True)
    np.savez_compressed(current_save_path, 
                        gamma_b_Nmax=gamma_b_Nmax_all, 
                        gamma_a_Nmax_true=gamma_a_Nmax_true_all, 
                        gamma_a_Nfull_true=gamma_a_Nfull_true_all)
    print(f"Dataset saved to {current_save_path}")
    print(f"Shapes: gamma_b (Nmax): {gamma_b_Nmax_all.shape}, gamma_a (Nmax_true): {gamma_a_Nmax_true_all.shape}, gamma_a (Nfull_true): {gamma_a_Nfull_true_all.shape}")
    
    return gamma_b_Nmax_all, gamma_a_Nmax_true_all, gamma_a_Nfull_true_all

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate dataset (phenomenological noise or PDE-based) for SNN training.")
    parser.add_argument('--pde_type', type=str, default="phenomenological_channel", 
                        choices=["phenomenological_channel", "poisson"], help="Type of data generation process.")
    
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--n_grid_sim', type=int, default=64)
    parser.add_argument('--k_psi0_limit', type=int, default=12, help='Max k for initial random state (phenomenological) or GRF base (Poisson).')
    parser.add_argument('--k_trunc_snn', type=int, default=32)
    parser.add_argument('--k_trunc_full', type=int, default=32) 
    parser.add_argument('--output_dir', type=str, default="datasets")

    # Phenomenological Channel Noise Parameters
    parser.add_argument('--apply_attenuation', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--attenuation_loss_factor', type=float, default=0.2)
    parser.add_argument('--apply_additive_sobolev_noise', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--sobolev_noise_level_base', type=float, default=0.01)
    parser.add_argument('--sobolev_order_s', type=float, default=1.0)
    parser.add_argument('--apply_phase_noise', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--phase_noise_std_rad', type=float, default=0.05)

    # GRF parameters (for Poisson source f)
    parser.add_argument('--grf_alpha', type=float, default=2.5, help="GRF alpha for Poisson source.")
    parser.add_argument('--grf_tau', type=float, default=7.0, help="GRF tau for Poisson source.")
    parser.add_argument('--grf_offset_sigma', type=float, default=0.5, help="Sigma for hierarchical offset in Poisson source.")
    
    args = parser.parse_args()

    if args.k_trunc_snn > args.n_grid_sim: raise ValueError("K_TRUNC_SNN > N_GRID_SIM.")
    if args.k_trunc_full < args.k_trunc_snn: args.k_trunc_full = args.k_trunc_snn
    if args.n_grid_sim < args.k_trunc_full: raise ValueError("N_GRID_SIM must be >= K_TRUNC_FULL.")

    filename_suffix = ""
    if args.pde_type == "phenomenological_channel":
        noise_parts = []
        if args.apply_attenuation: noise_parts.append(f"att{args.attenuation_loss_factor:.2f}")
        if args.apply_additive_sobolev_noise: noise_parts.append(f"sob{args.sobolev_noise_level_base:.3f}s{args.sobolev_order_s:.1f}")
        if args.apply_phase_noise: noise_parts.append(f"ph{args.phase_noise_std_rad:.2f}")
        filename_suffix = "_".join(noise_parts) if noise_parts else "no_noise"
    elif args.pde_type == "poisson":
        filename_suffix = f"poisson_grfA{args.grf_alpha:.1f}T{args.grf_tau:.1f}"


    print(f"--- Dataset Generation: PDE Type '{args.pde_type}' ---")
    print(f"Filename suffix: {filename_suffix}")

    channel_config_for_phenom = {
        'apply_attenuation': args.apply_attenuation,
        'attenuation_loss_factor': args.attenuation_loss_factor,
        'apply_additive_sobolev_noise': args.apply_additive_sobolev_noise,
        'sobolev_noise_level_base': args.sobolev_noise_level_base,
        'sobolev_order_s': args.sobolev_order_s,
        'apply_phase_noise': args.apply_phase_noise,
        'phase_noise_std_rad': args.phase_noise_std_rad
    }
    grf_config_for_poisson = {
        'alpha': args.grf_alpha,
        'tau': args.grf_tau,
        'offset_sigma': args.grf_offset_sigma
    }
    
    # Filename template now includes pde_type and the specific noise/source string
    filename_template = os.path.join(args.output_dir, "dataset_{pde_type}_Nmax{Nmax}_Nfull{Nfull}_{noise_str}.npz")

    gamma_b_data, gamma_a_Nmax_data, gamma_a_Nfull_data = generate_dataset_main(
        args.num_samples, args.n_grid_sim, args.k_psi0_limit,
        args.k_trunc_snn, args.k_trunc_full, 
        args.pde_type,
        channel_config_for_phenom, 
        grf_config_for_poisson,
        save_path_template=filename_template,
        filename_suffix_str=filename_suffix
    )

    if args.num_samples > 0 and gamma_b_data.size > 0:
        sample_idx = 0
        gb_Nmax_sample = gamma_b_data[sample_idx]
        ga_Nmax_true_sample = gamma_a_Nmax_data[sample_idx] 
        ga_Nfull_true_sample = gamma_a_Nfull_data[sample_idx] # Corrected index

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        im0 = axes[0].imshow(np.abs(gb_Nmax_sample)); axes[0].set_title(f"Sample $\gamma_b$ ($N_{{max}}={args.k_trunc_snn}$)"); plt.colorbar(im0, ax=axes[0])
        im1 = axes[1].imshow(np.abs(ga_Nmax_true_sample)); axes[1].set_title(f"Sample $\gamma_a$ ($N_{{max}}={args.k_trunc_snn}$)"); plt.colorbar(im1, ax=axes[1])
        im2 = axes[2].imshow(np.abs(ga_Nfull_true_sample)); axes[2].set_title(f"Sample $\gamma_a$ ($N_{{full}}={args.k_trunc_full}$)"); plt.colorbar(im2, ax=axes[2])
        fig.suptitle(f"Dataset Sample for PDE: {args.pde_type.upper()} (Noise/Source: {filename_suffix})", fontsize=14)
        plt.tight_layout(rect=[0,0,1,0.95])
        vis_dir = "results_dataset_gen_pde"
        os.makedirs(vis_dir, exist_ok=True)
        save_fig_path = os.path.join(vis_dir, f"sample_spectra_{args.pde_type}_{filename_suffix}.png")
        plt.savefig(save_fig_path)
        print(f"\nSample spectra visualization saved to {save_fig_path}")
    else:
        print("No samples generated or data is empty, skipping visualization.")

