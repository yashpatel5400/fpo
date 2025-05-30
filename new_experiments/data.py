import numpy as np
import os
import matplotlib.pyplot as plt 
import argparse 
import torch # For GaussianRF

# --- GaussianRF Class (from user) ---
class GaussianRF(object):
    def __init__(self, dim, size, alpha=2, tau=3, sigma=None, boundary="periodic", device=None):
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
            
            wavenumbers = wavenumbers_half.repeat(size,1) # This creates [size, size] if k_max=0 and wavenumbers_half=[0]

            k_x = wavenumbers.transpose(0, 1)
            k_y = wavenumbers
            
            self.sqrt_eig = ((size**2)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2+k_y**2)+tau**2)**(-alpha/2.0)))
            if self.sqrt_eig.numel() > 0 and k_max > 0 : 
                self.sqrt_eig[0,0] = 0.0 # DC component of the full spectrum
        else:
            raise ValueError("Dimension must be 1 or 2 for this GRF implementation.")
        
        self.size_tuple = tuple([size]*dim)

    def sample(self, N_samples):
        if self.sqrt_eig.numel() == 0 and self.size_tuple[0] > 0 : 
             # This can happen if size is 1, k_max is 0, and dim=1 k construction leads to empty sqrt_eig
             # or if k_max = 0 for dim=2, sqrt_eig might be scalar 0.
             # Should return zeros of appropriate shape if no spectral components.
            print(f"Warning: sqrt_eig is empty or scalar zero for GRF. size_tuple: {self.size_tuple}. Returning zeros.")
            return torch.zeros(N_samples, *self.size_tuple, dtype=torch.float32) 
            
        coeff = torch.randn(N_samples, *self.size_tuple, dtype=torch.cfloat, device=self.device)
        # Element-wise multiplication; sqrt_eig needs to broadcast or match shape
        if self.dim == 1 and N_samples > 1 and self.sqrt_eig.ndim == 1:
            coeff = self.sqrt_eig.unsqueeze(0) * coeff
        else:
            coeff = self.sqrt_eig * coeff # Assumes broadcasting for N_samples > 1
            
        return torch.fft.ifftn(coeff, dim=list(range(-self.dim, 0))).real

# --- Poisson Solver Functions (from user) ---
def solve_poisson_2d_rfft(f_hat_r: np.ndarray, enforce_zero_mean: bool = True) -> np.ndarray:
    N1, M = f_hat_r.shape
    u_hat_r = np.zeros_like(f_hat_r, dtype=np.complex128)
    
    if enforce_zero_mean:
        if abs(f_hat_r[0, 0]) > 1e-14:
            f_hat_r[0, 0] = 0.0
            
    for k1_idx in range(N1):
        if k1_idx <= N1 // 2:
            k1_shifted = k1_idx
        else:
            k1_shifted = k1_idx - N1

        for k2_idx in range(M):
            k2_shifted = k2_idx 
            
            if k1_shifted == 0 and k2_shifted == 0:
                u_hat_r[k1_idx, k2_idx] = 0.0
            else:
                k_sq = k1_shifted**2 + k2_shifted**2
                if k_sq < 1e-12: 
                    u_hat_r[k1_idx, k2_idx] = 0.0 
                else:
                    u_hat_r[k1_idx, k2_idx] = - f_hat_r[k1_idx, k2_idx] / k_sq
    return u_hat_r

def add_hierarchical_offset_2d(f: np.ndarray, K_max: int = 6, sigma: float = 0.5, rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
        
    Nx, Ny = f.shape
    x_vals = np.linspace(0, 2*np.pi, Nx, endpoint=False)
    y_vals = np.linspace(0, 2*np.pi, Ny, endpoint=False)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

    n_points = 6
    radius = np.pi / 2
    angles = [k * 2 * np.pi / n_points for k in range(n_points)]
    c_list = np.array([np.pi + radius * np.array([np.cos(angle), np.sin(angle)]) for angle in angles])
    
    center_indices = [rng.integers(low=0, high=len(c_list))] 
    
    mu = np.zeros((Nx, Ny), dtype=float)
    for idx in center_indices:
        c_x, c_y = c_list[idx]
        dist_sq = (X - c_x)**2 + (Y - c_y)**2 
        mu += np.exp(- dist_sq / (2*sigma**2)) * 0.5 
        
    return f + mu

# --- Helper Functions for Spectra ---
def get_full_centered_spectrum(psi_real_space):
    """ Computes full 2D FFT and shifts it. Output is NOT normalized here."""
    N_grid = psi_real_space.shape[0]
    if N_grid == 0:
        return np.array([], dtype=np.complex64)
    F_psi_shifted = np.fft.fftshift(np.fft.fft2(psi_real_space))
    return F_psi_shifted

def normalize_spectrum(spectrum_complex):
    """ Normalizes a given spectrum so sum(|coeffs|^2)=1. """
    if spectrum_complex.size == 0:
        return spectrum_complex
    norm_sq = np.sum(np.abs(spectrum_complex)**2)
    if norm_sq > 1e-14:
        return spectrum_complex / np.sqrt(norm_sq)
    return np.zeros_like(spectrum_complex)

def extract_center_block(full_spectrum_centered, K_extract):
    """ Extracts central K_extract x K_extract block. Output is NOT normalized here."""
    N_full = full_spectrum_centered.shape[0]
    if K_extract == N_full:
        return full_spectrum_centered
    if K_extract > N_full:
        raise ValueError(f"K_extract ({K_extract}) cannot be larger than full_spectrum size ({N_full}) for simple extraction.")
    if K_extract <= 0:
        return np.array([], dtype=np.complex64)
    
    start_idx = N_full // 2 - K_extract // 2
    end_idx = start_idx + K_extract
    extracted_block = full_spectrum_centered[start_idx:end_idx, start_idx:end_idx]
    return extracted_block

def apply_phenomenological_noise_channel(spectrum_b_full_normalized, config_channel_noise):
    """ Applies noise to a FULL (N_grid x N_grid) normalized spectrum. Output is also full and normalized. """
    c_current = spectrum_b_full_normalized.copy()
    N_grid = c_current.shape[0] 
    if N_grid == 0:
        return c_current
        
    if N_grid % 2 == 0:
        k_vals_one_dim_eff = np.arange(-N_grid//2, N_grid//2)
    else:
        k_vals_one_dim_eff = np.arange(-(N_grid-1)//2, (N_grid-1)//2 + 1)
    kx_eff_grid, ky_eff_grid = np.meshgrid(k_vals_one_dim_eff, k_vals_one_dim_eff, indexing='ij')

    if config_channel_noise.get('apply_attenuation', False):
        loss_factor = config_channel_noise.get('attenuation_loss_factor', 0.1) 
        norm_sq_grid = kx_eff_grid**2 + ky_eff_grid**2
        max_norm_sq_in_grid = np.max(norm_sq_grid) if N_grid > 1 else 1.0
        if max_norm_sq_in_grid < 1e-9: 
            max_norm_sq_in_grid = 1.0 
        attenuation_profile = np.exp(-loss_factor * norm_sq_grid / max_norm_sq_in_grid)
        c_current = c_current * attenuation_profile
        
    if config_channel_noise.get('apply_additive_sobolev_noise', False):
        noise_level_base = config_channel_noise.get('sobolev_noise_level_base', 0.01)
        sobolev_order_s = config_channel_noise.get('sobolev_order_s', 1.0) 
        base_complex_noise = np.random.randn(N_grid, N_grid) + 1j * np.random.randn(N_grid, N_grid)
        sobolev_denominators = (1 + kx_eff_grid**2 + ky_eff_grid**2)**(sobolev_order_s / 2.0)
        sobolev_denominators[sobolev_denominators < 1e-9] = 1.0 
        scaled_noise = noise_level_base * (base_complex_noise / sobolev_denominators)
        c_current = c_current + scaled_noise
        
    if config_channel_noise.get('apply_phase_noise', False):
        phase_noise_std = config_channel_noise.get('phase_noise_std_rad', 0.1) 
        random_phases = np.random.randn(N_grid, N_grid) * phase_noise_std 
        c_current = c_current * np.exp(1j * random_phases)
            
    # Re-normalize after all noise applications
    c_current = normalize_spectrum(c_current)
    return c_current

def generate_snn_dataset(num_samples, N_grid_sim_input, 
                         K_psi0_band_limit, # Used by random_low_order_state
                         K_trunc_snn_output, 
                         pde_type,
                         channel_config, 
                         grf_config,     
                         save_path_template,
                         filename_suffix_str):
    dataset_gamma_b_full_input = []
    dataset_gamma_a_snn_target = []
    dataset_gamma_a_true_full_output = [] 
    
    print(f"Generating {num_samples} samples for SNN. PDE: {pde_type}...")
    print(f"  Input resolution (gamma_b_full_input & gamma_a_true_full_output): {N_grid_sim_input}x{N_grid_sim_input}")
    print(f"  SNN Target output resolution (gamma_a_snn_target): {K_trunc_snn_output}x{K_trunc_snn_output}")

    grf_device = torch.device("cpu") # GRF on CPU for numpy compatibility later
    grf_generator = GaussianRF(dim=2, size=N_grid_sim_input, 
                               alpha=grf_config['alpha'], tau=grf_config['tau'], 
                               device=grf_device)

    for i in range(num_samples):
        if (i+1) % (num_samples // 10 or 1) == 0:
            print(f"  Generating sample {i+1}/{num_samples}")

        initial_real_space_state = None 
        gamma_b_full_spec_for_snn_input = None
        gamma_a_true_full_spec_for_output = None 

        if pde_type == "phenomenological_channel":
            initial_real_space_state = grf_generator.sample(1).cpu().numpy().squeeze()
            # For phenomenological channel, input state is normalized wavefunc
            norm_initial = np.linalg.norm(initial_real_space_state)
            if norm_initial > 1e-9: initial_real_space_state /= norm_initial
            else: initial_real_space_state.flat[0]=1.0; initial_real_space_state /= np.linalg.norm(initial_real_space_state)


            gamma_b_full_spec_unnorm = get_full_centered_spectrum(initial_real_space_state)
            gamma_b_full_spec_for_snn_input = normalize_spectrum(gamma_b_full_spec_unnorm) 
            
            gamma_a_true_full_spec_for_output = apply_phenomenological_noise_channel(
                gamma_b_full_spec_for_snn_input, channel_config
            ) 
        
        elif pde_type == "poisson":
            f_sample_grf = grf_generator.sample(1).cpu().numpy().squeeze()
            f_real_with_offset = add_hierarchical_offset_2d(f_sample_grf, sigma=grf_config.get('offset_sigma', 0.5))
            initial_real_space_state_f = f_real_with_offset - np.mean(f_real_with_offset) 
            
            gamma_b_full_spec_for_snn_input = get_full_centered_spectrum(initial_real_space_state_f) # Unnormalized spectrum of f

            f_hat_r = np.fft.rfft2(initial_real_space_state_f)
            u_hat_r = solve_poisson_2d_rfft(f_hat_r, enforce_zero_mean=True)
            true_output_real_space_state_u = np.fft.irfft2(u_hat_r, s=initial_real_space_state_f.shape) 
            
            gamma_a_true_full_spec_for_output = get_full_centered_spectrum(true_output_real_space_state_u) # Unnormalized spectrum of u
        else:
            raise ValueError(f"Unknown pde_type: {pde_type}")

        dataset_gamma_b_full_input.append(gamma_b_full_spec_for_snn_input)
        dataset_gamma_a_true_full_output.append(gamma_a_true_full_spec_for_output)
        
        gamma_a_snn_target_spec = extract_center_block(gamma_a_true_full_spec_for_output, K_trunc_snn_output) # Not normalized after truncation
        dataset_gamma_a_snn_target.append(gamma_a_snn_target_spec)

    gamma_b_full_all = np.array(dataset_gamma_b_full_input, dtype=np.complex64)
    gamma_a_snn_target_all = np.array(dataset_gamma_a_snn_target, dtype=np.complex64)
    gamma_a_true_full_all = np.array(dataset_gamma_a_true_full_output, dtype=np.complex64)
    
    current_save_path = save_path_template.format(Nin=N_grid_sim_input, Nout=K_trunc_snn_output, pde_type=pde_type, noise_str=filename_suffix_str)
    os.makedirs(os.path.dirname(current_save_path) or '.', exist_ok=True)
    np.savez_compressed(current_save_path, 
                        gamma_b_full_input=gamma_b_full_all, 
                        gamma_a_snn_target=gamma_a_snn_target_all, 
                        gamma_a_true_full_output=gamma_a_true_full_all) 
    print(f"Dataset saved to {current_save_path}")
    print(f"Shapes: gamma_b_full_input: {gamma_b_full_all.shape}, gamma_a_snn_target: {gamma_a_snn_target_all.shape}, gamma_a_true_full_output: {gamma_a_true_full_all.shape}")
    
    return gamma_b_full_all, gamma_a_snn_target_all, gamma_a_true_full_all

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate dataset (Full Input -> Truncated Output, with Full True Output) for SNN training.")
    parser.add_argument('--pde_type', type=str, default="phenomenological_channel", choices=["phenomenological_channel", "poisson"])
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--n_grid_sim_input', type=int, default=64)
    parser.add_argument('--k_psi0_limit', type=int, default=12, help="Max k for initial random_low_order_state (phenomenological).")
    parser.add_argument('--k_trunc_snn_output', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default="datasets")

    # Phenomenological Channel Noise Parameters
    parser.add_argument('--apply_attenuation', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--attenuation_loss_factor', type=float, default=0.2)
    parser.add_argument('--apply_additive_sobolev_noise', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--sobolev_noise_level_base', type=float, default=0.01)
    parser.add_argument('--sobolev_order_s', type=float, default=1.0)
    parser.add_argument('--apply_phase_noise', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--phase_noise_std_rad', type=float, default=0.05)
    
    # GRF parameters (for Poisson source f, AND optionally for phenomenological_channel input)
    parser.add_argument('--grf_alpha', type=float, default=4.0) 
    parser.add_argument('--grf_tau', type=float, default=1.0)   
    parser.add_argument('--grf_offset_sigma', type=float, default=0.5, help="Sigma for hierarchical offset in Poisson source.")
    parser.add_argument('--use_grf_for_phenom_input', action=argparse.BooleanOptionalAction, default=True, # New flag
                        help="If pde_type is phenomenological_channel, use GRF for initial state instead of random_low_order_state.")

    args = parser.parse_args()

    if args.k_trunc_snn_output > args.n_grid_sim_input:
        raise ValueError("K_TRUNC_SNN_OUTPUT cannot be larger than N_GRID_SIM_INPUT.")

    filename_suffix = ""
    if args.pde_type == "phenomenological_channel":
        noise_parts = []
        if args.use_grf_for_phenom_input: # Add GRF info to suffix if used for phenom. channel
            noise_parts.append(f"grfInA{args.grf_alpha:.1f}T{args.grf_tau:.1f}")
        if args.apply_attenuation: noise_parts.append(f"att{args.attenuation_loss_factor:.2f}")
        if args.apply_additive_sobolev_noise: noise_parts.append(f"sob{args.sobolev_noise_level_base:.3f}s{args.sobolev_order_s:.1f}")
        if args.apply_phase_noise: noise_parts.append(f"ph{args.phase_noise_std_rad:.2f}")
        filename_suffix = "_".join(noise_parts) if noise_parts else "no_noise_or_grf_input"
    elif args.pde_type == "poisson":
        filename_suffix = f"poisson_grfA{args.grf_alpha:.1f}T{args.grf_tau:.1f}"
    
    print(f"--- Dataset Generation: PDE Type '{args.pde_type}' (Full Input -> Truncated Output, with Full True Output) ---")
    print(f"Filename suffix: {filename_suffix}")

    channel_config_for_phenom = {
        'apply_attenuation': args.apply_attenuation, 'attenuation_loss_factor': args.attenuation_loss_factor,
        'apply_additive_sobolev_noise': args.apply_additive_sobolev_noise, 
        'sobolev_noise_level_base': args.sobolev_noise_level_base, 'sobolev_order_s': args.sobolev_order_s,
        'apply_phase_noise': args.apply_phase_noise, 'phase_noise_std_rad': args.phase_noise_std_rad
    }
    grf_config_for_pde = { # Renamed for clarity, used by both Poisson and optionally by Phenom.
        'alpha': args.grf_alpha, 'tau': args.grf_tau, 
        'offset_sigma': args.grf_offset_sigma,
        'use_grf_for_phenom_input': args.use_grf_for_phenom_input # Pass this flag
    }
    
    filename_template = os.path.join(args.output_dir, "dataset_{pde_type}_Nin{Nin}_Nout{Nout}_{noise_str}.npz")

    gamma_b_data, gamma_a_snn_target_data, gamma_a_true_full_data = generate_snn_dataset(
        args.num_samples, args.n_grid_sim_input, args.k_psi0_limit,
        args.k_trunc_snn_output, 
        args.pde_type,
        channel_config_for_phenom, 
        grf_config_for_pde, # Pass the GRF config
        save_path_template=filename_template,
        filename_suffix_str=filename_suffix
    )

    if args.num_samples > 0 and gamma_b_data.size > 0:
        sample_idx = 0
        gb_full_sample_spec = gamma_b_data[sample_idx]
        ga_snn_target_sample_spec = gamma_a_snn_target_data[sample_idx]
        ga_true_full_sample_spec = gamma_a_true_full_data[sample_idx]

        fig_spec, axes_spec = plt.subplots(1, 3, figsize=(18, 5)) 
        im0_spec = axes_spec[0].imshow(np.abs(gb_full_sample_spec))
        axes_spec[0].set_title(f"Input $\gamma_b$ Spectrum ($N_{{in}}={args.n_grid_sim_input}$)")
        plt.colorbar(im0_spec, ax=axes_spec[0])
        im1_spec = axes_spec[1].imshow(np.abs(ga_snn_target_sample_spec))
        axes_spec[1].set_title(f"Target $\gamma_a$ Spectrum ($N_{{out}}={args.k_trunc_snn_output}$)")
        plt.colorbar(im1_spec, ax=axes_spec[1])
        im2_spec = axes_spec[2].imshow(np.abs(ga_true_full_sample_spec))
        axes_spec[2].set_title(f"True Full $\gamma_a$ Spectrum ($N_{{in}}={args.n_grid_sim_input}$)")
        plt.colorbar(im2_spec, ax=axes_spec[2])
        fig_spec.suptitle(f"Dataset Sample Spectra (PDE: {args.pde_type.upper()}, Params: {filename_suffix})", fontsize=14)
        plt.tight_layout(rect=[0,0,1,0.93])
        vis_dir = "results_dataset_gen_snn_format_v3" 
        os.makedirs(vis_dir, exist_ok=True)
        save_fig_path_spec = os.path.join(vis_dir, f"sample_snn_spectra_v3_{args.pde_type}_{filename_suffix}.png")
        plt.savefig(save_fig_path_spec)
        print(f"\nSample SNN spectra visualization saved to {save_fig_path_spec}")
        plt.close(fig_spec) 

        psi_b_real_vis = np.fft.ifft2(np.fft.ifftshift(gb_full_sample_spec))
        psi_a_true_full_spatial_vis = np.fft.ifft2(np.fft.ifftshift(ga_true_full_sample_spec))
        padded_target_spec = np.zeros((args.n_grid_sim_input, args.n_grid_sim_input), dtype=np.complex64)
        start_idx = args.n_grid_sim_input // 2 - args.k_trunc_snn_output // 2
        end_idx = start_idx + args.k_trunc_snn_output
        padded_target_spec[start_idx:end_idx, start_idx:end_idx] = ga_snn_target_sample_spec
        psi_a_snn_target_spatial_vis = np.fft.ifft2(np.fft.ifftshift(padded_target_spec))
        fig_spatial, axes_spatial = plt.subplots(1, 3, figsize=(18, 5))
        plot_func = np.abs if args.pde_type == "phenomenological_channel" else lambda x: x.real
        im0_spatial = axes_spatial[0].imshow(plot_func(psi_b_real_vis))
        axes_spatial[0].set_title(f"Input $\gamma_b$ Spatial ($N_{{in}}={args.n_grid_sim_input}$)")
        plt.colorbar(im0_spatial, ax=axes_spatial[0])
        im1_spatial = axes_spatial[1].imshow(plot_func(psi_a_snn_target_spatial_vis))
        axes_spatial[1].set_title(f"SNN Target $\gamma_a$ Spatial (from $N_{{out}}$)")
        plt.colorbar(im1_spatial, ax=axes_spatial[1])
        im2_spatial = axes_spatial[2].imshow(plot_func(psi_a_true_full_spatial_vis))
        axes_spatial[2].set_title(f"True Full $\gamma_a$ Spatial ($N_{{in}}={args.n_grid_sim_input}$)")
        plt.colorbar(im2_spatial, ax=axes_spatial[2])
        fig_spatial.suptitle(f"Dataset Sample Spatial (PDE: {args.pde_type.upper()}, Params: {filename_suffix})", fontsize=14)
        plt.tight_layout(rect=[0,0,1,0.93])
        save_fig_path_spatial = os.path.join(vis_dir, f"sample_snn_spatial_v3_{args.pde_type}_{filename_suffix}.png")
        plt.savefig(save_fig_path_spatial)
        print(f"Sample SNN spatial visualization saved to {save_fig_path_spatial}")
        plt.close(fig_spatial)
    else:
        print("No samples generated or data is empty, skipping visualization.")

