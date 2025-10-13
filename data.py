import numpy as np
import os
import matplotlib.pyplot as plt 
import argparse 
import torch # For GaussianRF

# --- Global constants for solver (can be moved to args if needed) ---
HBAR_CONST = 1.0
MASS_CONST = 1.0

# --- GaussianRF Class ---
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

        if dim == 1:
            import math
            if k_max > 0:
                k_range = torch.arange(start=0, end=k_max, step=1, device=self.device)
                k = torch.cat((k_range, torch.arange(start=-k_max, end=0, step=1, device=self.device)), 0)
            else: 
                k = torch.tensor([0], device=self.device)
            
            self.sqrt_eig = (size * math.sqrt(2.0) * sigma * ((4 * (math.pi**2) * (k**2) + tau**2) ** (-alpha/2.0)))
            
            if self.sqrt_eig.numel() > 0 and k_max > 0 : 
                self.sqrt_eig[0] = 0.0
            elif self.sqrt_eig.numel() > 0 and k_max == 0 :
                pass 

        elif dim == 2:
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
            raise ValueError("Dimension must be 1 or 2 for this GRF implementation.")
        
        self.size_tuple = tuple([size]*dim)

    def sample(self, N_samples):
        if self.sqrt_eig.numel() == 0 and self.size_tuple[0] > 0 : 
            print(f"Warning: sqrt_eig is empty for GRF. size_tuple: {self.size_tuple}. Returning zeros.")
            return torch.zeros(N_samples, *self.size_tuple, dtype=torch.float32) 
            
        coeff = torch.randn(N_samples, *self.size_tuple, dtype=torch.cfloat, device=self.device)
        if self.sqrt_eig.numel() > 0 : 
            if self.dim == 1 and N_samples > 1 and self.sqrt_eig.ndim == 1:
                coeff = self.sqrt_eig.unsqueeze(0) * coeff
            else:
                coeff = self.sqrt_eig * coeff 
            
        return torch.fft.ifftn(coeff, dim=list(range(-self.dim, 0))).real

# --- Poisson Solver Functions ---
def solve_poisson_2d_rfft(f_hat_r: np.ndarray, enforce_zero_mean: bool = True) -> np.ndarray:
    N1, M_rfft = f_hat_r.shape
    u_hat_r = np.zeros_like(f_hat_r, dtype=np.complex128)
    
    if enforce_zero_mean:
        if abs(f_hat_r[0, 0]) > 1e-14:
            f_hat_r[0, 0] = 0.0
            
    for k1_idx in range(N1):
        if k1_idx <= N1 // 2:
            k1_shifted = k1_idx
        else:
            k1_shifted = k1_idx - N1

        for k2_idx in range(M_rfft):
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

# --- Schrodinger Solvers ---
def split_step_solver_2d(V_grid, psi0, N, dx, T, num_steps, hbar_val, m_val):
    dt = T/num_steps
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

# --- Heat Equation Solver ---
def solve_heat_equation_2d(u0, N, L, T, nu):
    u0_hat = np.fft.fft2(u0)
    dx = L / N
    k_vec = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    kx, ky = np.meshgrid(k_vec, k_vec, indexing='ij')
    k2 = kx**2 + ky**2
    uT_hat = u0_hat * np.exp(-nu * k2 * T)
    uT = np.fft.ifft2(uT_hat).real
    return uT

# --- Waveguide Potential ---
def get_step_index_fiber_potential(N_grid, L_domain, core_radius_factor, potential_depth):
    actual_core_radius = core_radius_factor * (L_domain / 2.0)
    coords1d = np.linspace(-L_domain / 2.0, L_domain / 2.0, N_grid, endpoint=False)
    x_mg, y_mg = np.meshgrid(coords1d, coords1d, indexing='ij')
    rr = np.sqrt(x_mg**2 + y_mg**2)
    potential = np.zeros((N_grid, N_grid), dtype=float)
    potential[rr < actual_core_radius] = -potential_depth 
    return potential

def get_grin_fiber_potential(N_grid, L_domain, grin_strength):
    coords1d = np.linspace(-L_domain / 2.0, L_domain / 2.0, N_grid, endpoint=False)
    x_mg, y_mg = np.meshgrid(coords1d, coords1d, indexing='ij')
    rr_sq = x_mg**2 + y_mg**2
    potential = grin_strength * rr_sq
    return potential

# --- Helper Functions for Spectra ---
def get_full_centered_spectrum(psi_real_space): 
    N_grid = psi_real_space.shape[0]
    if N_grid == 0:
        return np.array([], dtype=np.complex64)
    F_psi_shifted = np.fft.fftshift(np.fft.fft2(psi_real_space))
    return F_psi_shifted

def normalize_spectrum(spectrum_complex): 
    if spectrum_complex.size == 0:
        return spectrum_complex
    norm_sq = np.sum(np.abs(spectrum_complex)**2)
    if norm_sq > 1e-14:
        return spectrum_complex / np.sqrt(norm_sq)
    return np.zeros_like(spectrum_complex)

def extract_center_block(full_spectrum_centered, K_extract): 
    N_full = full_spectrum_centered.shape[0]
    if K_extract == N_full:
        return full_spectrum_centered
    if K_extract > N_full:
        raise ValueError(f"K_extract ({K_extract}) > full_spectrum size ({N_full}). Padding not implemented here.")
    if K_extract <= 0:
        return np.array([], dtype=np.complex64)
    
    start_idx = N_full // 2 - K_extract // 2
    end_idx = start_idx + K_extract
    return full_spectrum_centered[start_idx:end_idx, start_idx:end_idx]

# --- Main Data Generation Function ---
def generate_snn_dataset(num_samples, N_grid_sim_input, K_psi0_band_limit, 
                         K_trunc_snn_output, 
                         pde_type,
                         grf_config,     
                         waveguide_config):
    dataset_gamma_b_full_input = []
    dataset_gamma_a_snn_target = []
    dataset_gamma_a_true_full_output = [] 
    
    print(f"Generating {num_samples} total samples. PDE: {pde_type}...")

    if pde_type in ["poisson", "step_index_fiber", "grin_fiber", "heat_equation"]:
        grf_generator = GaussianRF(dim=2, size=N_grid_sim_input, 
                                   alpha=grf_config['alpha'], tau=grf_config['tau'], 
                                   device=torch.device("cpu"))

    for i in range(num_samples):
        if (i+1) % (num_samples // 10 or 1) == 0:
            print(f"  Generating sample {i+1}/{num_samples}")

        gamma_b_full_input_spec = None
        gamma_a_true_full_spec = None 
        
        if pde_type == "poisson":
            f_sample_grf = grf_generator.sample(1).cpu().numpy().squeeze()
            f_real_with_offset = add_hierarchical_offset_2d(f_sample_grf, sigma=grf_config.get('offset_sigma', 0.5))
            initial_real_space_state_f = f_real_with_offset - np.mean(f_real_with_offset) 
            gamma_b_full_input_spec = get_full_centered_spectrum(initial_real_space_state_f)
            f_hat_r = np.fft.rfft2(initial_real_space_state_f)
            u_hat_r = solve_poisson_2d_rfft(f_hat_r, enforce_zero_mean=True)
            true_output_real_space_state_u = np.fft.irfft2(u_hat_r, s=initial_real_space_state_f.shape) 
            gamma_a_true_full_spec = get_full_centered_spectrum(true_output_real_space_state_u)
        
        elif pde_type == "step_index_fiber" or pde_type == "grin_fiber":
            initial_real_space_state_unnormalized = grf_generator.sample(1).cpu().numpy().squeeze()
            norm_initial = np.linalg.norm(initial_real_space_state_unnormalized)
            if norm_initial > 1e-14:
                initial_real_space_state_psi0 = initial_real_space_state_unnormalized / norm_initial
            else:
                initial_real_space_state_psi0 = np.zeros_like(initial_real_space_state_unnormalized)
                if N_grid_sim_input > 0: 
                    initial_real_space_state_psi0.flat[0] = 1.0 
                    initial_real_space_state_psi0 = initial_real_space_state_psi0 / np.linalg.norm(initial_real_space_state_psi0)

            gamma_b_full_spec_unnorm = get_full_centered_spectrum(initial_real_space_state_psi0) 
            gamma_b_full_input_spec = normalize_spectrum(gamma_b_full_spec_unnorm) 

            potential_V = None
            if pde_type == "step_index_fiber":
                potential_V = get_step_index_fiber_potential(N_grid_sim_input, 
                                                             waveguide_config['L_domain'], 
                                                             waveguide_config['core_radius_factor'], 
                                                             waveguide_config['potential_depth'])
            elif pde_type == "grin_fiber":
                potential_V = get_grin_fiber_potential(N_grid_sim_input,
                                                       waveguide_config['L_domain'],
                                                       waveguide_config['grin_strength'])

            psi_T_real = solver_main(potential_V, initial_real_space_state_psi0, 
                                 N_grid=N_grid_sim_input, 
                                 L_domain=waveguide_config['L_domain'], 
                                 T_evolution=waveguide_config['evolution_time_T'], 
                                 num_solver_steps=waveguide_config['solver_num_steps'],
                                 hbar=waveguide_config['hbar_val'],
                                 m=waveguide_config['mass_val']) 
            
            gamma_a_true_full_spec_unnorm = get_full_centered_spectrum(psi_T_real) 
            gamma_a_true_full_spec = normalize_spectrum(gamma_a_true_full_spec_unnorm) 
        
        elif pde_type == "heat_equation":
            u0 = grf_generator.sample(1).cpu().numpy().squeeze()
            gamma_b_full_input_spec = get_full_centered_spectrum(u0)
            uT = solve_heat_equation_2d(u0, N=N_grid_sim_input, L=waveguide_config['L_domain'], 
                                        T=waveguide_config['evolution_time_T'], nu=waveguide_config['viscosity_nu'])
            gamma_a_true_full_spec = get_full_centered_spectrum(uT)
        else:
            raise ValueError(f"Unknown pde_type: {pde_type}")

        dataset_gamma_b_full_input.append(gamma_b_full_input_spec)
        dataset_gamma_a_true_full_output.append(gamma_a_true_full_spec)
        
        gamma_a_snn_target_spec = extract_center_block(gamma_a_true_full_spec, K_trunc_snn_output) 
        dataset_gamma_a_snn_target.append(gamma_a_snn_target_spec)

    gamma_b_full_all = np.array(dataset_gamma_b_full_input, dtype=np.complex64)
    gamma_a_snn_target_all = np.array(dataset_gamma_a_snn_target, dtype=np.complex64)
    gamma_a_true_full_all = np.array(dataset_gamma_a_true_full_output, dtype=np.complex64)
    
    return gamma_b_full_all, gamma_a_snn_target_all, gamma_a_true_full_all

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate dataset for SNN training and calibration.")
    
    # --- Core Parameters ---
    parser.add_argument('--pde_type', type=str, default="step_index_fiber", 
                        choices=["poisson", "step_index_fiber", "grin_fiber", "heat_equation"],
                        help="Type of data generation process.")
    parser.add_argument('--num_train', type=int, default=100, help="Number of samples for the training set.") 
    parser.add_argument('--num_calib', type=int, default=100, help="Number of samples for the calibration/test set.")
    parser.add_argument('--n_grid_sim_input', type=int, default=64,
                        help='Grid size for full input & full true output spectra (Nin).')
    parser.add_argument('--k_trunc_snn_output', type=int, default=32,
                        help='Truncation for SNN target output spectra (Nout).')
    parser.add_argument('--output_dir', type=str, default="datasets")
    parser.add_argument('--k_psi0_limit', type=int, default=32,
                        help="Max k for GRF base initial state (used if pde_type is step_index_fiber or poisson with GRF).")

    # --- GRF parameters ---
    parser.add_argument('--grf_alpha', type=float, default=4.0) 
    parser.add_argument('--grf_tau', type=float, default=1.0)   
    parser.add_argument('--grf_offset_sigma', type=float, default=0.5, 
                        help="Sigma for hierarchical offset in Poisson source (f term).")

    # --- Fiber & Evolution Parameters ---
    parser.add_argument('--L_domain', type=float, default=2*np.pi, 
                        help="Physical domain size (e.g., 2pi for periodicity).")
    parser.add_argument('--fiber_core_radius_factor', type=float, default=0.2, 
                        help="Core radius as fraction of L_domain/2.")
    parser.add_argument('--fiber_potential_depth', type=float, default=0.5, 
                        help="Depth V0 of the fiber potential well.")
    parser.add_argument('--grin_strength', type=float, default=0.01,
                        help="Strength C of the GRIN fiber potential V(r) = C*r^2.")
    parser.add_argument('--viscosity_nu', type=float, default=0.01,
                        help="Viscosity/diffusivity nu for the Heat Equation.")
    parser.add_argument('--evolution_time_T', type=float, default=0.1) 
    parser.add_argument('--solver_num_steps', type=int, default=50) 
    parser.add_argument('--hbar_val', type=float, default=HBAR_CONST) 
    parser.add_argument('--mass_val', type=float, default=MASS_CONST) 

    args = parser.parse_args()

    if args.k_trunc_snn_output > args.n_grid_sim_input:
        raise ValueError("K_TRUNC_SNN_OUTPUT cannot be larger than N_GRID_SIM_INPUT.")

    filename_suffix = ""
    if args.pde_type == "poisson":
        filename_suffix = f"poisson_grfA{args.grf_alpha:.1f}T{args.grf_tau:.1f}OffS{args.grf_offset_sigma:.1f}"
    elif args.pde_type == "step_index_fiber":
        filename_suffix = (f"fiber_GRFinA{args.grf_alpha:.1f}T{args.grf_tau:.1f}_"
                           f"coreR{args.fiber_core_radius_factor:.1f}_V{args.fiber_potential_depth:.1f}_"
                           f"evoT{args.evolution_time_T:.1e}_steps{args.solver_num_steps}")
    elif args.pde_type == "grin_fiber":
        filename_suffix = (f"grinfiber_GRFinA{args.grf_alpha:.1f}T{args.grf_tau:.1f}_"
                           f"strength{args.grin_strength:.2e}_"
                           f"evoT{args.evolution_time_T:.1e}_steps{args.solver_num_steps}")
    elif args.pde_type == "heat_equation":
        filename_suffix = (f"heat_GRFinA{args.grf_alpha:.1f}T{args.grf_tau:.1f}_"
                           f"nu{args.viscosity_nu:.2e}_evoT{args.evolution_time_T:.1e}")
    
    print(f"--- Generating dataset for PDE Type '{args.pde_type}' ---")
    print(f"  Training samples: {args.num_train}, Calibration samples: {args.num_calib}")
    print(f"  Filename suffix: {filename_suffix}")

    total_samples = args.num_train + args.num_calib

    grf_config_for_input = { 
        'alpha': args.grf_alpha, 'tau': args.grf_tau, 
        'offset_sigma': args.grf_offset_sigma 
    }
    waveguide_config_params = { 
        'L_domain': args.L_domain, 
        'evolution_time_T': args.evolution_time_T, 
        'solver_num_steps': args.solver_num_steps, 
        'hbar_val': args.hbar_val, 
        'mass_val': args.mass_val,
        'core_radius_factor': args.fiber_core_radius_factor, 
        'potential_depth': args.fiber_potential_depth,
        'grin_strength': args.grin_strength,
        'viscosity_nu': args.viscosity_nu
    }
    
    gamma_b_all, gamma_a_target_all, gamma_a_true_all = generate_snn_dataset(
        total_samples, args.n_grid_sim_input, args.k_psi0_limit,
        args.k_trunc_snn_output, 
        args.pde_type,
        grf_config_for_input, 
        waveguide_config_params
    )
    
    # Split the generated data into train and calib sets
    gamma_b_train = gamma_b_all[:args.num_train]
    gamma_b_calib = gamma_b_all[args.num_train:]
    
    gamma_a_snn_target_train = gamma_a_target_all[:args.num_train]
    gamma_a_snn_target_calib = gamma_a_target_all[args.num_train:]
    
    gamma_a_true_full_train = gamma_a_true_all[:args.num_train]
    gamma_a_true_full_calib = gamma_a_true_all[args.num_train:]

    # Save to a single npz file with multiple fields
    filename_template = os.path.join(args.output_dir, "dataset_{pde_type}_Nin{Nin}_Nout{Nout}_{suffix}.npz")
    save_path = filename_template.format(pde_type=args.pde_type, Nin=args.n_grid_sim_input, Nout=args.k_trunc_snn_output, suffix=filename_suffix)
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    
    np.savez_compressed(save_path, 
                        gamma_b_train=gamma_b_train, 
                        gamma_b_calib=gamma_b_calib,
                        gamma_a_snn_target_train=gamma_a_snn_target_train,
                        gamma_a_snn_target_calib=gamma_a_snn_target_calib,
                        gamma_a_true_full_train=gamma_a_true_full_train,
                        gamma_a_true_full_calib=gamma_a_true_full_calib)
    
    print(f"\nDataset saved to {save_path}")
    print(f"  Train shapes: gamma_b: {gamma_b_train.shape}, gamma_a_target: {gamma_a_snn_target_train.shape}, gamma_a_true: {gamma_a_true_full_train.shape}")
    print(f"  Calib shapes: gamma_b: {gamma_b_calib.shape}, gamma_a_target: {gamma_a_snn_target_calib.shape}, gamma_a_true: {gamma_a_true_full_calib.shape}")


    if total_samples > 0:
        sample_idx = 0 # Visualize first sample of the training set
        gb_full_sample_spec = gamma_b_train[sample_idx]
        ga_snn_target_sample_spec = gamma_a_snn_target_train[sample_idx]
        ga_true_full_sample_spec = gamma_a_true_full_train[sample_idx]

        # --- Visualization (identical to before, just using the train split data) ---
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
        vis_dir = "results_dataset_gen_pde_v3" 
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
        plot_func_input = lambda x: x.real if args.pde_type in ["poisson", "heat_equation"] else np.abs(x)
        plot_func_output = lambda x: x.real if args.pde_type in ["poisson", "heat_equation"] else np.abs(x)
        
        im0_spatial = axes_spatial[0].imshow(plot_func_input(psi_b_real_vis))
        axes_spatial[0].set_title(f"Input Spatial Field ($N_{{in}}={args.n_grid_sim_input}$)")
        plt.colorbar(im0_spatial, ax=axes_spatial[0])

        im1_spatial = axes_spatial[1].imshow(plot_func_output(psi_a_snn_target_spatial_vis))
        axes_spatial[1].set_title(f"SNN Target Spatial (from $N_{{out}}$)")
        plt.colorbar(im1_spatial, ax=axes_spatial[1])

        im2_spatial = axes_spatial[2].imshow(plot_func_output(psi_a_true_full_spatial_vis))
        axes_spatial[2].set_title(f"True Full Output Spatial ($N_{{in}}={args.n_grid_sim_input}$)")
        plt.colorbar(im2_spatial, ax=axes_spatial[2])

        fig_spatial.suptitle(f"Dataset Sample Spatial (PDE: {args.pde_type.upper()}, Params: {filename_suffix})", fontsize=14)
        plt.tight_layout(rect=[0,0,1,0.93])
        save_fig_path_spatial = os.path.join(vis_dir, f"sample_snn_spatial_v3_{args.pde_type}_{filename_suffix}.png")
        plt.savefig(save_fig_path_spatial)
        print(f"Sample SNN spatial visualization saved to {save_fig_path_spatial}")
        plt.close(fig_spatial)
    else:
        print("No samples generated or data is empty, skipping visualization.")

    print("\nDataset generation script finished.")
