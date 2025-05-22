import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.fft import ifftn, fft2, fftshift, ifftshift # Ensure all fft functions are imported
import matplotlib.pyplot as plt
import os # Added for creating directory
import argparse # Added for command-line arguments

# --- User-provided code: Wavefunction/State Generation ---
class DictionaryComplexDataset(Dataset): # Original, for reference, not used directly by spectral operator
    """
    A PyTorch Dataset that yields (phi_2chan, psi_2chan) from dictionary data,
    where each is shape (2, N, N):
      channel 0 => real part
      channel 1 => imaginary part
    """
    def __init__(self, train_samples):
        self.samples = []
        for (psi0, psiT) in train_samples:
            psi0_2ch = np.stack([psi0.real, psi0.imag], axis=0)
            psiT_2ch = np.stack([psiT.real, psiT.imag], axis=0)
            self.samples.append((psi0_2ch, psiT_2ch))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        psi0_2ch, psiT_2ch = self.samples[idx]
        psi0 = torch.from_numpy(psi0_2ch).float()
        psiT = torch.from_numpy(psiT_2ch).float()
        return psi0, psiT

def construct_dataset(samples, batch_size): # Original, for reference
    dataset = DictionaryComplexDataset(samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def random_low_order_state(N, K=16):
    """
    Construct a random wavefunction whose fft2 is nonzero only in 
    the band -K..K for both axes. Return it in real space.
    """
    freq_array = np.zeros((N,N), dtype=np.complex128)
    
    def wrap_index(k_val): 
        return k_val % N
    
    for kx_eff in range(-K, K+1):
        for ky_eff in range(-K, K+1):
            amp_real = np.random.randn()
            amp_imag = np.random.randn()
            c = amp_real + 1j*amp_imag
            kx = wrap_index(kx_eff)
            ky = wrap_index(ky_eff)
            freq_array[kx, ky] = c
            
    psi0 = np.fft.ifft2(freq_array) 
    
    norm_psi = np.linalg.norm(psi0)
    if norm_psi > 1e-14:
        psi0 /= norm_psi
    return psi0

def GRF(alpha, beta, gamma, N_grid): 
    """
    Generates a Gaussian Random Field on a 2D grid.
    """
    xi = np.random.randn(N_grid, N_grid) 
    K1_idx, K2_idx = np.meshgrid(np.arange(N_grid), np.arange(N_grid), indexing='ij')
    k_squared_term = 4 * np.pi**2 * (K1_idx**2 + K2_idx**2)
    
    # Handle potential division by zero if beta is zero and k_squared_term is zero
    # This is for the (0,0) mode.
    if beta == 0 and k_squared_term[0,0] == 0:
        # if gamma > 0, (0)^(-gamma/2) is inf.
        # To avoid issues, if beta is 0, we can ensure the (0,0) component of k_squared_term is not 0
        # or handle the coefficient for (0,0) mode separately.
        # However, L_fourier_coeffs[0,0] is set to 0 later, which handles mean-zero field.
        # For coef calculation, if beta is 0, (0 + 0)^(-gamma/2) is problematic.
        # A common fix is to ensure beta is small positive, or modify k_squared_term for (0,0) if beta is 0.
        # Let's add a small epsilon if beta is zero and k_squared_term is zero to avoid NaN/Inf in coef.
        if k_squared_term[0,0] == 0: # This check is redundant if K1_idx, K2_idx start from 0
             k_squared_term_safe = k_squared_term.copy()
             if beta == 0: # Ensure beta is actually zero before modifying
                 k_squared_term_safe[0,0] = 1e-9 # Avoid division by zero for the DC component if beta is 0
             coef = alpha**(1/2) * (k_squared_term_safe + beta)**(-gamma / 2)
        else: # This case should not be reached if K1_idx[0,0]==0 and K2_idx[0,0]==0
             coef = alpha**(1/2) * (k_squared_term + beta)**(-gamma / 2)
    else:
        coef = alpha**(1/2) * (k_squared_term + beta)**(-gamma / 2)

    L_fourier_coeffs = N_grid * coef * xi
    L_fourier_coeffs[0, 0] = 0.0 + 0.0j 
    field = np.fft.ifftn(L_fourier_coeffs, norm='forward')
    return field.real

def get_mesh(N_grid, L_domain, d=2): 
    dx = L_domain/N_grid 
    vals = [np.arange(N_grid)*dx for _ in range(d)]
    return np.meshgrid(*vals, indexing='ij')

def free_particle_potential(N_grid): 
    return np.zeros((N_grid, N_grid))

def barrier_potential(N_grid, L_domain, barrier_height=20.0, slit_width_ratio=0.4): 
    Vgrid = np.zeros((N_grid,N_grid))
    i0 = N_grid//2 
    num_slit_pixels = int(N_grid * slit_width_ratio)
    j_center = N_grid // 2
    j_low = j_center - num_slit_pixels // 2
    j_high = j_low + num_slit_pixels
    Vgrid[i0, :] = barrier_height 
    Vgrid[i0, j_low:j_high] = 0.0  
    return Vgrid

def harmonic_oscillator_potential(N_grid, L_domain, omega=1.0, m_potential=1.0): 
    X, Y = get_mesh(N_grid, L_domain)
    Xc = X - L_domain/2 
    Yc = Y - L_domain/2
    return 0.5*m_potential*(omega**2)*(Xc**2 + Yc**2)

def random_potential(N_grid, alpha, beta, gamma): 
    return GRF(alpha, beta, gamma, N_grid)

def paul_trap(N_grid, L_domain, t, U0=1.0, V0=1.0, omega_trap=1.0, r0_sq_factor=0.1): 
    X, Y = get_mesh(N_grid, L_domain) 
    Xc = X - L_domain/2 
    Yc = Y - L_domain/2
    r0_sq = (L_domain * r0_sq_factor)**2 
    factor = (U0 + V0 * np.cos(omega_trap*t)) / r0_sq
    return factor * (Xc**2 + Yc**2) 