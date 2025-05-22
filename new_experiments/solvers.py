import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.fft import ifftn, fft2, fftshift, ifftshift # Ensure all fft functions are imported
import matplotlib.pyplot as plt
import os # Added for creating directory
import argparse # Added for command-line arguments

from constants import GLOBAL_HBAR, GLOBAL_M

# --- User-provided code: Solvers (Split-Step Only) ---
def split_step_solver_2d(V_grid, psi0, N_grid, dx, T, num_steps, hbar=GLOBAL_HBAR, m=GLOBAL_M): 
    dt = T/num_steps
    psi = psi0.astype(np.complex128).copy() 
    V_half_exp = np.exp(-0.5j * V_grid * dt / hbar)
    kx_vec = 2.0 * np.pi * np.fft.fftfreq(N_grid, d=dx)
    ky_vec = 2.0 * np.pi * np.fft.fftfreq(N_grid, d=dx)
    Kx_grid, Ky_grid = np.meshgrid(kx_vec, ky_vec, indexing='ij')
    K_squared = Kx_grid**2 + Ky_grid**2 
    K_full_step_exp = np.exp(-1.0j * hbar * K_squared * dt / (2.0*m))
    for _ in range(num_steps):
        psi *= V_half_exp
        psi_k = np.fft.fft2(psi)
        psi_k *= K_full_step_exp 
        psi = np.fft.ifft2(psi_k)
        psi *= V_half_exp
    return psi

def split_step_solver_2d_time_varying(V_fn, psi0, N_grid, dx, T, num_steps, hbar=GLOBAL_HBAR, m=GLOBAL_M): 
    dt = T / num_steps
    psi = psi0.astype(np.complex128).copy()
    kx_vec = 2.0 * np.pi * np.fft.fftfreq(N_grid, d=dx)
    ky_vec = 2.0 * np.pi * np.fft.fftfreq(N_grid, d=dx)
    Kx_grid, Ky_grid = np.meshgrid(kx_vec, ky_vec, indexing='ij')
    K_squared = Kx_grid**2 + Ky_grid**2
    K_full_step_exp = np.exp(-1.0j * hbar * K_squared * dt / (2.0*m))
    current_time = 0.0
    for n_step in range(num_steps):
        V_at_tn = V_fn(current_time)
        V_half_exp_tn = np.exp(-0.5j * V_at_tn * dt / hbar)
        psi *= V_half_exp_tn
        psi_k = np.fft.fft2(psi)
        psi_k *= K_full_step_exp
        psi = np.fft.ifft2(psi_k)
        current_time_for_Vnp1 = (n_step + 1) * dt 
        V_at_tnp1 = V_fn(current_time_for_Vnp1) 
        V_half_exp_tnp1 = np.exp(-0.5j * V_at_tnp1 * dt / hbar)
        psi *= V_half_exp_tnp1
        current_time = current_time_for_Vnp1         
    return psi

def solver(V_input, psi0, N_grid, dx, T, num_steps, hbar=GLOBAL_HBAR, m=GLOBAL_M):
    """
    Selects the appropriate split-step solver based on whether the potential
    V_input is time-independent (numpy array) or time-dependent (callable).
    """
    if isinstance(V_input, np.ndarray): 
        # Time-independent potential
        return split_step_solver_2d(V_input, psi0, N_grid, dx, T, num_steps, hbar, m)
    elif callable(V_input): 
        # Time-dependent potential (V_input is a function V_fn(t))
        return split_step_solver_2d_time_varying(V_input, psi0, N_grid, dx, T, num_steps, hbar, m)
    else:
        raise TypeError("Potential V_input must be a numpy array (for time-independent) or a callable function (for time-dependent).")