import argparse
import copy
import einops
import math
import logging
import os
import pickle

import scipy.stats.qmc as qmc
import scipy.spatial.distance as distance
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3

import torch
import math
import scipy.io
from timeit import default_timer
from tqdm.notebook import tqdm

import utils

logger = logging.getLogger(__name__)
device = "cuda:0"

##############################################################################
# 1) The Gaussian Random Field class (as given)
##############################################################################

class GaussianRF(object):
    """
    Represents a Gaussian Random Field generator.
    (Provided in the question; repeated here for completeness.)
    """
    def __init__(self, dim, size, alpha=2, tau=3, sigma=None, boundary="periodic", device=None):
        self.dim = dim
        self.device = device

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))

        k_max = size // 2

        if dim == 1:
            import math
            k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                           torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

            self.sqrt_eig = (size * math.sqrt(2.0) * sigma *
                             ((4 * (math.pi**2) * (k**2) + tau**2) ** (-alpha/2.0)))
            self.sqrt_eig[0] = 0.0

        elif dim == 2:
            import math
            wavenumbers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                                      torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size, 1)

            k_x = wavenumbers.transpose(0, 1)
            k_y = wavenumbers

            self.sqrt_eig = ((size**2) * math.sqrt(2.0) * sigma *
                             ((4*(math.pi**2)*(k_x**2 + k_y**2) + tau**2) ** (-alpha/2.0)))
            self.sqrt_eig[0,0] = 0.0

        elif dim == 3:
            import math
            wavenumbers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                                      torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size, size, 1)

            k_x = wavenumbers.transpose(1, 2)
            k_y = wavenumbers
            k_z = wavenumbers.transpose(0, 2)

            self.sqrt_eig = ((size**3) * math.sqrt(2.0) * sigma *
                             ((4*(math.pi**2)*(k_x**2 + k_y**2 + k_z**2) + tau**2) ** (-alpha/2.0)))
            self.sqrt_eig[0,0,0] = 0.0

        self.size = [size]*dim
        self.size = tuple(self.size)

    def sample(self, N):
        """
        Generates N samples of the random field.

        Returns:
            torch.Tensor of shape (N, *self.size) with real values.
        """
        coeff = torch.randn(N, *self.size, dtype=torch.cfloat, device=self.device)
        coeff = self.sqrt_eig * coeff

        # Inverse FFT along all spatial dimensions
        return torch.fft.ifftn(coeff, dim=list(range(-1, -self.dim - 1, -1))).real


##############################################################################
# 2) The Poisson solver in Fourier space (using rfft2/irfft2)
##############################################################################

def solve_poisson_2d_fourier(f_hat: np.ndarray, enforce_zero_mean: bool = True) -> np.ndarray:
    """
    Solve the Poisson equation Δu = f on the 2D torus using the Fourier method.

    Parameters
    ----------
    f_hat : np.ndarray, shape (N1, N2)
        2D array of *raw* Fourier coefficients of f in unshifted (NumPy) order.
    enforce_zero_mean : bool
        If True, enforce that f_hat(0,0)=0 and also set u_hat(0,0)=0.

    Returns
    -------
    u_hat : np.ndarray, shape (N1, N2)
        2D array of Fourier coefficients of the solution u.
    """
    N1, N2 = f_hat.shape
    u_hat = np.zeros_like(f_hat, dtype=np.complex128)

    if enforce_zero_mean:
        # Force zero frequency of f to be zero if it's not:
        if abs(f_hat[0, 0]) > 1e-14:
            # You might want to log or warn here
            f_hat[0, 0] = 0.0

    for k1 in range(N1):
        # map to integer wave number k1_shifted
        if k1 <= N1 // 2:
            k1_shifted = k1
        else:
            k1_shifted = k1 - N1

        for k2 in range(N2):
            if k2 <= N2 // 2:
                k2_shifted = k2
            else:
                k2_shifted = k2 - N2

            if k1_shifted == 0 and k2_shifted == 0:
                # zero mode
                u_hat[k1, k2] = 0.0
            else:
                k_sq = k1_shifted**2 + k2_shifted**2
                # -|k|^2 * u_hat(k) = f_hat(k)  =>  u_hat(k) = -f_hat(k)/|k|^2
                u_hat[k1, k2] = - f_hat[k1, k2] / k_sq

    return u_hat

def solve_poisson_2d_rfft(f_hat_r: np.ndarray, enforce_zero_mean: bool = True) -> np.ndarray:
    """
    Solve the Poisson equation Δu = f on the 2D torus using real FFTs.

    Parameters
    ----------
    f_hat_r : np.ndarray, shape (N1, N2//2+1)
        2D array of half-complex Fourier coefficients of f, as produced by np.fft.rfft2.
    enforce_zero_mean : bool
        If True, enforce that the zero mode f_hat_r[0,0]=0.

    Returns
    -------
    u_hat_r : np.ndarray, shape (N1, N2//2+1)
        2D array of half-complex Fourier coefficients of the solution u.
    """
    N1, M = f_hat_r.shape  # here, M = N2//2 + 1
    u_hat_r = np.zeros_like(f_hat_r, dtype=np.complex128)

    if enforce_zero_mean:
        if abs(f_hat_r[0, 0]) > 1e-14:
            f_hat_r[0, 0] = 0.0

    for k1 in range(N1):
        # Map k1 to its shifted frequency
        if k1 <= N1 // 2:
            k1_shifted = k1
        else:
            k1_shifted = k1 - N1

        for k2 in range(M):
            # For rfft2, k2 runs from 0 to N2//2 (nonnegative only)
            k2_shifted = k2  # already nonnegative

            if k1_shifted == 0 and k2_shifted == 0:
                u_hat_r[k1, k2] = 0.0
            else:
                k_sq = k1_shifted**2 + k2_shifted**2
                u_hat_r[k1, k2] = - f_hat_r[k1, k2] / k_sq

    return u_hat_r

##############################################################################
# 3) Sobolev norm computation in 2D
##############################################################################

def sobolev_norm_2d(f_hat: np.ndarray, s: float) -> float:
    """
    Compute the H^s(T^2) norm of a function f whose full Fourier coefficients
    are given by f_hat in unshifted NumPy FFT order.

    Note: This function expects f_hat to be a full (N1 x N2) array.
          If you use rfft2, you need to expand the half-complex array to a full
          complex array (or modify this function accordingly).
    """
    N1, N2 = f_hat.shape
    total = 0.0

    for k1 in range(N1):
        if k1 <= N1 // 2:
            k1_shifted = k1
        else:
            k1_shifted = k1 - N1

        for k2 in range(N2):
            if k2 <= N2 // 2:
                k2_shifted = k2
            else:
                k2_shifted = k2 - N2

            k_sq = k1_shifted**2 + k2_shifted**2
            factor = (1.0 + k_sq)**s
            coeff_sq = np.abs(f_hat[k1, k2])**2
            total += factor * coeff_sq

    return np.sqrt(total)

def rfft2_to_fft2(rfft_result, input_shape):
    """
    Converts the result of numpy.fft.rfft2 to the equivalent result of numpy.fft.fft2.

    Args:
        rfft_result (numpy.ndarray): The result of numpy.fft.rfft2.
        input_shape (tuple): The shape of the original input array to rfft2.

    Returns:
        numpy.ndarray: The equivalent result of numpy.fft.fft2.
    """
    rows, cols = input_shape
    full_fft = np.zeros((rows, cols), dtype=np.complex128)
    full_fft[:, : cols // 2 + 1] = rfft_result

    # Reconstruct the negative frequencies
    for i in range(rows):
        for j in range(1, cols // 2 + (0 if cols % 2 == 0 else 1)):
            full_fft[i, cols - j] = np.conjugate(rfft_result[i, j])

    return full_fft

##############################################################################
# 4) Hierarchical offset in real space
##############################################################################

def add_hierarchical_offset_2d(
    f: np.ndarray,
    K_max: int = 6,
    sigma: float = 0.5,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Add a 2D 'Gaussian source offset' mu to an existing 2D function f.
    (See original code for details.)
    """
    if rng is None:
        rng = np.random.default_rng()

    Nx, Ny = f.shape
    x_vals = np.linspace(0, 2*np.pi, Nx, endpoint=False)
    y_vals = np.linspace(0, 2*np.pi, Ny, endpoint=False)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

    # Define possible centers (here using a circular arrangement)
    n_points = 6
    radius   = np.pi / 2
    angles   = [k * 2 * np.pi / n_points for k in range(n_points)]
    c_list   = np.array([np.pi + radius * np.array([np.cos(angle), np.sin(angle)]) for angle in angles])

    # Randomly pick one center (for simplicity)
    center_indices = [np.random.randint(low=0, high=len(c_list))]
    
    mu = np.zeros((Nx, Ny), dtype=float)
    for idx in center_indices:
        c_x, c_y = c_list[idx]
        dist_sq = (X - c_x)**2 + (Y - c_y)**2
        mu += np.exp(- dist_sq / (2*sigma**2))

    f_aug = f + mu
    return f_aug

##############################################################################
# 5) Putting it all together: generate random f, solve for u, check the bound
##############################################################################
import math

def solve_poisson(num_samples):
    # Parameters for the Gaussian random field
    dim = 2
    size = 256
    alpha = 2
    tau = 3
    device = "cpu"

    # Instantiate the GRF object
    grf = GaussianRF(dim=dim, size=size, alpha=alpha, tau=tau, sigma=None,
                     boundary="periodic", device=device)

    # Generate random samples in real space: shape (num_samples, size, size)
    f_samples = grf.sample(num_samples).cpu().numpy()

    f_hats, u_hats = [], []

    # Loop through samples
    for i in range(num_samples):
        f_real = add_hierarchical_offset_2d(f_samples[i])
        # Enforce zero-average in real space
        f_real = f_real - np.mean(f_real)

        # Use rfft2 since f_real is real; this returns an array of shape (size, size//2+1)
        f_hat_r = np.fft.rfft2(f_real)

        # Solve Poisson in the half-complex domain
        u_hat_r = solve_poisson_2d_rfft(f_hat_r, enforce_zero_mean=True)

        # (Optional) To compute norms with your existing sobolev_norm_2d, you would need to
        # expand the half-complex array to a full array. For now, we simply store the half-complex
        # coefficients.
        # One could use: f_hat_full = np.fft.irfft2(f_hat_r)  (but that gives back f_real),
        # so instead you could define a function to "complete" the half-complex array if needed.
        #
        # Here, we simply store the half-complex arrays.
        f_hats.append(f_hat_r)
        u_hats.append(u_hat_r)

        norm_u_H1  = sobolev_norm_2d(rfft2_to_fft2(u_hat_r, (256, 256)), s=1)
        norm_f_Hm1 = sobolev_norm_2d(rfft2_to_fft2(f_hat_r, (256, 256)), s=-1)

        ratio = norm_u_H1 / (norm_f_Hm1 + 1e-14)  # Avoid div by 0 if f=0
        print(f"Sample {i}: ||u||_H1 = {norm_u_H1:.5e}, ||f||_H^-1 = {norm_f_Hm1:.5e}, ratio = {ratio:.3f}")

    return np.array(f_hats), np.array(u_hats)
    

# Function to solve Navier-Stokes equation in 2D
def navier_stokes_2d(w0, f, visc, T, delta_t=1e-4, record_steps=1):
    """
    Solve the 2D Navier-Stokes equations using the Fourier spectral method.

    Parameters:
    - w0 (torch.Tensor): Initial vorticity field.
    - f (torch.Tensor): Forcing field.
    - visc (float): Viscosity coefficient.
    - T (float): Total time.
    - delta_t (float): Time step size (default: 1e-4).
    - record_steps (int): Number of steps between each recorded solution (default: 1).

    Returns:
    - sol (torch.Tensor): Solution tensor containing the vorticity field at each recorded time step.
    - sol_t (torch.Tensor): Time tensor containing the recorded time steps.
    """
    # Grid size - it must be power of 2
    N = w0.size()[-1]

    # Max wavenumber
    k_max = math.floor(N/2.0)

    # Total number of steps
    steps = math.ceil(T/delta_t)

    # Initial vortex field in Fourier space
    w_h = torch.fft.rfft2(w0)

    # Forcing field in Fourier space
    f_h = torch.fft.rfft2(f)

    # If the same forcing for the whole batch
    if len(f_h.size()) < len(w_h.size()):
        f_h = torch.unsqueeze(f_h, 0)

    # Save the solution every certain number of steps
    record_time = math.floor(steps/record_steps)

    # Wave numbers in y-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=w0.device), torch.arange(start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N,1)
    
    # Wave numbers in x-direction
    k_x = k_y.transpose(0,1)

    # Remove redundant modes
    k_x = k_x[..., :k_max + 1]
    k_y = k_y[..., :k_max + 1]

    # Negative of the Laplacian in Fourier space
    lap = 4*(math.pi**2)*(k_x**2 + k_y**2)
    lap[0,0] = 1.0
    
    # Dealiasing mask
    dealias = torch.unsqueeze(torch.logical_and(torch.abs(k_y) <= (2.0/3.0)*k_max, torch.abs(k_x) <= (2.0/3.0)*k_max).float(), 0)

    # Save the solution and time
    sol = torch.zeros(*w0.size(), record_steps, device=w0.device)
    
    #Record counter
    c = 0
    
    #Physical time
    t = 0.0
    for j in range(steps):
        
        #Stream function in Fourier space: solve Poisson equation
        psi_h = w_h / lap

        #Velocity field in x-direction = psi_y
        q = 2. * math.pi * k_y * 1j * psi_h
        q = torch.fft.irfft2(q, s=(N, N))

        #Velocity field in y-direction = -psi_x
        v = -2. * math.pi * k_x * 1j * psi_h
        v = torch.fft.irfft2(v, s=(N, N))

        #Partial x of vorticity
        w_x = 2. * math.pi * k_x * 1j * w_h
        w_x = torch.fft.irfft2(w_x, s=(N, N))

        #Partial y of vorticity
        w_y = 2. * math.pi * k_y * 1j * w_h
        w_y = torch.fft.irfft2(w_y, s=(N, N))

        #Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
        F_h = torch.fft.rfft2(q*w_x + v*w_y)

        #Dealias
        F_h = dealias* F_h

        #Crank-Nicolson update
        w_h = (-delta_t*F_h + delta_t*f_h + (1.0 - 0.5*delta_t*visc*lap)*w_h)/(1.0 + 0.5*delta_t*visc*lap)

        #Update real time (used only for recording)
        t += delta_t

        if (j+1) % record_time == 0:
            #Solution in physical space
            w = torch.fft.irfft2(w_h, s=(N, N))
            sol[...,c] = w
            c += 1

    return sol


def solve_navier_stokes(N):
    #Resolution
    s = 256

    #Set up 2d GRF with covariance parameters
    GRF = GaussianRF(2, s, alpha=2.5, tau=7, device=device)

    # Time grid
    t = torch.linspace(0, 1, s+1, device=device)
    t = t[0:-1]

    GRF = GaussianRF(2, s, alpha=4.5, tau=7, device=device)
    a = GRF.sample(1)[0].detach().cpu().numpy()

    # Grid size
    N = 256

    # Create normalized coordinate arrays [0,1] x [0,1]
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)

    # Create a meshgrid (X, Y) for evaluating the function on the entire 2D domain
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    n_points = 6
    radius   = 0.3
    angles   = [k * 2 * np.pi / n_points for k in range(n_points)]
    centers  = np.array([0.5 + radius * np.array([np.cos(angle), np.sin(angle)]) for angle in angles])

    (cx,cy) = centers[np.random.randint(low=1, high=len(centers))]

    # Standard deviation (spread) for the Gaussian peaks
    sigma = 0.25

    Z = np.exp(-(((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2)))
    # b = Z * .25
    
    f = torch.from_numpy(Z).to(device) * 0.1

    #Number of snapshots from solution
    record_steps = 200

    #Inputs
    a = torch.zeros(N, s, s)
    #Solutions
    u = torch.zeros(N, s, s, record_steps)

    #Solve equations in batches (order of magnitude speed-up)
    bsize = 50

    c = 0
    t0 = default_timer()
    for j in range(N // bsize):
        print(f"Batch: {j}")

        #Sample random feilds
        w0 = GRF.sample(bsize)

        #Solve NS
        sol = navier_stokes_2d(w0, f, 1e-5, 2.5, 1e-4, record_steps)

        a[c:(c+bsize),...] = w0
        u[c:(c+bsize),...] = sol

        c += bsize
        t1 = default_timer()
        print(j, c, t1-t0)
    X, Y = u[...,0], u[...,-1]
    
    # used convert to Dedalus coefficient representation for final dataset
    Lx, Ly = 2 * np.pi, 2 * np.pi
    dtype = np.float64
    coords = d3.CartesianCoordinates("x", "y")
    dist   = d3.Distributor(coords, dtype=dtype)
        
    xbasis = d3.RealFourier(coords["x"], size=s, bounds=(0, Lx))
    ybasis = d3.RealFourier(coords["y"], size=s, bounds=(0, Ly))
    field = dist.Field(name='u', bases=(xbasis, ybasis))

    u_i_cs, u_f_cs = [], []    
    for u_i, u_f in zip(X, Y):
        field["g"] = u_i.detach().cpu().numpy()
        u_i_cs.append(field["c"].copy())

        field["g"] = u_f.detach().cpu().numpy()
        u_f_cs.append(field["c"].copy())
    return np.array(u_i_cs), np.array(u_f_cs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde")
    parser.add_argument("--N", type=int)
    args = parser.parse_args()

    pde_to_func = {
        "poisson":  solve_poisson,
        "navier_stokes": solve_navier_stokes,
    }
    fs, us = pde_to_func[args.pde](args.N)

    os.makedirs(utils.PDE_DIR(args.pde), exist_ok=True)
    with open(utils.DATA_FN(args.pde), "wb") as f:
        pickle.dump((fs, us), f)