import argparse
import einops
import os
import torch

import dedalus.public as d3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy import signal

from spec_op import EncoderDecoderNet
import utils

device = "cuda:0"

import math

from scipy.special import j1

def rfft2_to_fft2(rfft, im_shape):
    fcols = im_shape[-1]
    N, _, fft_cols = rfft.shape

    result = np.zeros((N,) + im_shape, dtype=rfft.dtype)
    result[:, :, :fft_cols] = rfft
    top = rfft[:, 0, 1:]

    if fcols % 2 == 0:
        result[:, 0, fft_cols-1:] = top[:, ::-1].conj()
        mid = rfft[:, 1:, 1:]
        mid = np.concatenate((mid, mid[:, ::-1, ::-1][:, :, 1:].conj()), axis=-1)
    else:
        result[:, 0, fft_cols:] = top[:, ::-1].conj()
        mid = rfft[:, 1:, 1:]
        mid = np.concatenate((mid, mid[:, ::-1, ::-1].conj()), axis=-1)

    result[:, 1:, 1:] = mid
    return result

def compute_I(k1_shifted, k2_shifted, center, radius):
    """
    Compute I(k) = ∫_{B(center; radius)} e^{i k·x} dx
    for a 2D disk of radius 'radius' centered at 'center'=(wx,wy) in [0,2π]^2.

    We assume no boundary clipping, i.e. the entire disk B(center;radius) 
    is used as if in R^2. If you want partial intersection with [0,2π]^2, 
    adapt this integral.

    For the disk around 0, the integral is:
        2π * r * J1(r|k|)/|k|,  or π r^2 if |k|=0.
    We multiply by e^{ i k·center } to shift the center from 0 to 'center'.

    Args:
        k1_shifted, k2_shifted : the integer wave numbers (possibly negative),
                                 e.g. unshifted indexing on [0..255].
        center : (wx, wy) in [0,2π]^2 (real space).
        radius : float, the disk radius in real-space units.
    
    Returns:
        A single complex number for the integral I(k).
    """
    wx, wy = center
    kx = float(k1_shifted)
    ky = float(k2_shifted)
    k_mag = math.sqrt(kx**2 + ky**2)

    if k_mag < 1e-14:
        # k=0 => integral is area of disk => π * r^2
        return -np.pi * (radius**2)
    else:
        # Phase from e^{i k·center}
        phase = np.exp(1j*(kx*wx + ky*wy))
        # radial factor from integral over disk => 2π r J1(r|k|)/|k|
        val = 2.0 * np.pi * radius * j1(radius*k_mag) / k_mag
        return -val * phase

def maximize_u_hat(u_hat, q, s=1.0, center=(0.0, 0.0), radius=1.0):
    """
    Given:
      - u_hat: np.ndarray of shape (256,256), the original Fourier array of u.
      - q: float, the "budget" radius.
      - s: float, the Sobolev exponent in (1 + |k|^2)^s.
      - center, radius: define the real-space disk B(center, radius) in [0,2π]^2 
        used for computing I(k).

    Build the maximizer:
      u_hat^*(k) = u_hat(k) + (q / ||phi||_{H^s}) * [ I(k) / (1+|k|^2)^s ],
    where
      ||phi||_{H^s}^2 = ∑_k |I(k)|^2 / (1+|k|^2)^{s}.

    Returns:
      u_hat_star: np.ndarray of shape (256,256), the updated Fourier array.
    """
    rows, cols = u_hat.shape
    assert rows==256 and cols==256, "Expected a 256x256 Fourier array."

    # 1) Compute phi_norm^2
    phi_norm_sq = 0.0
    for k1 in range(rows):
        if k1 <= rows//2:
            k1_shifted = k1
        else:
            k1_shifted = k1 - rows

        for k2 in range(cols):
            if k2 <= cols//2:
                k2_shifted = k2
            else:
                k2_shifted = k2 - cols

            Ik = compute_I(k1_shifted, k2_shifted, center, radius)
            k_sq = k1_shifted**2 + k2_shifted**2
            denom = (1.0 + k_sq)**s
            phi_norm_sq += (abs(Ik)**2) / (denom**2)

    phi_norm = math.sqrt(phi_norm_sq)

    # 2) Construct u_hat^*
    u_hat_star = np.copy(u_hat)
    for k1 in range(rows):
        if k1 <= rows//2:
            k1_shifted = k1
        else:
            k1_shifted = k1 - rows

        for k2 in range(cols):
            if k2 <= cols//2:
                k2_shifted = k2
            else:
                k2_shifted = k2 - cols

            Ik = compute_I(k1_shifted, k2_shifted, center, radius)
            k_sq = k1_shifted**2 + k2_shifted**2
            denom = (1.0 + k_sq)**s

            # The increment
            incr = (q / phi_norm) * (Ik / denom)
            u_hat_star[k1, k2] += incr

    return u_hat_star


def maximize_u_hat_vectorized(u_hat, q, s=1.0, center=(0.0, 0.0), radius=1.0):
    """
    Vectorized version of maximize_u_hat, removing the explicit Python loops
    over k1 and k2.
    """
    rows, cols = u_hat.shape
    assert (rows, cols) == (256, 256), "Expected a 256x256 Fourier array."

    # Make a 2D grid for k1, k2 in the 'shifted' sense:
    k1_vals = np.fft.ifftshift(np.arange(-rows//2, rows//2))  # range: -128..127
    k2_vals = np.fft.ifftshift(np.arange(-cols//2, cols//2))  # range: -128..127
    K1, K2 = np.meshgrid(k1_vals, k2_vals, indexing='ij')  # each is (256,256)

    # Magnitude |k|
    k_sq = K1**2 + K2**2
    k_mag = np.sqrt(k_sq, dtype=np.float64)  # shape (256,256)
    
    # Phase factor e^{i k·center}
    # center = (wx, wy).  Dot product k·center = K1*wx + K2*wy
    wx, wy = center
    phase = np.exp(1j * (K1*wx + K2*wy))  # shape (256,256)
    
    # Radial factor for disk integral:
    #   I(k) = - [ 2π r J1(r|k|)/|k| ] * e^{i k·center},   except k=0 => -π r^2
    radial_factor = np.zeros_like(k_mag, dtype=np.complex128)
    # Where k != 0
    nonzero_mask = (k_mag > 1e-14)
    radial_factor[nonzero_mask] = -2.0 * np.pi * radius * j1(radius * k_mag[nonzero_mask]) / k_mag[nonzero_mask]
    # Where k = 0
    radial_factor[~nonzero_mask] = -np.pi * (radius**2)
    
    # Full I(k)
    I_k = radial_factor * phase  # shape (256,256)

    # Denominator (1 + |k|^2)^s
    denom = (1.0 + k_sq)**s  # shape (256,256)

    # 1) Compute ||phi||_{H^s}^2 = sum over k of (|I(k)|^2 / denom^2)
    abs_I_sq = np.abs(I_k)**2
    phi_norm_sq = np.sum(abs_I_sq / (denom**2))
    phi_norm = math.sqrt(phi_norm_sq)

    # 2) Construct u_hat^* = u_hat + (q / phi_norm) * [ I(k)/denom ]
    incr = (q / phi_norm) * (I_k / denom)  # shape (256,256)
    u_hat_star = u_hat + incr

    return u_hat_star


def compute_window_sum(u_hat_full, w, r, window_size, q_hat):
    """
    Given an integer pixel shift `w` in the 2D array,
    1) call maximize_u_hat_vectorized(...),
    2) ifft to real space,
    3) sum the sub-window around `w`.

    Returns: float
    """
    # Convert w (in pixel indices) to real-space center (wx, wy) in [0, 2π]^2
    to_real_space = lambda w : -2 * np.pi * (w / 256)
    c_real = to_real_space(w)
    
    # Build the new Fourier array
    u_star_hat = maximize_u_hat_vectorized(u_hat_full, q=q_hat, s=1.0, center=c_real, radius=r)
    u_star_real = np.fft.ifft2(u_star_hat).real
    
    # Sub-window around w
    x0, x1 = w[0] - window_size[0], w[0] + window_size[0] + 1
    y0, y1 = w[1] - window_size[1], w[1] + window_size[1] + 1
    well_window = u_star_real[x0:x1, y0:y1]
    return well_window.sum()


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


def get_collection_window(radius):
    # NOTE: radius is specified in *index* space, i.e. not in the actual discretized space units
    diam = 2 * radius + 1
    coords = cartesian_product(np.array(range(diam)), np.array(range(diam))).reshape((diam, diam, 2))
    offsets = coords - np.array([radius, radius])
    return offsets


def get_collection_border_normals(radius):
    offsets = get_collection_window(radius)
    lens = np.linalg.norm(offsets, axis=-1)
    
    well_border = (np.abs(lens - radius) < .5).astype(float) # HACK: .5 seems to work well in practice; no real reason (that I can see?) for it
    normals = offsets / np.expand_dims(lens, axis=-1)
    well_normals = np.expand_dims(well_border, axis=-1) * normals
    well_normals = np.nan_to_num(well_normals)
    return well_border, well_normals


def get_collection_masks(radius):
    offsets = get_collection_window(radius)
    lens = np.linalg.norm(offsets, axis=-1)

    collection_well = (lens <= radius).astype(float)
    return collection_well


def solve_nominal(field_g, radius):
    collection_well = get_collection_masks(radius)
    conv_offset = np.array(collection_well.shape) // 2
    collection = signal.convolve2d(field_g, collection_well, mode='valid')
    unadj_w_star = np.unravel_index(np.argmax(collection, axis=None), collection.shape)
    w_star = unadj_w_star + conv_offset
    return w_star


def solve_robust_grad(w_nom_star, r_pix, uhat_spectrum, conformal_radius):
    to_real_space = lambda w : -2 * np.pi * (w / 256)
    w = w_nom_star

    eta = 1.0
    T = 100
    ws = [w.copy()]
    w_prev = w.copy()

    for _ in range(T):
        u_star_hat = maximize_u_hat(uhat_spectrum, q=conformal_radius, s=1.0, center=to_real_space(w), radius=r)
        u_star_real = np.fft.ifft2(u_star_hat).real

        well_border, well_normals = get_collection_border_normals(r_pix)
        window_size = np.array(well_border.shape) // 2
        well_window = u_star_real[w[0] - window_size[0]:w[0] + window_size[0] + 1,w[1] - window_size[1]:w[1] + window_size[1] + 1]
        grad = np.array([
            np.sum(well_window * well_normals[...,0] * well_border),
            np.sum(well_window * well_normals[...,1] * well_border),
        ])
        
        w_prev = w
        w = np.round(w + eta * grad).astype(int)
        ws.append(w.copy())

        if np.allclose(w, w_prev): # form of early stopping
            break
    return w.copy()


def solve_robust(w_nom_star, r_pix, uhat_spectrum, conformal_radius):
    # Vectorized approach
    r_optimization = 10  # how far to search over
    delta_vals = np.arange(-r_optimization, r_optimization + 1)      # -5..5
    dwx, dwy = np.meshgrid(delta_vals, delta_vals, indexing='ij')  # each (11,11)
    # Combine into shape (11,11,2)
    shifts = np.stack([dwx, dwy], axis=-1)   # shape (11,11,2)
    window_size = np.array([r_pix, r_pix])

    # Next, add w_nom_star so that each entry is the actual integer shift in the image
    W = w_nom_star + shifts  # shape (11,11,2)

    # We'll define a vectorized version of the "compute_window_sum" function
    def compute_window_sum_for_all_shifts(w_array):
        # w_array will be shape (2,) if we treat the last axis as the "signature"
        return compute_window_sum(uhat_spectrum, w_array, r, window_size, conformal_radius)

    vectorized_func = np.vectorize(compute_window_sum_for_all_shifts, signature='(2)->()')
    robust_values_vec = vectorized_func(W)

    max_pos = np.unravel_index(robust_values_vec.argmax(), robust_values_vec.shape)
    rel_pos = np.array(max_pos) - np.array([r_optimization, r_optimization])
    return w_nom_star + rel_pos


def eval(field_g, radius, w):
    collection_well = get_collection_masks(radius)
    window_size = np.array(collection_well.shape) // 2
    window = field_g[w[0] - window_size[0]:w[0] + window_size[0] + 1,w[1] - window_size[1]:w[1] + window_size[1] + 1]
    return np.sum(window * collection_well)


def fpo_trial(u_real, uhat_real, uhat_spectrum, radius, conformal_radius): 
    r_pix = int(256 * (radius / (2 * np.pi)))
    w_stars = {
        "Truth": solve_nominal(u_real, r_pix),
        "Nominal": solve_nominal(uhat_real, r_pix),
    }

    # HACK: some solutions go out of bounds -- we could handle this pretty easily by just wrapping according to periodic BC
    # but just ignoring these cases for now
    try:
        w_stars["Robust"] = solve_robust(w_stars["Nominal"], r_pix, uhat_spectrum, conformal_radius)
        Js = {experiment : [eval(u_real, r_pix, w_stars[experiment])] for experiment in w_stars}
        return Js
    except:
        return None


def get_fields(fs_full, us_full, model):
    # 1) Convert to (N_full,2,256,129) float
    fs_full_real = np.stack([fs_full.real, fs_full.imag], axis=1).astype(np.float32)
    us_full_real = np.stack([us_full.real, us_full.imag], axis=1).astype(np.float32)

    fs_full_torch = torch.from_numpy(fs_full_real).to(device)  # shape (N_full,2,256,129)
    us_full_torch = torch.from_numpy(us_full_real).to(device)  # shape (N_full,2,256,129)

    # 2) Evaluate model
    model.eval()
    with torch.no_grad():
        uhat_full_torch = model(fs_full_torch)  # shape (N_full,2,256,129)

    # 3) Convert predictions and ground truth to full Nx x Ny => shape (N_full,256,256)
    uhat_full_np = uhat_full_torch.cpu().numpy()  # shape (N_full,2,256,129)
    us_full_np   = us_full_torch.cpu().numpy()    # shape (N_full,2,256,129)

    # We'll build the half-complex array back to shape (N_full,256,129) complex
    uhat_half_complex = uhat_full_np[:,0] + 1j*uhat_full_np[:,1]  # shape (N_full,256,129)
    us_half_complex   = us_full_np[:,0]   + 1j*us_full_np[:,1]    # shape (N_full,256,129)
    
    us_real     = np.fft.irfft2(us_half_complex).real
    uhats_real  = np.fft.irfft2(uhat_half_complex).real
    uhats_spectrum = rfft2_to_fft2(uhat_half_complex, (256, 256))

    return us_real, uhats_real, uhats_spectrum


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde")
    parser.add_argument("--sample", type=int)
    args = parser.parse_args()

    with open(utils.DATA_FN(args.pde), "rb") as f:
        (fs_np, us_np) = pickle.load(f)

    train_prop = 0.5
    N_full = int(len(fs_np) * (1 - train_prop))
    fs_full, us_full = fs_np[N_full:], us_np[N_full:]

    weight_fn = f"/home/yppatel/fpo/data/{args.pde}/model.pt"
    model = EncoderDecoderNet().to("cuda:0")
    model.load_state_dict(torch.load(weight_fn, weights_only=True))
    model.eval().to("cuda")
    torch.save(model.state_dict(), utils.MODEL_FN(args.pde))

    us_real, uhats_real, uhats_spectrum = get_fields(fs_full, us_full, model)
    r = 0.5
    q = 4_000

    os.makedirs(utils.RESULTS_DIR(args.pde), exist_ok=True)

    for sample_idx in range(len(us_real)):
        results_fn = os.path.join(utils.RESULTS_DIR(args.pde), f"{sample_idx}.csv")
        results = fpo_trial(us_real[sample_idx], uhats_real[sample_idx], uhats_spectrum[sample_idx], radius=r, conformal_radius=q)
        if results is not None:
            results_df = pd.DataFrame.from_dict(results)
            results_df.to_csv(results_fn)