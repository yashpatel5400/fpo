import argparse
import numpy as np
import pickle
import pandas as pd
import os

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops

import utils
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from complex_spec_op import construct_dataset, EncoderDecoderNet

sns.set_theme()

mpl.rcParams['text.usetex'] = True
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'
mpl.rcParams['figure.figsize'] = (12,8)

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

device = "cuda:0"

def sobolev_norm_2d_full_batch(f_hat_full_batch, s: float):
    """
    Vectorized Sobolev norm for a batch of full complex Fourier arrays in unshifted order.

    f_hat_full_batch: np.ndarray of shape (batch, Nx, Ny), dtype=complex
      Each [i,:,:] is one sample's full Fourier array in unshifted order.
    s: float, the Sobolev exponent.

    Returns
    -------
    norms : np.ndarray of shape (batch,)
      The H^s(T^2) norm for each sample.
    """
    batch_size, Nx, Ny = f_hat_full_batch.shape

    # Precompute (1 + k1^2 + k2^2)^s for each (k1, k2) in unshifted indexing
    # shape (Nx,Ny)
    k1_vals = np.arange(Nx)
    # shift negative frequencies
    k1_vals[k1_vals > Nx//2] -= Nx
    k2_vals = np.arange(Ny)
    k2_vals[k2_vals > Ny//2] -= Ny

    # Build 2D mesh
    K1, K2 = np.meshgrid(k1_vals, k2_vals, indexing='ij')  # shape (Nx, Ny)

    k_sq = K1**2 + K2**2  # shape (Nx, Ny)
    factor = (1.0 + k_sq)**s  # shape (Nx,Ny)

    # We'll do sum_{k1,k2} factor * |f_hat|^2 for each sample
    # f_hat_full_batch has shape (batch,Nx,Ny), factor has shape (Nx,Ny).
    # We want to multiply each sample by factor and sum.

    # absolute value squared
    abs_sq = np.abs(f_hat_full_batch)**2  # shape (batch,Nx,Ny)

    # multiply
    # shape (batch,Nx,Ny)
    weighted = abs_sq * factor  # broadcasting factor (Nx,Ny) over batch dimension

    # sum over Nx,Ny
    sums = np.sum(weighted, axis=(1,2))  # shape (batch,)

    # final norm is sqrt
    norms = np.sqrt(sums)
    return norms


def rfft2_to_fft2(rfft_result, input_shape):
    """
    Converts the result of numpy.fft.rfft2 to the equivalent result of numpy.fft.fft2.

    Args:
        rfft_result (numpy.ndarray): The result of numpy.fft.rfft2.
        input_shape (tuple): The shape of the original input array to rfft2.

    Returns:
        numpy.ndarray: The equivalent result of numpy.fft.fft2.
    """
    N, rows, cols = (rfft_result.shape[0], input_shape[0], input_shape[1])
    full_fft = np.zeros((N, rows, cols), dtype=np.complex128)
    full_fft[:,:,:cols // 2 + 1] = rfft_result
    full_fft[:,:,cols-1:cols // 2:-1]  = np.conjugate(rfft_result)[:,:,1:-1]
    return full_fft


def compute_scores(model, fs_full, us_full, K_trunc):
    # Suppose fs_full, us_full are shape (N_full, 256,129), np.complex128
    N_full = fs_full.shape[0]
    
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
    fs_full_np   = fs_full_torch.cpu().numpy()    # shape (N_full,2,256,129)

    # We'll build the half-complex array back to shape (N_full,256,129) complex
    uhat_half_complex = uhat_full_np[:,0] + 1j*uhat_full_np[:,1]  # shape (N_full,256,129)
    us_half_complex   = us_full_np[:,0]   + 1j*us_full_np[:,1]    # shape (N_full,256,129)
    fs_half_complex   = fs_full_np[:,0]   + 1j*fs_full_np[:,1]    # shape (N_full,256,129)

    # Truncated field representations
    uhat_half_complex_trunc = np.zeros(uhat_half_complex.shape).astype(np.complex128)
    uhat_half_complex_trunc[...,:,:K_trunc] = uhat_half_complex[...,:,:K_trunc].copy()

    us_half_complex_trunc = np.zeros(us_half_complex.shape).astype(np.complex128)
    us_half_complex_trunc[...,:,:K_trunc] = us_half_complex[...,:,:K_trunc].copy()

    fs_half_complex_trunc = fs_half_complex.copy()
    fs_half_complex_trunc[...,:,:K_trunc] = 0

    # 4) Differences in full Fourier domain
    diffs_half_complex = uhat_half_complex - us_half_complex
    diffs_half_complex_trunc = uhat_half_complex_trunc - us_half_complex_trunc

    # Now we do rfft2_to_fft2 for each item. We'll vectorize by a loop or list comprehension:
    # shape (N_full,256,256) complex
    scores_full  = sobolev_norm_2d_full_batch(rfft2_to_fft2(diffs_half_complex, (256, 256)), s=1.0)
    scores_trunc = sobolev_norm_2d_full_batch(rfft2_to_fft2(diffs_half_complex_trunc, (256, 256)), s=1.0)
    margins  = 2 * sobolev_norm_2d_full_batch(rfft2_to_fft2(fs_half_complex_trunc, (256, 256)), s=-1.0)
    
    # 5) Sobolev norms => shape (N_full,)
    return scores_full, scores_trunc, margins


def calibrate(fs_full, us_full, model):
    test_prop = 0.5
    N_test = int(test_prop * len(fs_full))

    truncations_to_coverages = {}
    for K_trunc in range(4, 17, 4):
        scores_full, scores_trunc, margins = compute_scores(model, fs_full, us_full, K_trunc=K_trunc)

        scores_cal, scores_test_trunc = scores_trunc[:N_test], scores_trunc[N_test:]
        _, scores_test_full = scores_full[:N_test], scores_full[N_test:]
        _, margins_test = margins[:N_test], margins[N_test:]

        alphas = np.arange(0.05, 0.951, 0.025)  # 0.05, 0.075, ... , 0.95
        coverages_trunc, coverages_full = [], []
        for alpha in alphas:
            qval = np.quantile(scores_cal, 1 - alpha)  # 1 - alpha
            coverages_trunc.append((scores_test_trunc < qval).sum() / len(scores_test_trunc))
            coverages_full.append((scores_test_full < qval + margins_test).sum() / len(scores_test_full))
        truncations_to_coverages[K_trunc] = coverages_full
    
    plt.plot(alphas, alphas)
    plt.plot(1 - alphas, coverages_trunc, label=r"$\mathrm{Truncated}$")
    
    for K_trunc in truncations_to_coverages:
        plt.plot(1 - alphas, truncations_to_coverages[K_trunc], label=r"$\mathrm{Full}, K_{\mathrm{Trunc}}=" + str(K_trunc) + "$")
    
    plt.xlabel(r"$1-\alpha$")
    plt.ylabel(r"$\mathrm{Coverage}$")
    plt.legend()

    plt.savefig("calibration.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde")
    parser.add_argument("--cutoff", type=int, default=8)
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

    calibrate(fs_full, us_full, model)