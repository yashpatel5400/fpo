import argparse
import einops
import os
import torch

import dedalus.public as d3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns

from spec_op import SpecOp
import utils

device = "cuda:0"

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

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

sns.set_theme()

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


def get_sobolev_weights(s, gamma, shape):
    coords = cartesian_product(
        np.array(range(shape[0])), 
        np.array(range(shape[1]))
    ).reshape((shape[0], shape[1], 2))
    ks = np.sum(coords, axis=-1)
    d = len(shape)
    return (1 + ks ** (2 * d)) ** (s - gamma)


def get_sobolev_res(u, u_hat, K):
    sobolev_scaling = get_sobolev_weights(s=2, gamma=1, shape=(256,256))
    full_sobolev_residual = (sobolev_scaling * (u - u_hat) ** 2)[:,:K,:K]
    return full_sobolev_residual.reshape(-1, np.prod(full_sobolev_residual.shape[1:])).sum(axis=-1)


def calibration(us, u_hats, cutoff, viz, pde):
    sobolev_scores_full  = get_sobolev_res(us, u_hats, K=256)
    sobolev_scores_trunc = get_sobolev_res(us, u_hats, K=cutoff)

    alphas = np.arange(0.025, 1.0, 0.05)

    q_hats = []
    coverages_full  = []
    coverages_trunc = []

    margin = 10
    N_cal = 75

    for alpha in alphas:
        q_hat = np.quantile(sobolev_scores_trunc[:N_cal], 1-alpha)
        q_hats.append(q_hat)
        coverages_full.append(np.sum(sobolev_scores_full[N_cal:] < q_hat + margin) / len(sobolev_scores_full[N_cal:]))
        coverages_trunc.append(np.sum(sobolev_scores_trunc[N_cal:] < q_hat) / len(sobolev_scores_trunc[N_cal:]))

    if viz:
        plt.title(r"$\mathrm{" + pde.capitalize() + r"\ Calibration\ Curve}$")
        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"$\mathrm{Coverage}$")
        
        plt.plot(alphas, alphas, label=r"$\mathrm{Reference}$")
        plt.plot(1-alphas, coverages_full, label=r"$\mathrm{Full}$")
        plt.plot(1-alphas, coverages_trunc, label=r"$\mathrm{Truncated}$")
        plt.legend()

        result_fn = os.path.join(utils.RESULTS_DIR(pde), "calibration.png")
        plt.savefig(result_fn)


def get_single_basis(k, coord):
    # using RealFourier basis, which has elements [cos(0*x), -sin(0*x), cos(1*x), -sin(1*x),...]
    if k % 2 == 0:
        return np.cos((k // 2) * coord)
    return -np.sin((k // 2) * coord)


def get_reference_field():
    dtype = np.float64
    coords = d3.CartesianCoordinates("x", "y")
    dist   = d3.Distributor(coords, dtype=dtype)
    xbasis = d3.RealFourier(coords["x"], size=Nx, bounds=(0, Lx))
    ybasis = d3.RealFourier(coords["y"], size=Ny, bounds=(0, Ly))
    return dist.Field(name='u', bases=(xbasis, ybasis))


def compute_robust_field(u_hat, Lx, Ly, Nx, Ny, radius, cutoff):
    print(u_hat.shape)
    # --- Setup --- #
    diam = 2 * radius + 1
    coords = cartesian_product(
        np.array(range(diam)), 
        np.array(range(diam))
    ).reshape((diam, diam, 2))

    offsets = coords - np.array([radius, radius])
    lens = np.linalg.norm(offsets, axis=-1)

    collection_well = (lens <= radius).astype(float)
    well_border = (np.abs(lens - radius) < .5).astype(float)

    normals = offsets / np.expand_dims(lens, axis=-1)
    well_normals = np.expand_dims(well_border, axis=-1) * normals
    well_normals = np.nan_to_num(well_normals)

    u = get_reference_field()

    # --- Compute windows for "collection" based on above setup --- #
    x = np.arange(0, Lx, Lx / Nx)
    y = np.arange(0, Ly, Ly / Ny)
    coords = cartesian_product(x, y).reshape((Nx, Ny, 2))

    k1 = np.array(range(Nx))
    k2 = np.array(range(Ny))
    modes = cartesian_product(k1, k2).reshape((Nx, Ny, 2))

    basis_grids = np.zeros((Nx, Ny, Nx, Ny)) # dimensions are k1, k2 -> x, y grid
    for k1 in range(256):
        for k2 in range(256):
            basis_x = get_single_basis(k1, x)
            basis_y = get_single_basis(k2, y)
            basis_grids[k1,k2] = np.expand_dims(basis_x, axis=-1) @ np.expand_dims(basis_y, axis=-1).T
        
    basis_grids_cut = basis_grids[:cutoff,:cutoff]
    window_size = np.array(collection_well.shape) // 2

    w0 = np.arange(window_size[0], 256-window_size[0], 1)
    w1 = np.arange(window_size[1], 256-window_size[1], 1)
    ws = cartesian_product(w0, w1).reshape((len(w0), len(w1), 2)).reshape(-1,2)
    
    # Extract the row and column starts:
    i = ws[:, 0]  # shape (K,)
    j = ws[:, 1]  # shape (K,)

    # Build the indices for the 3rd and 4th dimensions:
    #   i_idx will have shape (K, 15, 15) -- row indices for each patch
    #   j_idx will have shape (K, 15, 15) -- column indices for each patch
    i_idx = i[:, None, None] + np.arange(-window_size[0],window_size[0]+1)[None, :, None]  # shape (K, 15, 1)
    j_idx = j[:, None, None] + np.arange(-window_size[1],window_size[1]+1)[None, None, :]  # shape (K, 1, 15)

    # Broadcast i_idx, j_idx together to (K, 15, 15)
    # (Numpy does this automatically in the indexing step if they can broadcast.)
    # Perform advanced indexing along dimensions 2 and 3:
    #   - Dimension 0 and 1 of X remain untouched (256, 256).
    #   - Dimensions 2 and 3 are replaced by the new shape (K, 15, 15).
    patches = basis_grids_cut[:, :, i_idx, j_idx]  # shape (256, 256, K, 15, 15)
    patches = np.moveaxis(patches, 2, 0)
    scaled_patches = patches * einops.repeat(collection_well, "h w -> n1 n2 n3 h w", n1=1, n2=1, n3=1)

    Phi_ws = einops.reduce(scaled_patches, "n k1 k2 x y -> n k1 k2", "sum")
    
    # --- Perform optimization --- #
    lambda_ = 10
    d = 2       # ambient dimension space (i.e. space over which the vector field u is defined)
    s = 2       # smoothness of PDE (s-Sobolev space)
    gamma = 1.0 # gamma defines CP norm; must be between [1,...,s-1] from theory

    k_norms = np.linalg.norm(modes, axis=-1, ord=1)
    norm_factor = 1 / (2 * lambda_ * (1 + (k_norms) ** (2 * d)) ** (s - gamma))
    u_c_deltas = Phi_ws[...,:cutoff,:cutoff] * norm_factor[:cutoff,:cutoff]
    u_f_robusts = np.expand_dims(u_hat[:cutoff,:cutoff].copy(), axis=0) - u_c_deltas

    rob_vals = []
    for w_idx in range(len(u_f_robusts)):
        u_f_c_rob = np.zeros(u_hat.shape)
        u_f_c_rob[:cutoff,:cutoff] = u_f_robusts[w_idx]

        u["c"] = u_f_c_rob
        u_g_rob = u["g"].copy()
        rob_vals.append(u_g_rob[ws[w_idx][0],ws[w_idx][1]])
    rob_vals = np.array(rob_vals)
    return rob_vals.reshape((len(w0), len(w1)))


def viz_fields(pde, u, u_hat, u_rob):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  
    axs[0].imshow(u)
    axs[0].set_title(r"$u \mathrm{\ Field}$")
    
    axs[1].imshow(u_hat)
    axs[1].set_title(r"$\widehat{u} \mathrm{\ Field}$")
    
    axs[2].imshow(u_rob)
    axs[2].set_title(r"$u_{\mathrm{rob}} \mathrm{\ Field}$")
    
    result_fn = os.path.join(utils.RESULTS_DIR(pde), "fields.png")
    plt.savefig(result_fn)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde")
    parser.add_argument("--sample", type=int)
    parser.add_argument("--cutoff", type=int, default=8)
    args = parser.parse_args()

    if   args.pde == "poisson":        model = SpecOp(cutoff=args.cutoff).to(device)
    elif args.pde == "navier_stokes":  model = SpecOp(cutoff=args.cutoff).to(device)
    model.load_state_dict(torch.load(utils.MODEL_FN(args.pde), weights_only=True))
    model.eval().to("cuda")

    fs, us = utils.get_data(args.pde, train=False)
    fs = fs.reshape((-1, 1, 256, 256))
    us = us.reshape((-1, 1, 256, 256))

    u_hats = model(fs[...,:args.cutoff,:args.cutoff].to("cuda").to(torch.float32)).reshape((-1,1,args.cutoff,args.cutoff)).detach().cpu().numpy()
    u_hats_full = np.zeros(us.shape)
    u_hats_full[...,:args.cutoff,:args.cutoff] = u_hats
    us = us.detach().cpu().numpy()

    # --- Perform conformal calibration   --- #
    calibration(us, u_hats_full, args.cutoff, viz=True, pde=args.pde)

    # --- Define robust optimization task --- #
    radius = 10 # NOTE: this is in *index* space, i.e. not in the actual discretized space units
    Lx, Ly = 2 * np.pi, 2 * np.pi
    Nx, Ny = 256, 256
    
    robust_field = compute_robust_field(u_hats_full[args.sample,0], Lx, Ly, Nx, Ny, radius, cutoff=args.cutoff)
    viz_fields(args.pde, us[args.sample,0], u_hats_full[args.sample,0], robust_field)