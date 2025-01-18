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

from spec_op import SpecOp
import utils


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


def get_reference_field(Nx, Ny, Lx, Ly):
    dtype = np.float64

    # Bases
    coords = d3.CartesianCoordinates("x", "y")
    dist   = d3.Distributor(coords, dtype=dtype)
    xbasis = d3.RealFourier(coords["x"], size=Nx, bounds=(0, Lx))
    ybasis = d3.RealFourier(coords["y"], size=Ny, bounds=(0, Ly))

    # Fields
    return dist.Field(name='u', bases=(xbasis, ybasis))


def get_single_basis(k, coord):
    # using RealFourier basis, which has elements [cos(0*x), -sin(0*x), cos(1*x), -sin(1*x),...]
    if k % 2 == 0:
        return np.cos((k // 2) * coord)
    return -np.sin((k // 2) * coord)


def get_basis_fields(Lx, Ly, Nx, Ny):
    x = np.arange(0, Lx, Lx / Nx)
    y = np.arange(0, Ly, Ly / Ny)
    basis_fields = np.zeros((Nx, Ny, Nx, Ny)) # dimensions are k1, k2 -> x, y grid
    for k1 in range(Nx):
        for k2 in range(Ny):
            basis_x = get_single_basis(k1, x)
            basis_y = get_single_basis(k2, y)
            basis_fields[k1,k2] = np.expand_dims(basis_x, axis=-1) @ np.expand_dims(basis_y, axis=-1).T
    return basis_fields


def get_collection_window(radius):
    # NOTE: radius is specified in *index* space, i.e. not in the actual discretized space units
    diam = 2 * radius + 1
    coords = cartesian_product(np.array(range(diam)), np.array(range(diam))).reshape((diam, diam, 2))
    offsets = coords - np.array([radius, radius])
    return offsets


def get_collection_masks(radius):
    offsets = get_collection_window(radius)
    lens = np.linalg.norm(offsets, axis=-1)

    collection_well = (lens <= radius).astype(float)
    return collection_well


def get_collection_border_normals(radius):
    offsets = get_collection_window(radius)
    lens = np.linalg.norm(offsets, axis=-1)
    
    well_border = (np.abs(lens - radius) < .5).astype(float) # HACK: .5 seems to work well in practice; no real reason (that I can see?) for it
    normals = offsets / np.expand_dims(lens, axis=-1)
    well_normals = np.expand_dims(well_border, axis=-1) * normals
    well_normals = np.nan_to_num(well_normals)
    return well_border, well_normals


def solve_nominal(field_coeff, radius, Lx, Ly):
    collection_well = get_collection_masks(radius)

    Nx, Ny = field_coeff.shape
    field = get_reference_field(Nx, Ny, Lx, Ly)
    field["c"] = field_coeff
    field_g = field["g"].copy()

    conv_offset = np.array(collection_well.shape) // 2
    collection = signal.convolve2d(field_g, collection_well, mode='valid')
    unadj_w_star = np.unravel_index(np.argmax(collection, axis=None), collection.shape)
    w_star = unadj_w_star + conv_offset
    return w_star


def viz_results(field, w_star, w_nom, w_rob):
    # Create a figure and axes
    fig, ax = plt.subplots()

    ax.imshow(field)

    circle = Circle(w_star[::-1], 1, color='g')
    ax.add_patch(circle)
    ax.set_aspect('equal')

    circle = Circle(w_nom[::-1], 1, color='r')
    ax.add_patch(circle)
    ax.set_aspect('equal')

    circle = Circle(w_rob[::-1], 1, color='b')
    ax.add_patch(circle)
    ax.set_aspect('equal')
    plt.imsave("result.png")


def solve_robust(field_coeff, radius, Lx, Ly):
    # --------- Problem setup --------- #
    Nx, Ny = field_coeff.shape
    basis_fields = get_basis_fields(Lx, Ly, Nx, Ny)
    field = get_reference_field(Nx, Ny, Lx, Ly)

    k1 = np.array(range(Nx))
    k2 = np.array(range(Ny))
    modes = cartesian_product(k1, k2).reshape((Nx, Ny, 2))

    collection_well = get_collection_masks(radius)
    well_border, well_normals = get_collection_border_normals(radius)

    # ------- Optimization loop ------- #
    eta = 0.5
    T   = 100
    
    d = 2     # ambient dimension space (i.e. space over which the vector field u is defined)
    s = 2     # smoothness of PDE (s-Sobolev space)
    gamma = 1 # gamma defines CP norm; must be between [1,...,s-1] from theory

    lambda_ = 20
    k_norms = np.linalg.norm(modes, axis=-1, ord=1)
    norm_factor = 1 / (2 * lambda_ * (1 + (k_norms) ** (2 * d)) ** (s - gamma))

    w = solve_nominal(field_coeff, radius, Lx, Ly) # initialize with nominal solution
    ws = [w]
    field_g_robs = []

    for t in range(T):
        # have to do with np.take to get wrapped behavior for torus
        # tmp     = basis_grids.take(range(w[0] - window_size[0],w[0] + window_size[0] + 1), mode='wrap', axis=2)
        # windows = tmp.take(range(w[1] - window_size[1],w[1] + window_size[1] + 1), mode='wrap', axis=3)
        window_size = np.array(collection_well.shape) // 2
        windows = basis_fields[...,w[0] - window_size[0]:w[0] + window_size[0] + 1,w[1] - window_size[1]:w[1] + window_size[1] + 1]
        scaled_windows = windows * np.expand_dims(np.expand_dims(collection_well, axis=0), axis=0)
        Phi_w = einops.reduce(scaled_windows, "k1 k2 x y -> k1 k2", "sum")

        field_c_delta = Phi_w * norm_factor
        field["c"] = field_coeff - field_c_delta
        field_g_rob = field["g"].copy()
        field_g_robs.append(field_g_rob)

        # take gradient step with the computed robust field
        well_window = field_g_rob[w[0] - window_size[0]:w[0] + window_size[0] + 1,w[1] - window_size[1]:w[1] + window_size[1] + 1]
        grad = np.array([
            np.sum(well_window * well_normals[...,0] * well_border),
            np.sum(well_window * well_normals[...,1] * well_border),
        ])

        w = np.round(w + eta * grad).astype(int)
        w = np.mod(w, np.array(field_g_rob.shape))
        ws.append(w)
    return w


def eval(field_coeff, radius, w, Lx, Ly):
    Nx, Ny = field_coeff.shape
    field = get_reference_field(Nx, Ny, Lx, Ly)
    field["c"] = field_coeff
    field_g = field["g"].copy()
    collection_well = get_collection_masks(radius)

    window_size = np.array(collection_well.shape) // 2
    window = field_g[w[0] - window_size[0]:w[0] + window_size[0] + 1,w[1] - window_size[1]:w[1] + window_size[1] + 1]
    return np.sum(window * collection_well)


def fpo_trial(u_c, u_c_hat, radius):
    Lx, Ly = 2 * np.pi, 2 * np.pi
    w_stars = {
        "Truth": solve_nominal(u_c, radius, Lx, Ly),
        "Nominal": solve_nominal(u_c_hat, radius, Lx, Ly),
        "Robust": solve_robust(u_c_hat, radius, Lx, Ly),
    }
    Js = {experiment : [eval(u_c, radius, w_stars[experiment], Lx, Ly)] for experiment in w_stars}
    return Js


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde")
    parser.add_argument("--sample", type=int)
    args = parser.parse_args()

    fs, us = utils.get_data(args.pde, train=False)
    model = SpecOp().to("cuda")
    model.load_state_dict(torch.load(utils.MODEL_FN(args.pde), weights_only=True))
    model.eval().to("cuda")

    f_c = fs[args.sample].reshape((256,256))
    u_c = us[args.sample].reshape((256,256)).detach().cpu().numpy().copy()
    u_c_hat = model(f_c.unsqueeze(0).unsqueeze(0).to("cuda").to(torch.float32)).reshape((256,256)).detach().cpu().numpy().copy()
    
    os.makedirs(utils.RESULTS_DIR(args.pde), exist_ok=True)
    results_fn = os.path.join(utils.RESULTS_DIR(args.pde), f"{args.sample}.csv")
    results = fpo_trial(u_c, u_c_hat, radius=15)
    results_df = pd.DataFrame.from_dict(results)
    results_df.to_csv(results_fn)