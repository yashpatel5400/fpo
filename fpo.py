import argparse
import cvxpy as cp
import scipy
import torch
import time
import torch.nn as nn
import os
import pandas as pd
import numpy as np
import pickle
import yaml
import matplotlib.pyplot as plt
import torch.nn.functional as F

from calibrate import get_partials
from fno_utils import FNO2d, FNODatasetSingle
from wavelet import WaveletBasis, get_disc_grid    


def optimize(uhat, quantile, r, score_func):
    """Primary driver function of FPO pipeline -- solves the robust problem of min_{w} max_{u in B_q(uhat)} J(u, w)
    for J of the form prescribed in the paper and the ball defined in C^3 norm. 

    Arguments:
        uhat -- Function to use as the center of the function space ball
        quantile -- Radius of the ball (in C^3 norm). If given is None, the nominal solution is 
            provided instead (i.e. simply taking the prescribed uhat to be true)
        r -- Radius of the "collector" in the problem specification
        score_func -- Name of score function to use for the constraint (can be None if quantile is None)
    """
    wavelet_basis = WaveletBasis()

    # amortized definition of basis partials for Sobolev norm computation
    uhat_coeffs = wavelet_basis.get_decomp(uhat)
    shaped_basis = wavelet_basis.basis_func.reshape(wavelet_basis.basis_func.shape[1], uhat.shape[0], uhat.shape[1])
    partials = get_partials(shaped_basis)
    partials = [
        np.transpose(np.array([partial.reshape(partial.shape[0], -1) for partial in partial_order]), (0, 2, 1)) 
        for partial_order in partials
    ]
    x_grid = wavelet_basis.x_grid.reshape(uhat.shape[0], uhat.shape[1], -1)

    # problem specification (constant over optimization)
    eta       = 5e-4 if score_func == "c3" else 1e-3
    max_iters = 200

    w = np.array([0.6, 0.6])
    for iter in range(max_iters):
        # ---- Compute u^* using parametric wavelet formulation in robust formulation
        if quantile is not None:
            if score_func == "c3":
                u_coeff = cp.Variable(uhat_coeffs.shape)

                integral_mask = (np.linalg.norm(wavelet_basis.x_grid - w, axis=1) < r).astype(np.int8).reshape(uhat.shape)
                psi_w = np.sum(shaped_basis * integral_mask, axis=(1,2))
                objective = cp.Minimize(-u_coeff @ psi_w)

                constraints = [
                    cp.sum([
                        cp.max(cp.hstack([
                            cp.abs(partial @ (uhat_coeffs - u_coeff)) for partial in partial_order
                        ])) for partial_order in partials
                    ]) <= quantile
                ]

                prob = cp.Problem(objective, constraints)
                obj  = prob.solve()
                u_star = (wavelet_basis.basis_func @ u_coeff.value).reshape(uhat.shape)
            else:
                yhat = uhat.flatten()
                w_mask = (np.linalg.norm(wavelet_basis.x_grid - w, axis=1) < r).astype(np.int8)
                norm_ord = 2 if score_func == "l2" else "inf"

                u = cp.Variable(yhat.shape)
                objective = cp.Minimize(-u @ w_mask)
                constraints = [cp.norm(u - yhat, norm_ord) <= quantile]
                prob = cp.Problem(objective, constraints)
                obj  = prob.solve()
                u_star = u.value.reshape(uhat.shape)
        else:
            u_star = uhat

        # ---- Update to w^(t+1) using u^*
        eps = 5e-2
        bd_mask = (np.abs(np.linalg.norm(x_grid - w, axis=-1) - r) < eps).astype(np.int8)
        w_grad_field = np.expand_dims(bd_mask, axis=-1) * (w - x_grid) * np.expand_dims(u_star, axis=-1)
        w_grad = np.sum(w_grad_field, axis=(0, 1))
        w = w - eta * w_grad
        print(f"{iter} : w : {w} -- w_grad : {np.linalg.norm(w_grad)}")
    return w


def eval_regret_ratio(u, w, w_star, r):
    x_grid = get_disc_grid(u.shape[0])
    w_mask = (np.linalg.norm(x_grid - w, axis=1) < r).astype(np.int8).reshape(u.shape)
    w_star_mask = (np.linalg.norm(x_grid - w_star, axis=1) < r).astype(np.int8).reshape(u.shape)
    
    prop_obj = np.sum(u * w_mask)
    opt_obj  = np.sum(u * w_star_mask)
    return (prop_obj - opt_obj) / opt_obj


def fpo(cfg):
    pde_name = cfg["filename"].split(".h")[0]
    num_trials = 1
    trial_size = 1
    train_data = FNODatasetSingle(filename=os.path.join("experiments", cfg["filename"]), reduced_resolution=cfg["downsampling"])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=trial_size)
    model_weights = torch.load(os.path.join("experiments", f"{pde_name}_FNO.pt"), map_location="cuda:0")

    r = 0.05 # collection well radius
    regret_ratios = []
    for _ in range(num_trials):
        fno = FNO2d(
            num_channels=cfg["num_channels"], 
            modes1=cfg["modes"], 
            modes2=cfg["modes"], 
            width=cfg["width"], 
            initial_step=cfg["initial_step"]).to("cuda")
        fno.load_state_dict(model_weights["model_state_dict"])

        device = "cuda"
        for xxbatch, uu, gridbatch in train_loader:
            if cfg["training_type"] == "autoregressive":
                inp_shape = list(xxbatch.shape)
                inp_shape = inp_shape[:-2]
                inp_shape.append(-1)

                xxbatch = xxbatch.reshape(inp_shape)
                uuhat   = fno(xxbatch.to(device), gridbatch.to(device))
                uubatch = uu[:,:,:,10:11,:].to(device)
            else:
                uuhat   = fno(xxbatch[...,0,:].to(device), gridbatch.to(device))
                uubatch = uu.to(device)

        uhat = uuhat[0,...,0,0].detach().cpu().numpy()
        u    = uubatch[0,...,0,0].detach().cpu().numpy()
        
        w = optimize(uhat, cfg["cp_quantile"], r, cfg["score_func"])
        w_star = optimize(u, None, r, None)
        regret_ratios.append(eval_regret_ratio(u, w, w_star, r))
    return np.array(regret_ratios)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde", choices=["darcy", "diffreact", "rdb"])
    parser.add_argument("--score_func", choices=["c3", "l2", "linf"])
    parser.add_argument("--nominal", action="store_true")
    parser.add_argument("--downsampling", type=int, default=1)
    args = parser.parse_args()

    cfg_fn = os.path.join("experiments", f"config_{args.pde}.yaml")
    with open(cfg_fn, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["score_func"]   = args.score_func
    cfg["downsampling"] = args.downsampling

    if args.nominal:
        cfg["cp_quantile"] = None
    else:
        alpha = 0.95
        cp_quantiles_fn = os.path.join("experiments", f"quantiles_{args.pde}_{args.score_func}.csv")
        cfg["cp_quantile"] = pd.read_csv(cp_quantiles_fn)[str(cfg["downsampling"])].values[1]
    regret_ratios = fpo(cfg)

    result_fn = os.path.join("results", f"regret_{args.pde}_{args.score_func}.pkl")
    with open(result_fn, "wb") as f:
        pickle.dump(regret_ratios, f)