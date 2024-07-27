import argparse
import scipy
import torch
import time
import torch.nn as nn
import os
import pandas as pd
import seaborn as sns
import numpy as np
import pickle
import yaml
import matplotlib.pyplot as plt
import torch.nn.functional as F
from findiff import FinDiff

from fno_utils import FNO2d, FNODatasetSingle

device = "cuda"
sns.set_theme()
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

def get_partials(field):
    # axes here are specified assuming data is [batch, dim_x, dim_y], i.e. batch of 2D scalar fields
    _, dy, dx = 1 / np.array(field.shape)

    partials_fields = [np.array([field])]
    for m in range(1,4):
        perms = []
        for i in range(m + 1):
            perms.append([i, m-i]) # hardcoded for 2D for now, but 3D extension is fairly straightforward
        
        alphas = []
        for perm in perms:
            alpha = []
            if perm[0] != 0:
                alpha.append((1, dx, perm[0]))
            if perm[1] != 0:
                alpha.append((2, dy, perm[1]))
            alphas.append(alpha)
        
        alpha_partials_field = np.array([FinDiff(*alpha)(field) for alpha in alphas])
        partials_fields.append(alpha_partials_field)
    return partials_fields


def c3_metric(field):
    partials_fields = get_partials(field)
    partial_field_maxes = np.array([np.max(partial_field, axis=(0,2,3)) for partial_field in partials_fields])
    return np.sum(partial_field_maxes, axis=0)


def get_scores(train_loader, fno, training_type, score_func):
    scores = []
    for xxbatch, yy, gridbatch in train_loader:
        if training_type == "autoregressive":
            inp_shape = list(xxbatch.shape)
            inp_shape = inp_shape[:-2]
            inp_shape.append(-1)

            xxbatch = xxbatch.reshape(inp_shape)
            yyhat   = fno(xxbatch.to(device), gridbatch.to(device))
            yybatch = yy[:,:,:,10:11,:].to(device)
        else:
            yyhat   = fno(xxbatch[...,0,:].to(device), gridbatch.to(device))
            yybatch = yy.to(device)
        diff = (yyhat[...,0,0] - yybatch[...,0,0]).cpu().detach().numpy()

        if score_func == "c3":
            score_batch = c3_metric(diff)
        elif score_func == "l2":
            score_batch = np.linalg.norm(diff.reshape(diff.shape[0], -1), ord=2, axis=-1)
        elif score_func == "linf":
            score_batch = np.linalg.norm(diff.reshape(diff.shape[0], -1), ord=np.inf, axis=-1)
        scores.append(score_batch)
    return np.concatenate(scores)


def plot_calibration(alphas, ax, downsampling_to_cp_quantiles, downsampling_to_test_scores, pde_title):
    downsampling_to_resolution = {
        1 : r"$128\times 128$", 
        2 : r"$64\times 64$", 
        4 : r"$32\times 32$",
    }
    res_to_cov = {}
    for downsampling in downsampling_to_cp_quantiles:
        quantiles   = downsampling_to_cp_quantiles[downsampling]
        test_scores = downsampling_to_test_scores[downsampling]
        coverages = [np.sum(test_scores < quantile) / len(test_scores) for quantile in quantiles]
        res_to_cov[downsampling_to_resolution[downsampling]] = coverages[::-1]
    df = pd.DataFrame.from_dict(res_to_cov)
    df[r"$\alpha$"] = alphas
    sns.lineplot(df, palette="flare", ax=ax)
    
    ax.set_title(r"$\mathrm{" + pde_title + r"}$")
    ax.set_xlabel(r"$\mathrm{Expected\ Coverage}\ (1-\alpha)$")
    ax.legend_ = None # have just a single legend for the figure
    

def test_calibration(alphas, score_func, pde, ax):
    cfg_fn = os.path.join("experiments", f"config_{pde}.yaml")
    with open(cfg_fn, "r") as f:
        cfg = yaml.safe_load(f)
    
    pde_name = cfg["filename"].split(".h")[0]
    model_weights = torch.load(os.path.join("experiments", f"{pde_name}_FNO.pt"), map_location=torch.device('cuda'))

    fno = FNO2d(
        num_channels=cfg["num_channels"], 
        modes1=cfg["modes"], 
        modes2=cfg["modes"], 
        width=cfg["width"], 
        initial_step=cfg["initial_step"]).to("cuda")
    fno.load_state_dict(model_weights["model_state_dict"])

    downsampling_to_scores = {1 : [], 2 : [], 4 : []}
    for downsampling in downsampling_to_scores:
        batch_size = 25
        train_data = FNODatasetSingle(filename=os.path.join("experiments", cfg["filename"]), reduced_resolution=downsampling)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)    
        downsampling_to_scores[downsampling] = get_scores(train_loader, fno, cfg["training_type"], score_func)

    downsampling_to_cp_quantiles = {}
    downsampling_to_test_scores  = {}
    for downsampling in downsampling_to_scores:
        scores = downsampling_to_scores[downsampling]
        cal_scores, test_scores = scores[:-100], scores[-100:]
        quantiles = [np.quantile(cal_scores, q = 1-alpha) for alpha in alphas]
        
        downsampling_to_cp_quantiles[downsampling] = quantiles
        downsampling_to_test_scores[downsampling]  = test_scores

    pde_name_to_title = {
        "darcy": r"2D\ Darcy\ Flow",
        "diffreact": r"2D\ Diffusion\ Reaction",
        "rdb": r"2D\ Shallow\ Water",
    }
    plot_calibration(alphas, ax, downsampling_to_cp_quantiles, downsampling_to_test_scores, pde_name_to_title[pde])
    return downsampling_to_cp_quantiles


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    for score_func in ["c3", "l2", "linf"]:
        fig, axs = plt.subplots(1, 3, figsize=(18,6))
        for pde, ax in zip(["darcy", "diffreact", "rdb"], axs):
            print(f"Calibrating {score_func}: {pde}")
            alphas = np.arange(0, 1, 0.05)
            pde_cp_quantiles = test_calibration(alphas, score_func, pde, ax)

            result_fn = os.path.join("experiments", f"quantiles_{pde}_{score_func}.csv")
            scores_df = pd.DataFrame.from_dict(pde_cp_quantiles).set_index(alphas)
            scores_df.to_csv(result_fn)

        axs[0].set_ylabel(r"$\mathrm{Empirical\ Coverage}$")
        handles, labels = axs[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        fig.tight_layout()
        
        plt.savefig(os.path.join("results", f"calibration_{score_func}.png"))