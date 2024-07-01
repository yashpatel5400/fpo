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

def c3_metric(field):
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
    partial_field_maxes = np.array([np.max(partial_field, axis=(0,2,3)) for partial_field in partials_fields])
    return np.sum(partial_field_maxes, axis=0)

def get_scores(train_loader, fno, training_type):
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
        diff        = (yyhat[...,0,0] - yybatch[...,0,0]).cpu().detach().numpy()
        score_batch = c3_metric(diff)
        scores.append(score_batch)
    return np.concatenate(scores)

def plot_scores(ax, scores, pde_title):
    num_scores_per_res = len(scores[0]) // 3
    scores_per_res = [
        scores[0][:num_scores_per_res], 
        scores[1][num_scores_per_res:2 * num_scores_per_res], 
        scores[2][2 * num_scores_per_res:]
    ]
    for res_scores, resolution in zip(scores_per_res, [r"128\times 128", r"64\times 64", r"32\times 32"]):
        cal_scores, test_scores = res_scores[:-100], res_scores[-100:]

        alphas = np.arange(0, 1, 0.05)
        coverages = []
        for alpha in alphas:
            q = np.quantile(cal_scores, q = 1-alpha)
            coverages.append(np.sum(test_scores < q) / len(test_scores))
        sns.lineplot(x=(1-alphas), y=coverages, label="$\mathrm{" + resolution + "}$", ax=ax)
    sns.lineplot(x=(1-alphas), y=(1-alphas), linestyle='--', ax=ax)
    ax.set_title(r"$\mathrm{" + pde_title + r"}$")
    ax.set_xlabel(r"$\mathrm{Expected\ Coverage}\ (1-\alpha)$")
    ax.legend_ = None # have just a single legend for the figure
    
def test_calibration(pde, ax):
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

    scores = []
    downsampling = [1,2,4]
    for resolution in downsampling:
        batch_size = 25
        train_data = FNODatasetSingle(filename=os.path.join("experiments", cfg["filename"]), reduced_resolution=resolution)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)    
        scores.append(get_scores(train_loader, fno, cfg["training_type"]))

    pde_name_to_title = {
        "darcy": r"2D\ Darcy\ Flow",
        "diffreact": r"2D\ Diffusion\ Reaction",
        "rdb": r"2D\ Shallow\ Water",
    }

    plot_scores(ax, scores, pde_name_to_title[pde])

if __name__ == "__main__":
    fig, axs = plt.subplots(1, 3, figsize=(18,6))
    for pde, ax in zip(["darcy", "diffreact", "rdb"], axs):
        test_calibration(pde, ax)
    axs[0].set_ylabel(r"$\mathrm{Empirical\ Coverage}$")
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.tight_layout()
    
    os.makedirs("results", exist_ok=True)
    plt.savefig(os.path.join("results", "calibration.png"))