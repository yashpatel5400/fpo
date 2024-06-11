import argparse
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

from fno_utils import FNO2d, FNODatasetSingle
    
def fpo(cfg):
    pde_name = cfg["filename"].split(".h")[0]
    times = []
    num_trials = 10
    trial_size = 50
    train_data = FNODatasetSingle(filename=os.path.join("experiments", cfg["filename"]))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=trial_size)
    model_weights = torch.load(os.path.join("experiments", f"{pde_name}_FNO.pt"))

    for _ in range(num_trials):
        fno = FNO2d(
            num_channels=cfg["num_channels"], 
            modes1=cfg["modes"], 
            modes2=cfg["modes"], 
            width=cfg["width"], 
            initial_step=cfg["initial_step"]).to("cuda")
        fno.load_state_dict(model_weights["model_state_dict"])

        device = "cuda"
        start_time = time.time()
        for xxbatch, yy, gridbatch in train_loader:
            if cfg["training_type"] == "autoregressive":
                inp_shape = list(xxbatch.shape)
                inp_shape = inp_shape[:-2]
                inp_shape.append(-1)
                
                xxbatch = xxbatch.reshape(inp_shape)
                for xidx in range(trial_size):
                    xx = xxbatch[xidx:xidx+1,...].to(device)
                    grid = gridbatch[xidx:xidx+1,...].to(device)
                    yhat = fno(xx, grid)
            else:
                for xidx in range(trial_size):
                    xx = xxbatch[xidx:xidx+1,...].to(device)
                    grid = gridbatch[xidx:xidx+1,...].to(device)
                    yhat = fno(xx[...,0,:], grid)
            break
        end_time = time.time()
        times.append((end_time - start_time) / trial_size)

    # debug_plot = False
    # if debug_plot:
    #     fig, (ax1, ax2) = plt.subplots(1, 2)
    #     ax1.imshow(y.cpu().detach().numpy()[0,:,:,0,0])
    #     ax2.imshow(yhat.cpu().detach().numpy()[0,:,:,0,0])
    #     plt.savefig("result.png")
    return np.array(times)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde", choices=["Darcy", "diff-react", "rdb"])
    args = parser.parse_args()

    cfg_fn = os.path.join("experiments", f"config_{args.pde}.yaml")
    with open(cfg_fn, "r") as f:
        cfg = yaml.safe_load(f)
    opt_dec = fpo(cfg)