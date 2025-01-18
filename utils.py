import os
import pickle
import torch

PDE_DIR     = lambda pde : os.path.join("data", pde)
RESULTS_DIR = lambda pde : os.path.join("results", pde)
DATA_FN     = lambda pde : os.path.join(PDE_DIR(pde), "data_old.pkl")
MODEL_FN    = lambda pde : os.path.join(PDE_DIR(pde), "model.pt")


def get_data(pde, train):
    with open(DATA_FN(pde), "rb") as f:
        (fs, us) = pickle.load(f)

    prop_train = 0.75
    N = fs.shape[0]
    N_train = int(N * prop_train)

    if train:
        fs = torch.from_numpy(fs[:N_train]).to(torch.float32).to("cuda")
        us = torch.from_numpy(us[:N_train]).to(torch.float32).to("cuda")
    else:
        fs = torch.from_numpy(fs[N_train:]).to(torch.float32).to("cuda")
        us = torch.from_numpy(us[N_train:]).to(torch.float32).to("cuda")
    return fs, us