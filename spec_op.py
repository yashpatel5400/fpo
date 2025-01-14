import argparse
import numpy as np
import pickle

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

class SpecOp(nn.Module):
    def __init__(self, k_in, k_out):
        super().__init__()

        hidden_features = 16
        self.linear1 = nn.Linear(k_in, hidden_features)
        self.linear2 = nn.Linear(hidden_features, hidden_features)
        self.linear3 = nn.Linear(hidden_features, k_out)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x
    

def get_data(pde):
    with open(utils.DATA_FN(pde), "rb") as f:
        (fs, us) = pickle.load(f)

    prop_train = 0.75
    N = fs.shape[0]
    N_train = int(N * prop_train)

    fs = torch.from_numpy(fs[:N_train]).to(torch.float32).to("cuda")
    us = torch.from_numpy(us[:N_train]).to(torch.float32).to("cuda")
    return fs, us


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


def sobolev_loss(uhat, u, weight, loss_type="sobolev"):
    if loss_type == "l2":
        return torch.mean((uhat - u) ** 2)
    return torch.mean((weight * (uhat - u) ** 2))
                      

def train_model(pde, fs, us):
    net  = SpecOp(fs.shape[-1], us.shape[-1])
    loss = nn.MSELoss()
    opt  = torch.optim.Adam(net.parameters())
    net = net.to("cuda")

    # sobolev_weight = get_sobolev_weights(s=2, gamma=1.9, shape=(fs.shape[1],fs.shape[2]))
    # sobolev_weight = torch.from_numpy(sobolev_weight).to("cuda")
    sobolev_weight = None

    batch_size = 25
    num_batches = fs.shape[0] // batch_size
    epochs = 500

    losses = []
    for epoch in range(epochs):
        for batch in range(num_batches):
            f_batch = fs[batch * batch_size:(batch+1) * batch_size]
            u_batch = us[batch * batch_size:(batch+1) * batch_size]

            u_hat = net(f_batch)
            loss_val = sobolev_loss(u_hat, u_batch, sobolev_weight, loss_type="l2")
            
            opt.zero_grad()
            loss_val.backward()
            opt.step()

        losses.append(loss_val.cpu().detach().numpy())
        if epoch % 10 == 0:
            print(f"Epoch {epoch} / {epochs} -- {loss_val}")

    torch.save(net.state_dict(), utils.MODEL_FN(pde))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde")
    args = parser.parse_args()

    fs, us = get_data(args.pde)
    train_model(args.pde, fs, us)