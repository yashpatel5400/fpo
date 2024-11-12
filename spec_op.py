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


def train_model(pde, fs, us):
    net  = SpecOp(fs.shape[-1], us.shape[-1])
    loss = nn.MSELoss()
    opt  = torch.optim.Adam(net.parameters())
    net = net.to("cuda")

    batch_size  = 25
    num_batches = fs.shape[0] // batch_size
    epochs = 100

    losses = []
    for epoch in range(epochs):
        for batch in range(num_batches):
            f_batch = fs[batch * batch_size:(batch+1) * batch_size]
            u_batch = us[batch * batch_size:(batch+1) * batch_size]

            u_hat = net(f_batch)
            loss_val = loss(u_hat, u_batch)
            
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