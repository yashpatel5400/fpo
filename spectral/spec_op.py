import numpy as np
import pickle

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
with open("poisson.pkl", "rb") as f:
    (fs, us) = pickle.load(f)

net  = SpecOp(fs.shape[-1], us.shape[-1])
loss = nn.MSELoss()
opt  = torch.optim.Adam(net.parameters())

fs = torch.from_numpy(fs).to(torch.float32).to("cuda")
us = torch.from_numpy(us).to(torch.float32).to("cuda")
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

torch.save(net.state_dict(), "poisson.pt")