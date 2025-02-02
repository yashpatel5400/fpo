import argparse
import numpy as np
import pickle

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import utils

class PDEDataset(Dataset):
    """
    Example dataset that returns pairs of (input_image, target_image).
    Here, we just create random tensors for demonstration.
    Replace this with your real dataset and loading logic.
    """
    def __init__(self, fs, us):
        super().__init__()

        self.length = len(fs)
        self.X = fs.reshape(-1,1,256,256)
        self.y = us.reshape(-1,1,256,256)

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ResBlock(nn.Module):
    """
    A basic residual block: Conv -> BN -> ReLU -> Conv -> BN -> skip connection
    """
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Residual/skip connection
        out += identity
        out = self.relu(out)
        return out


class SpecOp(nn.Module):
    """
    More complex encoder-decoder architecture for Navier Stokes operator. Necessary to achieve decent results in
    Navier Stokes mapping
    - Input:  1×256×256
    - Output: 1×256×256
    """
    def __init__(self, cutoff=8):
        super(SpecOp, self).__init__()
    
        self.cutoff = cutoff

        # ----------------
        # Encoder
        # ----------------
        # (1) Initial conv: 1->8
        self.enc_conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(8)
        self.enc_relu1 = nn.ReLU(inplace=True)

        # (2) Residual block (8 channels)
        self.resblock1 = ResBlock(channels=8)

        # (3) Downsample cutoff->cutoff/2 by stride=2 (8->16 channels)
        self.enc_conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(16)
        self.enc_relu2 = nn.ReLU(inplace=True)

        # (4) Residual block (16 channels)
        self.resblock2 = ResBlock(channels=16)

        # ----------------
        # Decoder
        # ----------------
        # (1) Upsample cutoff/2->cutoff, 16->8 channels
        self.dec_convT1 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, stride=2)
        self.dec_bn1 = nn.BatchNorm2d(8)
        self.dec_relu1 = nn.ReLU(inplace=True)

        # (2) Residual block (8 channels)
        self.resblock3 = ResBlock(channels=8)

        # (3) Final conv to get single-channel output
        self.dec_conv_out = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1)
        # Optionally, add nn.Sigmoid() if you want output in [0,1].

    def forward(self, x):
        # Expect x.shape == (B, 1, cutoff, cutoff).

        # ------------- ENCODER -------------
        x = self.enc_conv1(x)      # (B, 8, cutoff, cutoff)
        x = self.enc_bn1(x)
        x = self.enc_relu1(x)

        x = self.resblock1(x)      # (B, 8, cutoff, cutoff)

        x = self.enc_conv2(x)      # (B, 16, cutoff/2, cutoff/2)
        x = self.enc_bn2(x)
        x = self.enc_relu2(x)

        x = self.resblock2(x)      # (B, 16, cutoff/2, cutoff/2)

        # ------------- DECODER -------------
        x = self.dec_convT1(x)     # (B, 8, cutoff, cutoff)
        x = self.dec_bn1(x)
        x = self.dec_relu1(x)

        x = self.resblock3(x)      # (B, 8, cutoff, cutoff)

        x = self.dec_conv_out(x)   # (B, 1, cutoff, cutoff)
        return x


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
                      

def train_model(pde, dataset, cutoff, num_epochs=600, batch_size=25, lr=1e-3, device='cuda'):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if   pde == "poisson":        model = SpecOp(cutoff=cutoff).to(device)
    elif pde == "navier_stokes":  model = SpecOp(cutoff=cutoff).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs[...,:cutoff,:cutoff]
            targets = targets[...,:cutoff,:cutoff]

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)

        # Compute average loss over the epoch
        epoch_loss = total_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")

    torch.save(model.state_dict(), utils.MODEL_FN(pde))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde")
    parser.add_argument("--cutoff", type=int, default=8)
    args = parser.parse_args()

    fs, us = utils.get_data(args.pde, train=True)
    dataset = PDEDataset(fs, us)
    train_model(args.pde, dataset, args.cutoff)