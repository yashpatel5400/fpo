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

class SpecOp(nn.Module):
    """
    A simple CNN-based encoder-decoder.
    - Input:  1×256×256
    - Output: 1×256×256
    """
    def __init__(self):
        super(SpecOp, self).__init__()
        
        # Encoder: downsample
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),    # [B, 8, 256, 256]
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),   # [B, 16, 256, 256]
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),                           # Slight dropout
            nn.MaxPool2d(2),                              # [B, 16, 128, 128]
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # [B, 32, 128, 128]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.MaxPool2d(2),                              # [B, 32, 64, 64]
        )

        # Decoder: upsample
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # [B, 16, 128, 128]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),   # [B, 8, 256, 256]
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),            # [B, 1, 256, 256]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
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
                      

def train_model(pde, dataset, num_epochs=100, batch_size=4, lr=1e-3, device='cuda'):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SpecOp().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

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
    args = parser.parse_args()

    fs, us = utils.get_data(args.pde, train=True)
    dataset = PDEDataset(fs, us)
    train_model(args.pde, dataset)