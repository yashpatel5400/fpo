import argparse
import numpy as np
import pickle

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import utils

device = "cuda:0"

class FourierPoissonDataset(Dataset):
    def __init__(self, f_tensor, u_tensor):
        # f_tensor, u_tensor shape: (N, 2, 256,129)
        # store them
        self.f_tensor = f_tensor
        self.u_tensor = u_tensor

    def __len__(self):
        return self.f_tensor.shape[0]

    def __getitem__(self, idx):
        # returns a single sample
        f_sample = self.f_tensor[idx]  # shape (2,256,129)
        u_sample = self.u_tensor[idx]  # shape (2,256,129)
        return f_sample, u_sample


class EncoderDecoderNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Suppose input shape is (2, 256,129). We'll do a small architecture.

        # 1) Encoder
        self.enc_conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)

        # 2) Decoder
        self.dec_conv1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.dec_conv2 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1)

    def forward(self, x):
        # x shape: (batch, 2, 256, 129)

        # --- Encoder ---
        x = F.relu(self.enc_conv1(x))    # shape (batch,16,256,129)
        x = F.relu(self.enc_conv2(x))    # shape (batch,32,256,129)
        x = self.pool(x)                 # shape (batch,32,128,64)   # floor(129/2)=64

        # --- "Bottleneck" or pass-through might happen here ---

        # --- Decoder ---
        # We'll upsample back to (batch,32,256,128) then do conv:
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # shape (batch,32,256,128)
        # Note that we have 1 less column than 129 => if we want EXACT 129, we can do e.g. size=(256,129).
        # Let's keep it consistent. We'll do size explicitly:
        x = F.interpolate(x, size=(256,129), mode='nearest')  # shape (batch,32,256,129)
        x = F.relu(self.dec_conv1(x))    # shape (batch,16,256,129)
        x = self.dec_conv2(x)            # shape (batch, 2,256,129)

        return x


def construct_dataset(fs_np, us_np, train=True):
    train_prop = 0.5
    N_train = int(len(fs_np) * train_prop)

    if train:   fs_np, us_np = fs_np[:N_train], us_np[:N_train]
    else:       fs_np, us_np = fs_np[N_train:], us_np[N_train:]

    f_stack = np.stack([fs_np.real, fs_np.imag], axis=1)  # shape (300,2,256,129)
    u_stack = np.stack([us_np.real, us_np.imag], axis=1)  # shape (300,2,256,129)

    f_torch = torch.from_numpy(f_stack.astype(np.float32))  # shape (300,2,256,129)
    u_torch = torch.from_numpy(u_stack.astype(np.float32))  # shape (300,2,256,129)

    # build the dataset
    return FourierPoissonDataset(f_torch, u_torch)


def construct_model(data_loader):
    model = EncoderDecoderNet().to(device)  # move to GPU if device="cuda:0"
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    num_epochs = 400

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for f_batch, u_batch in data_loader:
            # f_batch, u_batch shape: (batch_size, 2, 256,129)
            # move to device
            f_batch = f_batch.to(device)
            u_batch = u_batch.to(device)

            # forward
            u_pred = model(f_batch)  # shape (batch_size, 2,256,129)

            # MSE loss
            loss = criterion(u_pred, u_batch)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * f_batch.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.6f}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde")
    parser.add_argument("--cutoff", type=int, default=8)
    args = parser.parse_args()

    with open(utils.DATA_FN(args.pde), "rb") as f:
        (fs_np, us_np) = pickle.load(f)

    dataset = construct_dataset(fs_np, us_np)
    model = construct_model(DataLoader(dataset, batch_size=16, shuffle=True))
    torch.save(model.state_dict(), utils.MODEL_FN(args.pde))