import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import os
import argparse 

# --- SNN Model Definition ---
class SimpleSpectralOperatorCNN(nn.Module):
    def __init__(self, K_trunc, hidden_channels=64, num_hidden_layers=3):
        super().__init__()
        self.K_trunc = K_trunc
        layers = []
        layers.append(nn.Conv2d(2, hidden_channels, kernel_size=3, padding='same'))
        layers.append(nn.ReLU())
        for _ in range(num_hidden_layers):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding='same'))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(hidden_channels, 2, kernel_size=3, padding='same'))
        self.model = nn.Sequential(*layers)
    def forward(self, x_spec_ch): return self.model(x_spec_ch)

# --- Data Handling ---
def spectrum_complex_to_channels_torch(spectrum_mat_complex):
    if not isinstance(spectrum_mat_complex, torch.Tensor):
        spectrum_mat_complex = torch.from_numpy(spectrum_mat_complex)
    if not torch.is_complex(spectrum_mat_complex): 
        if spectrum_mat_complex.ndim == 3 and spectrum_mat_complex.shape[0] == 2:
            return spectrum_mat_complex.float()
        raise ValueError(f"Input spectrum_mat_complex has shape {spectrum_mat_complex.shape} and is real. Expected complex [K,K] or real [2,K,K].")
    return torch.stack([spectrum_mat_complex.real, spectrum_mat_complex.imag], dim=0).float()

def channels_to_spectrum_complex_torch(channels_mat_real_imag):
    if channels_mat_real_imag.ndim != 3 or channels_mat_real_imag.shape[0] != 2: 
        raise ValueError(f"Input must have 2 channels as the first dimension, got shape {channels_mat_real_imag.shape}")
    return torch.complex(channels_mat_real_imag[0], channels_mat_real_imag[1])

class SpectralDataset(Dataset):
    def __init__(self, gamma_b_spectra, gamma_a_spectra):
        if gamma_b_spectra.shape != gamma_a_spectra.shape:
            raise ValueError("Input and target spectra arrays must have the same shape.")
        self.gamma_b_spectra = gamma_b_spectra
        self.gamma_a_spectra = gamma_a_spectra
    def __len__(self): return len(self.gamma_b_spectra)
    def __getitem__(self, idx):
        gamma_b_complex = self.gamma_b_spectra[idx]
        gamma_a_complex = self.gamma_a_spectra[idx]
        gamma_b_channels = spectrum_complex_to_channels_torch(gamma_b_complex)
        gamma_a_channels = spectrum_complex_to_channels_torch(gamma_a_complex)
        return gamma_b_channels, gamma_a_channels

def load_and_prepare_dataloaders(dataset_path, K_trunc_expected, batch_size, val_split=0.2, random_seed=42):
    try:
        data = np.load(dataset_path)
        gamma_b_all = data['gamma_b_Nmax'] 
        gamma_a_all = data['gamma_a_Nmax_true'] 
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}")
        return None, None, False 
    except KeyError as e:
        print(f"Error: Dataset file {dataset_path} missing expected key: {e}")
        return None, None, False
    if gamma_b_all.ndim < 3 or gamma_b_all.shape[1] != K_trunc_expected or gamma_b_all.shape[2] != K_trunc_expected:
        raise ValueError(f"K_trunc_expected ({K_trunc_expected}) does not match K_trunc in loaded data ({gamma_b_all.shape[1:] if gamma_b_all.ndim >=3 else 'Invalid Shape'}).")
    dataset = SpectralDataset(gamma_b_all, gamma_a_all)
    num_samples = len(dataset)
    val_size = int(val_split * num_samples)
    train_size = num_samples - val_size
    if train_size <= 0 :
        print(f"Error: Not enough samples for training. Train size: {train_size}. Need more data or smaller val_split.")
        return None, None, False
    if val_size <= 0:
        train_dataset = dataset; val_dataset = None; has_val_data = False
    else:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(random_seed))
        has_val_data = True
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0) if has_val_data else None
    print(f"Dataset loaded: {num_samples} total samples. Training: {len(train_dataset)}, Validation: {len(val_dataset) if val_dataset else 0}.")
    return train_loader, val_loader, has_val_data

# --- Training Loop ---
def train_snn_model(model, train_loader, val_loader, has_val_data, num_epochs, learning_rate, device, model_save_path):
    model.to(device); optimizer = optim.Adam(model.parameters(), lr=learning_rate); criterion = nn.MSELoss() 
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    print(f"\nStarting SNN training for {num_epochs} epochs on {device}...")
    for epoch in range(num_epochs):
        model.train(); epoch_train_loss = 0.0
        for gamma_b_ch, gamma_a_ch_true in train_loader:
            gamma_b_ch = gamma_b_ch.to(device); gamma_a_ch_true = gamma_a_ch_true.to(device)
            optimizer.zero_grad(); gamma_a_ch_pred = model(gamma_b_ch)
            loss = criterion(gamma_a_ch_pred, gamma_a_ch_true)
            loss.backward(); optimizer.step()
            epoch_train_loss += loss.item() * gamma_b_ch.size(0)
        avg_epoch_train_loss = epoch_train_loss / len(train_loader.dataset); train_losses.append(avg_epoch_train_loss)
        
        current_epoch_val_loss = float('inf')
        if has_val_data and val_loader is not None and len(val_loader.dataset) > 0:
            model.eval(); epoch_val_loss_sum = 0.0
            with torch.no_grad():
                for gamma_b_ch_val, gamma_a_ch_true_val in val_loader:
                    gamma_b_ch_val = gamma_b_ch_val.to(device); gamma_a_ch_true_val = gamma_a_ch_true_val.to(device)
                    gamma_a_ch_pred_val = model(gamma_b_ch_val) 
                    val_loss_item = criterion(gamma_a_ch_pred_val, gamma_a_ch_true_val)
                    epoch_val_loss_sum += val_loss_item.item() * gamma_b_ch_val.size(0)
            current_epoch_val_loss = epoch_val_loss_sum / len(val_loader.dataset)
            val_losses.append(current_epoch_val_loss)
            print_val_loss_str = f"{current_epoch_val_loss:.4e}"
        else: 
            val_losses.append(float('inf')); print_val_loss_str = "N/A" # Or append last train loss if no val
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_epoch_train_loss:.4e}, Val Loss: {print_val_loss_str}")
        if has_val_data and current_epoch_val_loss < best_val_loss:
            best_val_loss = current_epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  Model saved to {model_save_path} (Val Loss: {best_val_loss:.4e})")
    if not has_val_data and num_epochs > 0 : # If no validation, save model at the end
        torch.save(model.state_dict(), model_save_path)
        print(f"  Model saved to {model_save_path} (End of training, no validation set)")
    print("Training complete.")
    return train_losses, val_losses, best_val_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Spectral Neural Operator.")
    # Dataset/SNN structure parameters
    parser.add_argument('--pde_type', type=str, default="phenomenological_channel", 
                        choices=["phenomenological_channel", "poisson"], help="Type of data generation process the dataset corresponds to.")
    parser.add_argument('--k_trunc_snn', type=int, default=32, help='K_trunc for SNN (must match dataset Nmax).')
    parser.add_argument('--k_trunc_full', type=int, default=32, help='K_trunc_full used for dataset generation (for constructing dataset filename).')
    parser.add_argument('--dataset_dir', type=str, default="datasets", help="Directory of the .npz dataset.")
    parser.add_argument('--snn_hidden_channels', type=int, default=64)
    parser.add_argument('--snn_num_hidden_layers', type=int, default=3)
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--val_split', type=float, default=0.2)
    # Output directories
    parser.add_argument('--model_save_dir', type=str, default="trained_snn_models")
    parser.add_argument('--plot_save_dir', type=str, default="results_snn_training")

    # Noise/Source Parameters (for constructing correct filenames)
    # Phenomenological Channel
    parser.add_argument('--apply_attenuation', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--attenuation_loss_factor', type=float, default=0.2)
    parser.add_argument('--apply_additive_sobolev_noise', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--sobolev_noise_level_base', type=float, default=0.01)
    parser.add_argument('--sobolev_order_s', type=float, default=1.0)
    parser.add_argument('--apply_phase_noise', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--phase_noise_std_rad', type=float, default=0.05)
    # Poisson Source (GRF parameters)
    parser.add_argument('--grf_alpha', type=float, default=2.5)
    parser.add_argument('--grf_tau', type=float, default=7.0)
    parser.add_argument('--grf_offset_sigma', type=float, default=0.5, help="Sigma for hierarchical offset in Poisson source.")
    # Note: grf_offset_sigma was in data_gen but not part of its filename_suffix_str. 
    # If it influences the data significantly AND filename, it should be added.

    args = parser.parse_args()
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.plot_save_dir, exist_ok=True)

    # Construct filename_suffix_str based on pde_type and relevant params
    filename_suffix = ""
    if args.pde_type == "phenomenological_channel":
        noise_parts = []
        if args.apply_attenuation: noise_parts.append(f"att{args.attenuation_loss_factor:.2f}")
        if args.apply_additive_sobolev_noise: noise_parts.append(f"sob{args.sobolev_noise_level_base:.3f}s{args.sobolev_order_s:.1f}")
        if args.apply_phase_noise: noise_parts.append(f"ph{args.phase_noise_std_rad:.2f}")
        filename_suffix = "_".join(noise_parts) if noise_parts else "no_noise"
    elif args.pde_type == "poisson":
        filename_suffix = f"poisson_grfA{args.grf_alpha:.1f}T{args.grf_tau:.1f}"
    
    DATASET_FILENAME = f"dataset_{args.pde_type}_Nmax{args.k_trunc_snn}_Nfull{args.k_trunc_full}_{filename_suffix}.npz"
    DATASET_FILE_PATH = os.path.join(args.dataset_dir, DATASET_FILENAME)
    
    MODEL_FILENAME = f"snn_PDE{args.pde_type}_K{args.k_trunc_snn}_H{args.snn_hidden_channels}_L{args.snn_num_hidden_layers}_{filename_suffix}.pth"
    PLOT_FILENAME = f"snn_training_PDE{args.pde_type}_K{args.k_trunc_snn}_{filename_suffix}.png"
    
    full_model_save_path = os.path.join(args.model_save_dir, MODEL_FILENAME)
    full_plot_save_path = os.path.join(args.plot_save_dir, PLOT_FILENAME)
    
    print("--- Spectral Neural Operator Training ---")
    print(f"Using device: {DEVICE}")
    print(f"SNN K_trunc: {args.k_trunc_snn}")
    print(f"PDE Type for dataset: {args.pde_type}")
    print(f"Attempting to load dataset: {DATASET_FILE_PATH}")
    print(f"Model will be saved to: {full_model_save_path}")
    print(f"Plot will be saved to: {full_plot_save_path}")

    train_loader, val_loader, has_val_data_flag = load_and_prepare_dataloaders(
        DATASET_FILE_PATH, args.k_trunc_snn, args.batch_size, val_split=args.val_split
    )

    if train_loader is None: print("Failed to load data. Exiting."); exit()

    snn_model = SimpleSpectralOperatorCNN(args.k_trunc_snn, args.snn_hidden_channels, args.snn_num_hidden_layers)
    print(f"\nSNN Model Architecture:\n{snn_model}")
    num_params = sum(p.numel() for p in snn_model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")

    train_loss_hist, val_loss_hist, best_val_loss = train_snn_model(
        snn_model, train_loader, val_loader, has_val_data_flag, args.epochs, 
        args.lr, DEVICE, full_model_save_path
    )
    if has_val_data_flag: print(f"Best validation MSE loss achieved: {best_val_loss:.4e}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.epochs + 1), train_loss_hist, label='Training Loss')
    if has_val_data_flag and val_loss_hist: plt.plot(range(1, args.epochs + 1), val_loss_hist, label='Validation Loss')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss (2-channel spectra)'); plt.yscale('log')
    plt.title(f'SNN Training (K_trunc={args.k_trunc_snn}, PDE: {args.pde_type}, Params: {filename_suffix})')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(full_plot_save_path); print(f"\nTraining loss plot saved to {full_plot_save_path}"); plt.show()
    print("\nSNN training script finished.")
