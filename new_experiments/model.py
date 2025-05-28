import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import os
import argparse # Added for command-line arguments

# --- SNN Model Definition ---
class SimpleSpectralOperatorCNN(nn.Module):
    def __init__(self, K_trunc, hidden_channels=64, num_hidden_layers=3):
        super().__init__()
        self.K_trunc = K_trunc
        
        layers = []
        # Input: 2 channels (real/imag). Output: hidden_channels
        layers.append(nn.Conv2d(2, hidden_channels, kernel_size=3, padding='same'))
        layers.append(nn.ReLU())

        for _ in range(num_hidden_layers):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding='same'))
            layers.append(nn.ReLU())
        
        # Output: 2 channels (real/imag prediction)
        layers.append(nn.Conv2d(hidden_channels, 2, kernel_size=3, padding='same'))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x_spec_ch): # x_spec_ch: (batch, 2, K_trunc, K_trunc)
        return self.model(x_spec_ch)

# --- Data Handling ---
def spectrum_complex_to_channels_torch(spectrum_mat_complex):
    """Converts a complex KxK spectrum tensor/array to a 2xKxK real/imaginary channel tensor."""
    if not isinstance(spectrum_mat_complex, torch.Tensor):
        spectrum_mat_complex = torch.from_numpy(spectrum_mat_complex)
    if not torch.is_complex(spectrum_mat_complex): 
        if spectrum_mat_complex.ndim == 3 and spectrum_mat_complex.shape[0] == 2:
            return spectrum_mat_complex.float()
        raise ValueError(f"Input spectrum_mat_complex has shape {spectrum_mat_complex.shape} and is real. Expected complex [K,K] or real [2,K,K].")
    return torch.stack([spectrum_mat_complex.real, spectrum_mat_complex.imag], dim=0).float()

def channels_to_spectrum_complex_torch(channels_mat_real_imag):
    """Converts a 2xKxK real/imaginary channel tensor back to a complex KxK spectrum tensor."""
    if channels_mat_real_imag.ndim != 3 or channels_mat_real_imag.shape[0] != 2: 
        raise ValueError(f"Input must have 2 channels (real and imaginary parts) as the first dimension, got shape {channels_mat_real_imag.shape}")
    return torch.complex(channels_mat_real_imag[0], channels_mat_real_imag[1])


class SpectralDataset(Dataset):
    def __init__(self, gamma_b_spectra, gamma_a_spectra):
        """
        Args:
            gamma_b_spectra (np.array): Array of input spectra, shape (num_samples, K, K), complex.
            gamma_a_spectra (np.array): Array of target spectra, shape (num_samples, K, K), complex.
        """
        if gamma_b_spectra.shape != gamma_a_spectra.shape:
            raise ValueError("Input and target spectra arrays must have the same shape.")
        
        self.gamma_b_spectra = gamma_b_spectra
        self.gamma_a_spectra = gamma_a_spectra

    def __len__(self):
        return len(self.gamma_b_spectra)

    def __getitem__(self, idx):
        gamma_b_complex = self.gamma_b_spectra[idx]
        gamma_a_complex = self.gamma_a_spectra[idx]
        gamma_b_channels = spectrum_complex_to_channels_torch(gamma_b_complex)
        gamma_a_channels = spectrum_complex_to_channels_torch(gamma_a_complex)
        return gamma_b_channels, gamma_a_channels

def load_and_prepare_dataloaders(dataset_path, K_trunc_expected, batch_size, val_split=0.2, random_seed=42):
    """Loads data, creates datasets, and prepares dataloaders."""
    try:
        data = np.load(dataset_path)
        # Use the keys as saved by the multi-resolution data generator
        gamma_b_all = data['gamma_b_Nmax'] # SNN input
        gamma_a_all = data['gamma_a_Nmax_true'] # SNN target
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
        print(f"Warning: Not enough samples for a validation set (val_size={val_size}). Proceeding without validation.")
        train_dataset = dataset; val_dataset = None; has_val_data = False
    else:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], 
                                                  generator=torch.Generator().manual_seed(random_seed))
        has_val_data = True
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0) if has_val_data else None
    
    print(f"Dataset loaded: {num_samples} total samples.")
    print(f"Training set: {len(train_dataset)} samples.")
    if has_val_data: print(f"Validation set: {len(val_dataset)} samples.")
    else: print("No validation set created.")
    return train_loader, val_loader, has_val_data

# --- Training Loop ---
def train_snn_model(model, train_loader, val_loader, has_val_data, num_epochs, learning_rate, device, model_save_path):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss() 
    
    train_losses = []
    val_losses = []
    val_gram_matrix_errors_epoch_avg = [] 
    best_val_loss = float('inf')
    
    print(f"\nStarting SNN training for {num_epochs} epochs on {device}...")
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for gamma_b_ch, gamma_a_ch_true in train_loader:
            gamma_b_ch = gamma_b_ch.to(device); gamma_a_ch_true = gamma_a_ch_true.to(device)
            optimizer.zero_grad()
            gamma_a_ch_pred = model(gamma_b_ch)
            loss = criterion(gamma_a_ch_pred, gamma_a_ch_true)
            loss.backward(); optimizer.step()
            epoch_train_loss += loss.item() * gamma_b_ch.size(0)
        avg_epoch_train_loss = epoch_train_loss / len(train_loader.dataset)
        train_losses.append(avg_epoch_train_loss)
        
        current_epoch_val_loss = float('inf'); current_epoch_avg_gram_error = float('nan')
        if has_val_data and val_loader is not None and len(val_loader.dataset) > 0:
            model.eval()
            epoch_val_loss_sum = 0.0; epoch_gram_matrix_error_sum_for_avg = 0.0; num_valid_gram_batches = 0
            with torch.no_grad():
                for gamma_b_ch_val, gamma_a_ch_true_val in val_loader:
                    gamma_b_ch_val = gamma_b_ch_val.to(device); gamma_a_ch_true_val = gamma_a_ch_true_val.to(device)
                    gamma_a_ch_pred_val = model(gamma_b_ch_val) 
                    val_loss_item = criterion(gamma_a_ch_pred_val, gamma_a_ch_true_val)
                    epoch_val_loss_sum += val_loss_item.item() * gamma_b_ch_val.size(0)
                    # Gram matrix error calculation (optional, can be removed if not needed for SNN training eval)
                    batch_size_current = gamma_b_ch_val.size(0)
                    if batch_size_current >= 2: 
                        gamma_a_complex_true_batch_list = [channels_to_spectrum_complex_torch(gamma_a_ch_true_val[i]) for i in range(batch_size_current)]
                        gamma_a_complex_pred_batch_list = [channels_to_spectrum_complex_torch(gamma_a_ch_pred_val[i]) for i in range(batch_size_current)]
                        gamma_a_complex_true_batch = torch.stack(gamma_a_complex_true_batch_list).cpu() 
                        gamma_a_complex_pred_batch = torch.stack(gamma_a_complex_pred_batch_list).cpu()
                        G_true_batch = torch.zeros((batch_size_current, batch_size_current), dtype=torch.complex128)
                        G_est_batch = torch.zeros((batch_size_current, batch_size_current), dtype=torch.complex128)
                        for r_idx in range(batch_size_current):
                            for c_idx in range(batch_size_current):
                                G_true_batch[r_idx, c_idx] = torch.vdot(gamma_a_complex_true_batch[r_idx].flatten(), gamma_a_complex_true_batch[c_idx].flatten())
                                G_est_batch[r_idx, c_idx] = torch.vdot(gamma_a_complex_pred_batch[r_idx].flatten(), gamma_a_complex_pred_batch[c_idx].flatten())
                        gram_diff_fro_norm = torch.linalg.norm(G_est_batch - G_true_batch, ord='fro')
                        epoch_gram_matrix_error_sum_for_avg += gram_diff_fro_norm.item()
                        num_valid_gram_batches += 1
            current_epoch_val_loss = epoch_val_loss_sum / len(val_loader.dataset)
            val_losses.append(current_epoch_val_loss)
            if num_valid_gram_batches > 0: current_epoch_avg_gram_error = epoch_gram_matrix_error_sum_for_avg / num_valid_gram_batches
            val_gram_matrix_errors_epoch_avg.append(current_epoch_avg_gram_error)
            print_val_loss_str = f"{current_epoch_val_loss:.4e}"
            print_gram_err_str = f"{current_epoch_avg_gram_error:.4e}" if not np.isnan(current_epoch_avg_gram_error) else "N/A"
        else: 
            val_losses.append(float('inf')); val_gram_matrix_errors_epoch_avg.append(float('nan'))
            print_val_loss_str = "N/A"; print_gram_err_str = "N/A"
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_epoch_train_loss:.4e}, Val Loss: {print_val_loss_str}, Val Gram Err (Fro): {print_gram_err_str}")
        if has_val_data and current_epoch_val_loss < best_val_loss:
            best_val_loss = current_epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  Model saved to {model_save_path} (Val Loss: {best_val_loss:.4e})")
    print("Training complete.")
    return train_losses, val_losses, val_gram_matrix_errors_epoch_avg, best_val_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Spectral Neural Operator on phenomenological noise data.")
    parser.add_argument('--k_trunc_snn', type=int, default=32, help='K_trunc for SNN (must match dataset).')
    parser.add_argument('--k_trunc_full', type=int, default=32, help='K_trunc_full used for dataset generation (for filename).')
    parser.add_argument('--dataset_dir', type=str, default="datasets", help="Directory of the .npz dataset.")
    
    parser.add_argument('--snn_hidden_channels', type=int, default=64)
    parser.add_argument('--snn_num_hidden_layers', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--val_split', type=float, default=0.2)
    
    parser.add_argument('--model_save_dir', type=str, default="trained_snn_models")
    parser.add_argument('--plot_save_dir', type=str, default="results_snn_training")

    # Arguments for noise parameters (for logging and filename generation)
    parser.add_argument('--apply_attenuation', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--attenuation_loss_factor', type=float, default=0.2)
    parser.add_argument('--apply_additive_sobolev_noise', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--sobolev_noise_level_base', type=float, default=0.01)
    parser.add_argument('--sobolev_order_s', type=float, default=1.0)
    parser.add_argument('--apply_phase_noise', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--phase_noise_std_rad', type=float, default=0.05)

    args = parser.parse_args()
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.plot_save_dir, exist_ok=True)

    # Construct a descriptive filename based on SNN arch and key noise params
    noise_desc = []
    if args.apply_attenuation: noise_desc.append(f"att{args.attenuation_loss_factor:.2f}")
    if args.apply_additive_sobolev_noise: noise_desc.append(f"sob{args.sobolev_noise_level_base:.3f}s{args.sobolev_order_s:.1f}")
    if args.apply_phase_noise: noise_desc.append(f"ph{args.phase_noise_std_rad:.2f}")
    noise_str = "_".join(noise_desc) if noise_desc else "no_noise"

    MODEL_FILENAME = f"snn_K{args.k_trunc_snn}_H{args.snn_hidden_channels}_L{args.snn_num_hidden_layers}_{noise_str}.pth"
    PLOT_FILENAME = f"snn_training_K{args.k_trunc_snn}_{noise_str}.png"
    
    full_model_save_path = os.path.join(args.model_save_dir, MODEL_FILENAME)
    full_plot_save_path = os.path.join(args.plot_save_dir, PLOT_FILENAME)
    
    # Dataset path still relies on K_trunc_snn and K_trunc_full from data generation script
    dataset_filename = f"phenomenological_channel_dataset_Nmax{args.k_trunc_snn}_Nfull{args.k_trunc_full}.npz"
    DATASET_FILE_PATH = os.path.join(args.dataset_dir, dataset_filename)
    
    print("--- Spectral Neural Operator Training ---")
    print(f"Using device: {DEVICE}")
    print(f"SNN K_trunc: {args.k_trunc_snn}")
    print(f"Loading dataset from: {DATASET_FILE_PATH}")
    print(f"Model will be saved to: {full_model_save_path}")
    print("Assumed noise config for the dataset being loaded (for naming output):")
    print(f"  Attenuation: {args.apply_attenuation}, Factor: {args.attenuation_loss_factor}")
    print(f"  Sobolev Noise: {args.apply_additive_sobolev_noise}, Base: {args.sobolev_noise_level_base}, Order s: {args.sobolev_order_s}")
    print(f"  Phase Noise: {args.apply_phase_noise}, StdDev: {args.phase_noise_std_rad}")


    train_loader, val_loader, has_val_data_flag = load_and_prepare_dataloaders(
        DATASET_FILE_PATH, args.k_trunc_snn, args.batch_size, val_split=args.val_split
    )

    if train_loader is None: exit()

    snn_model = SimpleSpectralOperatorCNN(args.k_trunc_snn, args.snn_hidden_channels, args.snn_num_hidden_layers)
    print(f"\nSNN Model Architecture:\n{snn_model}")
    num_params = sum(p.numel() for p in snn_model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")

    train_loss_hist, val_loss_hist, val_gram_err_hist, best_val_loss = train_snn_model(
        snn_model, train_loader, val_loader, has_val_data_flag, args.epochs, 
        args.lr, DEVICE, full_model_save_path
    )
    if has_val_data_flag: print(f"Best validation MSE loss achieved: {best_val_loss:.4e}")

    fig, ax1 = plt.subplots(figsize=(12, 7))
    color = 'tab:red'; ax1.set_xlabel('Epoch'); ax1.set_ylabel('MSE Loss (2-channel spectra)', color=color)
    ax1.plot(range(1, args.epochs + 1), train_loss_hist, color=color, linestyle='-', label='Training MSE Loss')
    if has_val_data_flag and val_loss_hist: ax1.plot(range(1, args.epochs + 1), val_loss_hist, color=color, linestyle='--', label='Validation MSE Loss')
    ax1.tick_params(axis='y', labelcolor=color); ax1.set_yscale('log'); ax1.grid(True, axis='y', linestyle=':', alpha=0.7, which='major')
    lines, labels = ax1.get_legend_handles_labels()
    if has_val_data_flag and val_gram_err_hist: 
        ax2 = ax1.twinx(); color = 'tab:blue'
        ax2.set_ylabel('Avg Gram Matrix Error (Frobenius Norm)', color=color)  
        ax2.plot(range(1, args.epochs + 1), val_gram_err_hist, color=color, linestyle='-.', label='Validation Gram Matrix Error (Fro)')
        ax2.tick_params(axis='y', labelcolor=color)
        valid_gram_errors = [e for e in val_gram_err_hist if e is not None and not np.isnan(e) and e > 0]
        if valid_gram_errors and np.max(valid_gram_errors) / np.min(valid_gram_errors) > 100: ax2.set_yscale('log')
        ax2.grid(True, axis='y', linestyle=':', alpha=0.7, which='major') 
        lines2, labels2 = ax2.get_legend_handles_labels(); ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    else: ax1.legend(loc='upper right') 
    fig.suptitle(f'SNN Training (K_trunc={args.k_trunc_snn}, Noise: {noise_str})', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  
    plt.savefig(full_plot_save_path); print(f"\nTraining metrics plot saved to {full_plot_save_path}"); plt.show()
