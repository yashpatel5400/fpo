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
    def __init__(self, K_input_resolution, K_output_resolution, hidden_channels=64, num_hidden_layers=3):
        super().__init__()
        self.K_input_resolution = K_input_resolution
        self.K_output_resolution = K_output_resolution
        
        if K_output_resolution > K_input_resolution:
            raise ValueError("K_output_resolution cannot be greater than K_input_resolution for this SNN design (cropping output).")
        
        layers = []
        # CNN body operates at K_input_resolution
        layers.append(nn.Conv2d(2, hidden_channels, kernel_size=3, padding='same'))
        layers.append(nn.ReLU())
        
        for _ in range(num_hidden_layers):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding='same'))
            layers.append(nn.ReLU())
            
        layers.append(nn.Conv2d(hidden_channels, 2, kernel_size=3, padding='same'))
        self.cnn_body = nn.Sequential(*layers)

    def forward(self, x_spec_ch_full_input): # x_spec_ch_full_input: (batch, 2, K_input, K_input)
        # CNN body processes at full input resolution
        x_processed_full = self.cnn_body(x_spec_ch_full_input) # Output: (batch, 2, K_input, K_input)
        
        # Truncate/crop the output to K_output_resolution x K_output_resolution
        if self.K_input_resolution == self.K_output_resolution:
            return x_processed_full
        else: 
            # K_input_resolution > K_output_resolution
            start_idx = self.K_input_resolution // 2 - self.K_output_resolution // 2
            end_idx = start_idx + self.K_output_resolution
            return x_processed_full[:, :, start_idx:end_idx, start_idx:end_idx]

# --- Data Handling ---
def spectrum_complex_to_channels_torch(spectrum_mat_complex):
    if not isinstance(spectrum_mat_complex, torch.Tensor):
        spectrum_mat_complex = torch.from_numpy(spectrum_mat_complex)
    
    if not torch.is_complex(spectrum_mat_complex): 
        if spectrum_mat_complex.ndim == 3 and spectrum_mat_complex.shape[0] == 2: # Already [2,K,K] real tensor
            return spectrum_mat_complex.float() 
        if spectrum_mat_complex.ndim == 2 and not torch.is_complex(spectrum_mat_complex): # Real [K,K] tensor
             raise ValueError(f"Input spectrum_mat_complex is real [K,K] but expected complex or [2,K,K] real.")
        # Other unexpected real tensor shapes
        raise ValueError(f"Input spectrum_mat_complex has shape {spectrum_mat_complex.shape}. Expected complex [K,K] or real [2,K,K].")
        
    return torch.stack([spectrum_mat_complex.real, spectrum_mat_complex.imag], dim=0).float()

def channels_to_spectrum_complex_torch(channels_mat_real_imag):
    if channels_mat_real_imag.ndim != 3 or channels_mat_real_imag.shape[0] != 2: 
        raise ValueError(f"Input must have 2 channels as the first dimension, got shape {channels_mat_real_imag.shape}")
    return torch.complex(channels_mat_real_imag[0], channels_mat_real_imag[1])

class SNNFullInputDataset(Dataset):
    def __init__(self, gamma_b_full_spectra, gamma_a_truncated_target_spectra):
        if gamma_b_full_spectra.shape[0] != gamma_a_truncated_target_spectra.shape[0]:
            raise ValueError("Input and target arrays must have the same number of samples.")
        
        self.gamma_b_full_spectra = gamma_b_full_spectra
        self.gamma_a_truncated_target_spectra = gamma_a_truncated_target_spectra

    def __len__(self):
        return len(self.gamma_b_full_spectra)

    def __getitem__(self, idx):
        gamma_b_full_complex = self.gamma_b_full_spectra[idx] 
        gamma_a_target_complex = self.gamma_a_truncated_target_spectra[idx]

        gamma_b_channels = spectrum_complex_to_channels_torch(gamma_b_full_complex) # Full input
        gamma_a_channels = spectrum_complex_to_channels_torch(gamma_a_target_complex) # Truncated target
        
        return gamma_b_channels, gamma_a_channels

def load_and_prepare_dataloaders(dataset_path, 
                                 snn_input_res_expected, 
                                 snn_target_res_expected, 
                                 batch_size, val_split=0.2, random_seed=42):
    try:
        data = np.load(dataset_path)
        gamma_b_full_all = data['gamma_b_full_input'] 
        gamma_a_truncated_target_all = data['gamma_a_snn_target'] 
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}")
        return None, None, False 
    except KeyError as e:
        print(f"Error: Dataset file {dataset_path} missing expected key: {e}")
        return None, None, False

    if gamma_b_full_all.ndim < 3 or gamma_b_full_all.shape[1] != snn_input_res_expected:
        raise ValueError(f"Dataset's gamma_b_full_input res ({gamma_b_full_all.shape[1]}) != expected SNN input res ({snn_input_res_expected}).")
    if gamma_a_truncated_target_all.ndim < 3 or gamma_a_truncated_target_all.shape[1] != snn_target_res_expected:
        raise ValueError(f"Dataset's gamma_a_snn_target res ({gamma_a_truncated_target_all.shape[1]}) != expected SNN target res ({snn_target_res_expected}).")

    dataset = SNNFullInputDataset(gamma_b_full_all, gamma_a_truncated_target_all)
    num_samples = len(dataset)
    val_size = int(val_split * num_samples)
    train_size = num_samples - val_size
    
    if train_size <= 0 :
        print(f"Error: Not enough samples for training. Train size: {train_size}. Need more data or smaller val_split.")
        return None, None, False
    
    has_val_data = True
    if val_size <= 0:
        print(f"Warning: Not enough samples for a validation set (val_size={val_size}). Proceeding without validation.")
        train_dataset = dataset
        val_dataset = None
        has_val_data = False
    else:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], 
                                                  generator=torch.Generator().manual_seed(random_seed))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0) if has_val_data else None
    
    print(f"Dataset loaded: {num_samples} total samples. Training: {len(train_dataset)}, Validation: {len(val_dataset) if val_dataset else 0}.")
    return train_loader, val_loader, has_val_data

# --- Training Loop ---
def train_snn_model(model, 
                    train_loader, val_loader, has_val_data, 
                    num_epochs, learning_rate, device, model_save_path):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss() 
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\nStarting SNN training for {num_epochs} epochs on {device}...")
    print(f"SNN input res: {model.K_input_resolution}, SNN output res (for loss): {model.K_output_resolution}")

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for gamma_b_ch_full, gamma_a_ch_target_truncated in train_loader: 
            gamma_b_ch_full = gamma_b_ch_full.to(device)
            gamma_a_ch_target_truncated = gamma_a_ch_target_truncated.to(device) 
            
            optimizer.zero_grad()
            gamma_a_ch_pred_truncated = model(gamma_b_ch_full) 
            
            loss = criterion(gamma_a_ch_pred_truncated, gamma_a_ch_target_truncated)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * gamma_b_ch_full.size(0)
        
        avg_epoch_train_loss = epoch_train_loss / len(train_loader.dataset)
        train_losses.append(avg_epoch_train_loss)
        
        current_epoch_val_loss = float('inf')
        print_val_loss_str = "N/A"
        if has_val_data and val_loader is not None and len(val_loader.dataset) > 0:
            model.eval()
            epoch_val_loss_sum = 0.0
            with torch.no_grad():
                for gamma_b_ch_full_val, gamma_a_ch_target_truncated_val in val_loader:
                    gamma_b_ch_full_val = gamma_b_ch_full_val.to(device)
                    gamma_a_ch_target_truncated_val = gamma_a_ch_target_truncated_val.to(device)
                    gamma_a_ch_pred_truncated_val = model(gamma_b_ch_full_val) 
                    
                    val_loss_item = criterion(gamma_a_ch_pred_truncated_val, gamma_a_ch_target_truncated_val)
                    epoch_val_loss_sum += val_loss_item.item() * gamma_b_ch_full_val.size(0)
            current_epoch_val_loss = epoch_val_loss_sum / len(val_loader.dataset)
            val_losses.append(current_epoch_val_loss)
            print_val_loss_str = f"{current_epoch_val_loss:.4e}"
        else: 
            val_losses.append(float('inf')) 
            
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_epoch_train_loss:.4e}, Val Loss: {print_val_loss_str}")

        if has_val_data and current_epoch_val_loss < best_val_loss:
            best_val_loss = current_epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  Model saved to {model_save_path} (Val Loss: {best_val_loss:.4e})")
            
    if not has_val_data and num_epochs > 0 : 
        torch.save(model.state_dict(), model_save_path)
        print(f"  Model saved to {model_save_path} (End of training, no validation set used)")
        
    print("Training complete.")
    return train_losses, val_losses, best_val_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Spectral Neural Operator (Full Input -> Truncated Output for Loss).")
    
    # --- PDE Type and Dataset Parameters ---
    parser.add_argument('--pde_type', type=str, default="step_index_fiber", 
                        choices=["poisson", "step_index_fiber", "grin_fiber"],
                        help="Type of data generation process the dataset corresponds to.")
    parser.add_argument('--n_grid_sim_input_ds', type=int, default=64, 
                        help='Nin: Resolution of gamma_b_full_input in dataset. SNN will take this as input resolution.')
    parser.add_argument('--k_snn_target_res', type=int, default=32, 
                        help='Nout: Resolution of gamma_a_snn_target in dataset. SNN output will be truncated to this for loss.')
    parser.add_argument('--dataset_dir', type=str, default="datasets",
                        help="Directory of the .npz dataset.")

    # --- SNN Architecture Parameters ---
    parser.add_argument('--snn_hidden_channels', type=int, default=64)
    parser.add_argument('--snn_num_hidden_layers', type=int, default=3)
    
    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--val_split', type=float, default=0.2)
    
    # --- Output Directories ---
    parser.add_argument('--model_save_dir', type=str, default="trained_snn_models")
    parser.add_argument('--plot_save_dir', type=str, default="results_snn_training")

    # --- Parameters for Filename Construction (must match data_gen_script) ---
    parser.add_argument('--grf_alpha', type=float, default=4.0) 
    parser.add_argument('--grf_tau', type=float, default=1.0)   
    parser.add_argument('--grf_offset_sigma', type=float, default=0.5, 
                        help="Sigma for hierarchical offset in Poisson source (f term).")
    parser.add_argument('--L_domain', type=float, default=2*np.pi) 
    parser.add_argument('--fiber_core_radius_factor', type=float, default=0.2)
    parser.add_argument('--fiber_potential_depth', type=float, default=1.0)
    parser.add_argument('--grin_strength', type=float, default=0.01, help="Strength for GRIN fiber potential.")
    parser.add_argument('--evolution_time_T', type=float, default=0.1) 
    parser.add_argument('--solver_num_steps', type=int, default=50) 
    
    args = parser.parse_args()
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.plot_save_dir, exist_ok=True)

    filename_suffix = ""
    if args.pde_type == "poisson":
        filename_suffix = f"poisson_grfA{args.grf_alpha:.1f}T{args.grf_tau:.1f}OffS{args.grf_offset_sigma:.1f}"
    elif args.pde_type == "step_index_fiber":
        filename_suffix = (f"fiber_GRFinA{args.grf_alpha:.1f}T{args.grf_tau:.1f}_"
                           f"coreR{args.fiber_core_radius_factor:.1f}_V{args.fiber_potential_depth:.1f}_"
                           f"evoT{args.evolution_time_T:.1e}_steps{args.solver_num_steps}")
    elif args.pde_type == "grin_fiber":
        filename_suffix = (f"grinfiber_GRFinA{args.grf_alpha:.1f}T{args.grf_tau:.1f}_"
                           f"strength{args.grin_strength:.2e}_"
                           f"evoT{args.evolution_time_T:.1e}_steps{args.solver_num_steps}")
    
    DATASET_FILENAME = f"dataset_{args.pde_type}_Nin{args.n_grid_sim_input_ds}_Nout{args.k_snn_target_res}_{filename_suffix}.npz"
    DATASET_FILE_PATH = os.path.join(args.dataset_dir, DATASET_FILENAME)
    
    MODEL_FILENAME = f"snn_PDE{args.pde_type}_Kin{args.n_grid_sim_input_ds}_Kout{args.k_snn_target_res}_H{args.snn_hidden_channels}_L{args.snn_num_hidden_layers}_{filename_suffix}.pth"
    PLOT_FILENAME = f"snn_training_PDE{args.pde_type}_Kin{args.n_grid_sim_input_ds}_Kout{args.k_snn_target_res}_{filename_suffix}.png"
    
    full_model_save_path = os.path.join(args.model_save_dir, MODEL_FILENAME)
    full_plot_save_path = os.path.join(args.plot_save_dir, PLOT_FILENAME)
    
    print("--- Spectral Neural Operator Training (Full Input -> Truncated Output for Loss) ---")
    print(f"SNN Input Resolution (from n_grid_sim_input_ds): {args.n_grid_sim_input_ds}")
    print(f"SNN Target Output Resolution (k_snn_target_res): {args.k_snn_target_res}")
    print(f"Attempting to load dataset: {DATASET_FILE_PATH}")

    train_loader, val_loader, has_val_data_flag = load_and_prepare_dataloaders(
        DATASET_FILE_PATH, 
        args.n_grid_sim_input_ds, 
        args.k_snn_target_res,    
        args.batch_size, 
        val_split=args.val_split
    )

    if train_loader is None:
        print("Failed to load data. Exiting.")
        exit()

    snn_model = SimpleSpectralOperatorCNN(
        K_input_resolution=args.n_grid_sim_input_ds, 
        K_output_resolution=args.k_snn_target_res, 
        hidden_channels=args.snn_hidden_channels,
        num_hidden_layers=args.snn_num_hidden_layers
    )
    print(f"\nSNN Model Architecture (Input K={args.n_grid_sim_input_ds}, Output K={args.k_snn_target_res}):\n{snn_model}")
    num_params = sum(p.numel() for p in snn_model.parameters() if p.requires_grad)
    print(f"Trainable SNN parameters: {num_params}")

    train_loss_hist, val_loss_hist, best_val_loss = train_snn_model(
        snn_model, 
        train_loader, val_loader, has_val_data_flag, args.epochs, 
        args.lr, DEVICE, full_model_save_path
    )
    if has_val_data_flag:
        print(f"Best validation MSE loss achieved: {best_val_loss:.4e}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.epochs + 1), train_loss_hist, label='Training Loss')
    if has_val_data_flag and val_loss_hist: 
        valid_val_losses = [loss for loss in val_loss_hist if loss != float('inf')]
        if valid_val_losses: 
             epochs_with_val = [i+1 for i, loss in enumerate(val_loss_hist) if loss != float('inf')]
             plt.plot(epochs_with_val, valid_val_losses, label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (on Truncated Output)')
    plt.yscale('log')
    plt.title(f'SNN Training (K_in={args.n_grid_sim_input_ds}, K_target={args.k_snn_target_res}, PDE: {args.pde_type}, Params: {filename_suffix})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(full_plot_save_path)
    print(f"\nTraining loss plot saved to {full_plot_save_path}")
    plt.show()
    print("\nSNN training script finished.")