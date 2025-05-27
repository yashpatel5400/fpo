import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import os

# --- Spectral Neural Operator Model ---
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
    """Converts a complex KxK spectrum tensor to a 2xKxK real/imaginary channel tensor."""
    if not isinstance(spectrum_mat_complex, torch.Tensor):
        spectrum_mat_complex = torch.from_numpy(spectrum_mat_complex)
    # Ensure input is complex
    if not torch.is_complex(spectrum_mat_complex):
        raise ValueError("Input spectrum_mat_complex must be a complex tensor.")
    return torch.stack([spectrum_mat_complex.real, spectrum_mat_complex.imag], dim=0)

def channels_to_spectrum_complex_torch(channels_mat_real_imag):
    """Converts a 2xKxK real/imaginary channel tensor back to a complex KxK spectrum tensor."""
    if channels_mat_real_imag.shape[0] != 2:
        raise ValueError("Input must have 2 channels (real and imaginary parts).")
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

        # Convert to 2-channel float tensors (real, imag)
        # The conversion function should handle numpy array input
        gamma_b_channels = spectrum_complex_to_channels_torch(gamma_b_complex).float()
        gamma_a_channels = spectrum_complex_to_channels_torch(gamma_a_complex).float()
        
        return gamma_b_channels, gamma_a_channels

def load_and_prepare_dataloaders(dataset_path, K_trunc_expected, batch_size, val_split=0.2, random_seed=42):
    """Loads data, creates datasets, and prepares dataloaders."""
    try:
        data = np.load(dataset_path)
        gamma_b_all = data['gamma_b']
        gamma_a_all = data['gamma_a']
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}")
        print("Please ensure 'phenomenological_channel_dataset_generation.py' (or 'noised_data.py')")
        print("has been run and the dataset is saved at the correct location.")
        return None, None
    except KeyError:
        print(f"Error: Dataset file {dataset_path} does not contain 'gamma_b' or 'gamma_a' keys.")
        return None, None

    if gamma_b_all.shape[1] != K_trunc_expected or gamma_b_all.shape[2] != K_trunc_expected:
        raise ValueError(f"K_trunc_expected ({K_trunc_expected}) does not match K_trunc in loaded data ({gamma_b_all.shape[1]}).")

    dataset = SpectralDataset(gamma_b_all, gamma_a_all)
    
    # Split into training and validation sets
    num_samples = len(dataset)
    val_size = int(val_split * num_samples)
    train_size = num_samples - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], 
                                              generator=torch.Generator().manual_seed(random_seed))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) # num_workers=0 for simplicity
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Dataset loaded: {num_samples} total samples.")
    print(f"Training set: {train_size} samples, Validation set: {val_size} samples.")
    return train_loader, val_loader

# --- Training Loop ---
def train_snn_model(model, train_loader, val_loader, num_epochs, learning_rate, device, model_save_path):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss() # Mean Squared Error on the 2-channel real/imaginary representation
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\nStarting SNN training for {num_epochs} epochs on {device}...")
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for gamma_b_ch, gamma_a_ch_true in train_loader:
            gamma_b_ch = gamma_b_ch.to(device)
            gamma_a_ch_true = gamma_a_ch_true.to(device)
            
            optimizer.zero_grad()
            gamma_a_ch_pred = model(gamma_b_ch)
            loss = criterion(gamma_a_ch_pred, gamma_a_ch_true)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * gamma_b_ch.size(0) # Accumulate loss weighted by batch size
        
        avg_epoch_train_loss = epoch_train_loss / len(train_loader.dataset)
        train_losses.append(avg_epoch_train_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for gamma_b_ch_val, gamma_a_ch_true_val in val_loader:
                gamma_b_ch_val = gamma_b_ch_val.to(device)
                gamma_a_ch_true_val = gamma_a_ch_true_val.to(device)
                gamma_a_ch_pred_val = model(gamma_b_ch_val)
                val_loss_item = criterion(gamma_a_ch_pred_val, gamma_a_ch_true_val)
                epoch_val_loss += val_loss_item.item() * gamma_b_ch_val.size(0)
        
        avg_epoch_val_loss = epoch_val_loss / len(val_loader.dataset)
        val_losses.append(avg_epoch_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_epoch_train_loss:.4e}, Val Loss: {avg_epoch_val_loss:.4e}")

        # Save the model if validation loss improves
        if avg_epoch_val_loss < best_val_loss:
            best_val_loss = avg_epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  Model saved to {model_save_path} (Val Loss: {best_val_loss:.4e})")
            
    print("Training complete.")
    return train_losses, val_losses, best_val_loss

if __name__ == '__main__':
    # --- Configuration ---
    # This K_TRUNC_FOR_SPECTRA must match the one used in dataset generation
    K_TRUNC_SNN = 32  
    DATASET_FILE_PATH = "datasets/phenomenological_channel_dataset.npz" # Path to the generated dataset
    
    # SNN Architecture
    SNN_HIDDEN_CHANNELS = 64
    SNN_NUM_HIDDEN_LAYERS = 3
    
    # Training Parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50 # Adjust as needed (e.g., 50-200)
    VALIDATION_SPLIT = 0.2 # 20% of data for validation
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_SAVE_DIR = "trained_snn_models"
    MODEL_FILENAME = f"snn_K{K_TRUNC_SNN}_H{SNN_HIDDEN_CHANNELS}_L{SNN_NUM_HIDDEN_LAYERS}.pth"
    PLOT_SAVE_DIR = "results_snn_training"

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
    full_model_save_path = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)
    
    print("--- Spectral Neural Operator Training ---")
    print(f"Using device: {DEVICE}")
    print(f"Expected K_trunc for SNN: {K_TRUNC_SNN}")
    print(f"Loading dataset from: {DATASET_FILE_PATH}")

    # 1. Load and Prepare Data
    train_loader, val_loader = load_and_prepare_dataloaders(
        DATASET_FILE_PATH, 
        K_TRUNC_SNN, 
        BATCH_SIZE, 
        val_split=VALIDATION_SPLIT
    )

    if train_loader is None or val_loader is None:
        print("Failed to load data. Exiting.")
        exit()

    # 2. Initialize SNN Model
    snn_model = SimpleSpectralOperatorCNN(
        K_trunc=K_TRUNC_SNN,
        hidden_channels=SNN_HIDDEN_CHANNELS,
        num_hidden_layers=SNN_NUM_HIDDEN_LAYERS
    )
    print(f"\nSNN Model Architecture:\n{snn_model}")
    num_params = sum(p.numel() for p in snn_model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")

    # 3. Train the Model
    train_loss_history, val_loss_history, best_val_loss_achieved = train_snn_model(
        snn_model, 
        train_loader, 
        val_loader, 
        NUM_EPOCHS, 
        LEARNING_RATE, 
        DEVICE,
        full_model_save_path
    )
    print(f"Best validation loss achieved: {best_val_loss_achieved:.4e}")

    # 4. Plot Loss Curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, NUM_EPOCHS + 1), train_loss_history, label='Training Loss')
    plt.plot(range(1, NUM_EPOCHS + 1), val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (on 2-channel spectra)')
    plt.title(f'SNN Training Progress (K_trunc={K_TRUNC_SNN})')
    plt.legend()
    plt.grid(True)
    plt.yscale('log') # Log scale is often helpful for losses
    plt.tight_layout()
    
    loss_plot_filename = os.path.join(PLOT_SAVE_DIR, f"snn_training_loss_K{K_TRUNC_SNN}.png")
    plt.savefig(loss_plot_filename)
    print(f"\nTraining loss plot saved to {loss_plot_filename}")
    plt.show()

    print("\nTo use the trained model for predictions:")
    print(f"1. Re-initialize the model: model = SimpleSpectralOperatorCNN(K_trunc={K_TRUNC_SNN}, ...)")
    print(f"2. Load the saved state dict: model.load_state_dict(torch.load('{full_model_save_path}'))")
    print(f"3. Set to evaluation mode: model.eval()")
