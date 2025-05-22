import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.fft import ifftn, fft2, fftshift, ifftshift # Ensure all fft functions are imported
import matplotlib.pyplot as plt
import os # Added for creating directory
import argparse # Added for command-line arguments

from constants import GLOBAL_HBAR, GLOBAL_M
from dataset import random_low_order_state, free_particle_potential, barrier_potential, harmonic_oscillator_potential, random_potential, paul_trap
from solvers import solver

# --- New Components for Spectral Operator ---
def get_truncated_spectrum(psi_real_space, K_trunc):
    if psi_real_space.ndim != 2 or psi_real_space.shape[0] != psi_real_space.shape[1]:
        raise ValueError("Input psi_real_space must be a square 2D array.")
    N_grid_psi = psi_real_space.shape[0]
    F_psi_shifted = np.fft.fftshift(np.fft.fft2(psi_real_space))
    if K_trunc > N_grid_psi:
        raise ValueError(f"K_trunc ({K_trunc}) cannot be larger than N_grid_psi ({N_grid_psi}).")
    start_idx = N_grid_psi // 2 - K_trunc // 2
    end_idx = start_idx + K_trunc
    return F_psi_shifted[start_idx:end_idx, start_idx:end_idx]

def spectrum_to_channels(spectrum_mat_complex):
    return np.stack([spectrum_mat_complex.real, spectrum_mat_complex.imag], axis=0)

def channels_to_spectrum(channels_mat_real_imag):
    if channels_mat_real_imag.shape[0] != 2:
        raise ValueError("Input must have 2 channels (real and imaginary parts).")
    return channels_mat_real_imag[0] + 1j * channels_mat_real_imag[1]

class SpectralDataset(Dataset):
    def __init__(self, spectral_pairs_ch_list): 
        self.samples = spectral_pairs_ch_list
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        spec0_ch, specT_ch = self.samples[idx]
        return torch.from_numpy(spec0_ch).float(), torch.from_numpy(specT_ch).float()

def construct_spectral_loader(spectral_pairs_ch_list, batch_size, shuffle=True):
    dataset = SpectralDataset(spectral_pairs_ch_list)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class SimpleSpectralOperatorCNN(nn.Module):
    def __init__(self, K_trunc_net, hidden_channels=64, num_hidden_layers=3):
        super().__init__()
        self.K_trunc_net = K_trunc_net         
        layers = []
        layers.append(nn.Conv2d(2, hidden_channels, kernel_size=3, padding='same'))
        layers.append(nn.ReLU())
        for _ in range(num_hidden_layers):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding='same'))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(hidden_channels, 2, kernel_size=3, padding='same'))
        self.model = nn.Sequential(*layers)
    def forward(self, x_spec_ch_batch): 
        return self.model(x_spec_ch_batch)

def generate_spectral_data(num_samples, N_grid_sim, L_domain_sim, T_evol, num_sol_steps,
                           ham_name, ham_params, K_psi0_modes, K_trunc_for_network,
                           hbar_sim=GLOBAL_HBAR, m_sim=GLOBAL_M): # Removed solver_choice_sim
    spectral_pairs_ch_list = []
    dx_sim = L_domain_sim / N_grid_sim
    print(f"Generating {num_samples} spectral data pairs using split-step solver...")
    print(f"  Simulation grid: {N_grid_sim}x{N_grid_sim}, Domain size: {L_domain_sim:.2f}")
    print(f"  Initial state K_modes: {K_psi0_modes}, Network K_trunc: {K_trunc_for_network}")

    for i in range(num_samples):
        if (i+1) % max(1, num_samples//20) == 0: 
            print(f"  Generating sample {i+1}/{num_samples}...")
        psi0_real_space = random_low_order_state(N_grid_sim, K=K_psi0_modes)
        V_potential_arg_sim = None
        if ham_name == 'free_particle':
            V_potential_arg_sim = free_particle_potential(N_grid_sim)
        elif ham_name == 'barrier':
            V_potential_arg_sim = barrier_potential(N_grid_sim, L_domain_sim, **ham_params)
        elif ham_name == 'harmonic_oscillator':
            V_potential_arg_sim = harmonic_oscillator_potential(N_grid_sim, L_domain_sim, **ham_params)
        elif ham_name == 'random_potential':
            V_potential_arg_sim = random_potential(N_grid_sim, **ham_params)
        elif ham_name == 'paul_trap':
            V_potential_arg_sim = lambda t: paul_trap(N_grid_sim, L_domain_sim, t, **ham_params)
        else:
            raise ValueError(f"Unknown Hamiltonian name: {ham_name}")
        
        # Solver call now directly uses the appropriate split-step method
        psiT_real_space = solver(V_potential_arg_sim, psi0_real_space, N_grid_sim, dx_sim, 
                                 T_evol, num_sol_steps, 
                                 hbar=hbar_sim, m=m_sim) 
                                 
        spec0_trunc_complex = get_truncated_spectrum(psi0_real_space, K_trunc_for_network)
        specT_trunc_complex = get_truncated_spectrum(psiT_real_space, K_trunc_for_network)
        spec0_ch_arr = spectrum_to_channels(spec0_trunc_complex)
        specT_ch_arr = spectrum_to_channels(specT_trunc_complex)
        spectral_pairs_ch_list.append((spec0_ch_arr, specT_ch_arr))
    print("Data generation complete.")
    return spectral_pairs_ch_list

def train_spectral_operator(model_nn, train_data_loader, val_data_loader, num_train_epochs, 
                            learn_rate, torch_device):
    model_nn.to(torch_device)
    optimizer = optim.Adam(model_nn.parameters(), lr=learn_rate)
    criterion = nn.MSELoss() 
    train_loss_history = []
    val_loss_history = []
    print(f"\nStarting training of spectral operator for {num_train_epochs} epochs on {torch_device}...")
    for epoch_idx in range(num_train_epochs):
        model_nn.train() 
        current_epoch_train_loss = 0.0
        for spec_psi0_ch_batch, spec_psiT_ch_batch_true in train_data_loader:
            spec_psi0_ch_batch = spec_psi0_ch_batch.to(torch_device)
            spec_psiT_ch_batch_true = spec_psiT_ch_batch_true.to(torch_device)
            optimizer.zero_grad() 
            spec_psiT_ch_batch_pred = model_nn(spec_psi0_ch_batch)
            loss = criterion(spec_psiT_ch_batch_pred, spec_psiT_ch_batch_true)
            loss.backward() 
            optimizer.step() 
            current_epoch_train_loss += loss.item() * spec_psi0_ch_batch.size(0) 
        avg_epoch_train_loss = current_epoch_train_loss / len(train_data_loader.dataset)
        train_loss_history.append(avg_epoch_train_loss)
        model_nn.eval() 
        current_epoch_val_loss = 0.0
        with torch.no_grad(): 
            for spec_psi0_ch_batch_val, spec_psiT_ch_batch_true_val in val_data_loader:
                spec_psi0_ch_batch_val = spec_psi0_ch_batch_val.to(torch_device)
                spec_psiT_ch_batch_true_val = spec_psiT_ch_batch_true_val.to(torch_device)
                spec_psiT_ch_batch_pred_val = model_nn(spec_psi0_ch_batch_val)
                val_loss_item = criterion(spec_psiT_ch_batch_pred_val, spec_psiT_ch_batch_true_val)
                current_epoch_val_loss += val_loss_item.item() * spec_psi0_ch_batch_val.size(0)
        avg_epoch_val_loss = current_epoch_val_loss / len(val_data_loader.dataset)
        val_loss_history.append(avg_epoch_val_loss)
        print(f"Epoch [{epoch_idx+1}/{num_train_epochs}], Train Loss: {avg_epoch_train_loss:.4e}, Val Loss: {avg_epoch_val_loss:.4e}")
    return train_loss_history, val_loss_history

# --- Main Execution Block ---
def main(args):
    # Apply parsed arguments
    N_GRID_SIMULATION = args.n_grid
    L_DOMAIN_SIMULATION = args.l_domain
    K_PSI0_MAX_MODES = args.k_psi0_modes
    K_TRUNC_FOR_NETWORK = args.k_trunc_network
    NUM_TRAIN_SAMPLES = args.num_train_samples
    NUM_VAL_SAMPLES = args.num_val_samples
    T_EVOLUTION_TOTAL = args.t_evolution
    NUM_SOLVER_STEPS_SIM = args.num_solver_steps
    SELECTED_HAMILTONIAN_NAME = args.hamiltonian
    # SOLVER_TYPE_FOR_DATA_GEN = args.solver_type # Removed
    TRAIN_BATCH_SIZE = args.batch_size
    NN_LEARNING_RATE = args.lr
    NN_NUM_EPOCHS = args.epochs
    CNN_HIDDEN_CHANNELS = args.hidden_channels
    CNN_NUM_HIDDEN_LAYERS = args.hidden_layers
    RESULTS_DIR = args.results_dir

    CHOSEN_HAMILTONIAN_PARAMS = {}
    if SELECTED_HAMILTONIAN_NAME == 'barrier':
        CHOSEN_HAMILTONIAN_PARAMS = {'barrier_height': args.barrier_height, 'slit_width_ratio': args.slit_width_ratio}
    elif SELECTED_HAMILTONIAN_NAME == 'harmonic_oscillator':
        CHOSEN_HAMILTONIAN_PARAMS = {'omega': args.ho_omega, 'm_potential': args.ho_m_potential}
    elif SELECTED_HAMILTONIAN_NAME == 'random_potential':
        CHOSEN_HAMILTONIAN_PARAMS = {'alpha': args.rp_alpha, 'beta': args.rp_beta, 'gamma': args.rp_gamma}
    elif SELECTED_HAMILTONIAN_NAME == 'paul_trap':
        CHOSEN_HAMILTONIAN_PARAMS = {
            'U0': args.pt_U0, 'V0': args.pt_V0, 
            'omega_trap': args.pt_omega_trap_factor * (2.0 * np.pi / T_EVOLUTION_TOTAL), 
            'r0_sq_factor': args.pt_r0_sq_factor
        }

    # Logic for solver_type is simplified as only split-step is used.
    # Paul trap inherently implies time-varying split_step.
    # Other Hamiltonians (if ndarray) use time-independent split_step.

    PYTORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {PYTORCH_DEVICE}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Figures will be saved to '{RESULTS_DIR}/' directory.")

    print("\n--- Generating Training Data ---")
    train_spectral_data_list = generate_spectral_data(
        num_samples=NUM_TRAIN_SAMPLES, N_grid_sim=N_GRID_SIMULATION, L_domain_sim=L_DOMAIN_SIMULATION, 
        T_evol=T_EVOLUTION_TOTAL, num_sol_steps=NUM_SOLVER_STEPS_SIM, ham_name=SELECTED_HAMILTONIAN_NAME, 
        ham_params=CHOSEN_HAMILTONIAN_PARAMS, K_psi0_modes=K_PSI0_MAX_MODES, 
        K_trunc_for_network=K_TRUNC_FOR_NETWORK
    )
    print("\n--- Generating Validation Data ---")
    val_spectral_data_list = generate_spectral_data(
        num_samples=NUM_VAL_SAMPLES, N_grid_sim=N_GRID_SIMULATION, L_domain_sim=L_DOMAIN_SIMULATION, 
        T_evol=T_EVOLUTION_TOTAL, num_sol_steps=NUM_SOLVER_STEPS_SIM, ham_name=SELECTED_HAMILTONIAN_NAME, 
        ham_params=CHOSEN_HAMILTONIAN_PARAMS, K_psi0_modes=K_PSI0_MAX_MODES, 
        K_trunc_for_network=K_TRUNC_FOR_NETWORK
    )

    train_loader_nn = construct_spectral_loader(train_spectral_data_list, TRAIN_BATCH_SIZE, shuffle=True)
    val_loader_nn = construct_spectral_loader(val_spectral_data_list, TRAIN_BATCH_SIZE, shuffle=False)

    spectral_op_model = SimpleSpectralOperatorCNN(
        K_trunc_net=K_TRUNC_FOR_NETWORK, hidden_channels=CNN_HIDDEN_CHANNELS, 
        num_hidden_layers=CNN_NUM_HIDDEN_LAYERS
    )
    print(f"\n--- Spectral Operator Model Architecture ---")
    print(spectral_op_model)
    num_trainable_params = sum(p.numel() for p in spectral_op_model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_trainable_params}")

    training_losses, validation_losses = train_spectral_operator(
        model_nn=spectral_op_model, train_data_loader=train_loader_nn, val_data_loader=val_loader_nn, 
        num_train_epochs=NN_NUM_EPOCHS, learn_rate=NN_LEARNING_RATE, torch_device=PYTORCH_DEVICE
    )

    plt.figure(figsize=(12, 7)) 
    plt.plot(range(1, NN_NUM_EPOCHS + 1), training_losses, marker='o', linestyle='-', label='Training Loss')
    plt.plot(range(1, NN_NUM_EPOCHS + 1), validation_losses, marker='x', linestyle='--', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (MSE) Loss')
    plt.title(rf'Spectral Operator Training ({SELECTED_HAMILTONIAN_NAME})') 
    plt.legend()
    plt.grid(True)
    plt.yscale('log') 
    plt.tight_layout()
    loss_fig_path = os.path.join(RESULTS_DIR, f"training_loss_{SELECTED_HAMILTONIAN_NAME}.png")
    plt.savefig(loss_fig_path)
    print(f"Loss plot saved to {loss_fig_path}")
    plt.show()

    if len(val_spectral_data_list) > 0:
        print("\n--- Visualizing Example Predictions ---")
        spectral_op_model.eval() 
        num_samples_to_plot = min(len(val_spectral_data_list), args.num_plot_samples) 

        for i_sample in range(num_samples_to_plot):
            with torch.no_grad():
                psi0_spec_ch_val_sample, psiT_spec_ch_true_val_sample = val_spectral_data_list[i_sample]
                psi0_torch_input = torch.from_numpy(psi0_spec_ch_val_sample).float().unsqueeze(0).to(PYTORCH_DEVICE)
                psiT_spec_ch_pred_val_sample_torch = spectral_op_model(psi0_torch_input)
                psiT_spec_ch_pred_val_sample = psiT_spec_ch_pred_val_sample_torch.squeeze(0).cpu().numpy()

                psi0_spec_complex_val = channels_to_spectrum(psi0_spec_ch_val_sample)
                psiT_spec_complex_true_val = channels_to_spectrum(psiT_spec_ch_true_val_sample)
                psiT_spec_complex_pred_val = channels_to_spectrum(psiT_spec_ch_pred_val_sample)

                error_norm = np.linalg.norm(psiT_spec_complex_true_val - psiT_spec_complex_pred_val)
                true_norm = np.linalg.norm(psiT_spec_complex_true_val)
                relative_spectral_error = error_norm / true_norm if true_norm > 1e-12 else error_norm 
                print(f"Validation Sample [{i_sample+1}/{num_samples_to_plot}] - Relative Spectral Error: {relative_spectral_error:.4e}")

                psi0_real_from_spec = np.fft.ifft2(np.fft.ifftshift(psi0_spec_complex_val))
                psiT_real_true_from_spec = np.fft.ifft2(np.fft.ifftshift(psiT_spec_complex_true_val))
                psiT_real_pred_from_spec = np.fft.ifft2(np.fft.ifftshift(psiT_spec_complex_pred_val))
                
                fig_preds, axes_preds = plt.subplots(2, 3, figsize=(18, 12)) 
                plot_options = {'cmap': 'viridis'} 
                
                mag_psi0_spec = np.abs(psi0_spec_complex_val)
                im0 = axes_preds[0,0].imshow(mag_psi0_spec, **plot_options)
                axes_preds[0,0].set_title(rf'Input Spec $|\mathcal{{F}}(\psi_0)|$ (K={K_TRUNC_FOR_NETWORK})')
                axes_preds[0,0].axis('off')
                fig_preds.colorbar(im0, ax=axes_preds[0,0], fraction=0.046, pad=0.04)

                mag_psiT_spec_true = np.abs(psiT_spec_complex_true_val)
                im1 = axes_preds[0,1].imshow(mag_psiT_spec_true, **plot_options)
                axes_preds[0,1].set_title(rf'True Target Spec $|\mathcal{{F}}(\psi_T)|$')
                axes_preds[0,1].axis('off')
                fig_preds.colorbar(im1, ax=axes_preds[0,1], fraction=0.046, pad=0.04)

                mag_psiT_spec_pred = np.abs(psiT_spec_complex_pred_val)
                im2 = axes_preds[0,2].imshow(mag_psiT_spec_pred, **plot_options)
                axes_preds[0,2].set_title(rf'Predicted Spec $|\mathcal{{F}}(\widehat{{\psi}}_T)|$')
                axes_preds[0,2].axis('off')
                fig_preds.colorbar(im2, ax=axes_preds[0,2], fraction=0.046, pad=0.04)

                mag_psi0_real = np.abs(psi0_real_from_spec)
                im3 = axes_preds[1,0].imshow(mag_psi0_real, **plot_options)
                axes_preds[1,0].set_title(rf'Input Real $|\psi_0|$ (from K={K_TRUNC_FOR_NETWORK} spec)')
                axes_preds[1,0].axis('off')
                fig_preds.colorbar(im3, ax=axes_preds[1,0], fraction=0.046, pad=0.04)
                
                mag_psiT_real_true = np.abs(psiT_real_true_from_spec)
                im4 = axes_preds[1,1].imshow(mag_psiT_real_true, **plot_options)
                axes_preds[1,1].set_title(rf'True Target Real $|\psi_T|$ (from spec)')
                axes_preds[1,1].axis('off')
                fig_preds.colorbar(im4, ax=axes_preds[1,1], fraction=0.046, pad=0.04)

                mag_psiT_real_pred = np.abs(psiT_real_pred_from_spec)
                im5 = axes_preds[1,2].imshow(mag_psiT_real_pred, **plot_options)
                axes_preds[1,2].set_title(rf'Predicted Real $|\widehat{{\psi}}_T|$ (from spec)')
                axes_preds[1,2].axis('off')
                fig_preds.colorbar(im5, ax=axes_preds[1,2], fraction=0.046, pad=0.04)
                
                fig_preds.suptitle(rf"Example Spectral & Real-Space Prediction Sample {i_sample+1} ({SELECTED_HAMILTONIAN_NAME}, T={T_EVOLUTION_TOTAL:.2f})", fontsize=16)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
                pred_fig_path = os.path.join(RESULTS_DIR, f"prediction_sample_{i_sample+1}_{SELECTED_HAMILTONIAN_NAME}_spec_and_real.png")
                plt.savefig(pred_fig_path)
                print(f"Prediction plot for sample {i_sample+1} saved to {pred_fig_path}")
                plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Spectral Neural Operator for Schrödinger Equation using Split-Step Solver.")
    # Data Generation & Simulation Parameters
    parser.add_argument('--n_grid', type=int, default=64, help='Real-space grid size for simulation (e.g., N x N).')
    parser.add_argument('--l_domain', type=float, default=2*np.pi, help='Physical size of the simulation domain.')
    parser.add_argument('--k_psi0_modes', type=int, default=8, help='Max mode index K for initial random state (modes from -K to K).')
    parser.add_argument('--k_trunc_network', type=int, default=32, help='Size of the truncated spectrum for NN (K_trunc x K_trunc).')
    parser.add_argument('--num_train_samples', type=int, default=500, help='Number of training samples.')
    parser.add_argument('--num_val_samples', type=int, default=100, help='Number of validation samples.')
    # Evolution Parameters
    parser.add_argument('--t_evolution', type=float, default=0.1, help='Total evolution time.')
    parser.add_argument('--num_solver_steps', type=int, default=50, help='Number of steps for the numerical solver.')
    # Hamiltonian Choice & Parameters
    parser.add_argument('--hamiltonian', type=str, default='harmonic_oscillator', 
                        choices=['free_particle', 'barrier', 'harmonic_oscillator', 'random_potential', 'paul_trap'],
                        help='Name of the Hamiltonian to use.')
    parser.add_argument('--barrier_height', type=float, default=100.0, help='Height of the barrier (for barrier Hamiltonian).')
    parser.add_argument('--slit_width_ratio', type=float, default=0.15, help='Slit width as a ratio of N_grid (for barrier Hamiltonian).')
    parser.add_argument('--ho_omega', type=float, default=5.0, help='Omega for harmonic oscillator.')
    parser.add_argument('--ho_m_potential', type=float, default=1.0, help='Mass parameter in harmonic oscillator potential (distinct from particle mass).')
    parser.add_argument('--rp_alpha', type=float, default=0.5, help='Alpha for random potential (GRF).')
    parser.add_argument('--rp_beta', type=float, default=0.2, help='Beta for random potential (GRF).')
    parser.add_argument('--rp_gamma', type=float, default=2.5, help='Gamma for random potential (GRF).')
    parser.add_argument('--pt_U0', type=float, default=2.0, help='U0 for Paul trap.')
    parser.add_argument('--pt_V0', type=float, default=10.0, help='V0 for Paul trap.')
    parser.add_argument('--pt_omega_trap_factor', type=float, default=2.0, help='Factor to multiply (2pi/T) for Paul trap omega.')
    parser.add_argument('--pt_r0_sq_factor', type=float, default=0.05, help='Factor for r0^2 calculation in Paul trap ( (L_domain * factor)^2 ).')
    # Neural Network Training Parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for Adam optimizer.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs.')
    parser.add_argument('--hidden_channels', type=int, default=32, help='Number of hidden channels in CNN.')
    parser.add_argument('--hidden_layers', type=int, default=2, help='Number of hidden layers in CNN.')
    # Output and Plotting
    parser.add_argument('--results_dir', type=str, default="results", help='Directory to save output figures.')
    parser.add_argument('--num_plot_samples', type=int, default=3, help='Number of validation samples to plot.')

    cli_args = parser.parse_args()
    main(cli_args)
