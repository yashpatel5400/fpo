import numpy as np
import matplotlib.pyplot as plt
import os

from train import (random_low_order_state, solver, 
                                           harmonic_oscillator_potential, 
                                           barrier_potential, free_particle_potential,
                                           random_potential, paul_trap, GLOBAL_HBAR, GLOBAL_M)

def setup_state_evolution_for_discrimination(config):
    """
    Generates two distinct initial states and evolves them using the true solver.
    """
    N_grid = config['n_grid']
    L_domain = config['l_domain']
    K_psi0_modes = config['k_psi0_modes']
    T_evolution = config['t_evolution']
    num_solver_steps = config['num_solver_steps']
    hamiltonian_name = config['hamiltonian_name']
    hamiltonian_params = config['hamiltonian_params']
    
    dx = L_domain / N_grid

    print(f"Setting up states for discrimination with Hamiltonian: {hamiltonian_name}")

    # 1. Generate two distinct initial states (psi0_1, psi0_2)
    # Ensure they are different by, for example, different random seeds or just calling twice
    # For simplicity, we call random_low_order_state twice; they are statistically independent.
    psi0_1_real_space = random_low_order_state(N_grid, K=K_psi0_modes)
    psi0_2_real_space = random_low_order_state(N_grid, K=K_psi0_modes)

    # Optional: Check if states are indeed different (e.g. by norm of difference)
    # This is highly likely for random_low_order_state.
    # if np.linalg.norm(psi0_1_real_space - psi0_2_real_space) < 1e-9:
    #     print("Warning: Initial states psi0_1 and psi0_2 are very similar. Resampling psi0_2.")
    #     psi0_2_real_space = random_low_order_state(N_grid, K=K_psi0_modes)


    # 2. Define the potential V
    V_potential_arg = None
    if hamiltonian_name == 'free_particle':
        V_potential_arg = free_particle_potential(N_grid)
    elif hamiltonian_name == 'barrier':
        V_potential_arg = barrier_potential(N_grid, L_domain, **hamiltonian_params)
    elif hamiltonian_name == 'harmonic_oscillator':
        V_potential_arg = harmonic_oscillator_potential(N_grid, L_domain, **hamiltonian_params)
    elif hamiltonian_name == 'random_potential':
        V_potential_arg = random_potential(N_grid, **hamiltonian_params)
    elif hamiltonian_name == 'paul_trap':
        # For Paul trap, V_potential_arg is a function of time
        V_potential_arg = lambda t: paul_trap(N_grid, L_domain, t, **hamiltonian_params)
    else:
        raise ValueError(f"Unknown Hamiltonian name: {hamiltonian_name}")

    # 3. Evolve psi0_1 and psi0_2 to psiT_1 and psiT_2 using the "true" solver
    print(f"Evolving psi0_1 (shape: {psi0_1_real_space.shape}) for T={T_evolution}...")
    psiT_1_real_space = solver(V_potential_arg, psi0_1_real_space, N_grid, dx, 
                               T_evolution, num_solver_steps)
    
    print(f"Evolving psi0_2 (shape: {psi0_2_real_space.shape}) for T={T_evolution}...")
    psiT_2_real_space = solver(V_potential_arg, psi0_2_real_space, N_grid, dx, 
                               T_evolution, num_solver_steps)

    print("Evolution complete.")
    print(f"  psi0_1 norm: {np.linalg.norm(psi0_1_real_space):.4f}, psiT_1 norm: {np.linalg.norm(psiT_1_real_space):.4f}")
    print(f"  psi0_2 norm: {np.linalg.norm(psi0_2_real_space):.4f}, psiT_2 norm: {np.linalg.norm(psiT_2_real_space):.4f}")
    
    # Optional: Calculate overlap <psiT_1 | psiT_2>
    overlap_T = np.vdot(psiT_1_real_space.ravel(), psiT_2_real_space.ravel())
    print(f"  Overlap <psiT_1 | psiT_2>: {overlap_T:.4f} (Magnitude: {np.abs(overlap_T):.4f})")


    return (psi0_1_real_space, psiT_1_real_space), (psi0_2_real_space, psiT_2_real_space), V_potential_arg


if __name__ == '__main__':
    # --- Configuration for State Discrimination Setup ---
    config_discrimination = {
        'n_grid': 64,
        'l_domain': 2 * np.pi,
        'k_psi0_modes': 8, # Max K for initial states, e.g. -8 to 8
        't_evolution': 0.1,
        'num_solver_steps': 50,
        'hamiltonian_name': 'harmonic_oscillator', #'paul_trap', #'barrier', 
        'hamiltonian_params': {} # Specific params below
    }

    # Set Hamiltonian-specific parameters
    if config_discrimination['hamiltonian_name'] == 'barrier':
        config_discrimination['hamiltonian_params'] = {'barrier_height': 100.0, 'slit_width_ratio': 0.15}
    elif config_discrimination['hamiltonian_name'] == 'harmonic_oscillator':
        config_discrimination['hamiltonian_params'] = {'omega': 5.0, 'm_potential': 1.0}
    elif config_discrimination['hamiltonian_name'] == 'random_potential':
        config_discrimination['hamiltonian_params'] = {'alpha': 0.5, 'beta': 0.2, 'gamma': 2.5}
    elif config_discrimination['hamiltonian_name'] == 'paul_trap':
         config_discrimination['hamiltonian_params'] = {
            'U0': 2.0, 'V0': 10.0, 
            'omega_trap': (2.0 * np.pi / config_discrimination['t_evolution']) * 2.0, 
            'r0_sq_factor': 0.05 
        }
    
    # --- Run the setup ---
    (psi0_1, psiT_1), (psi0_2, psiT_2), V_potential = setup_state_evolution_for_discrimination(config_discrimination)

    # --- Basic Visualization (Optional) ---
    # Create a directory for plots if it doesn't exist
    RESULTS_DIR_DISCRIM = "results_discrimination"
    os.makedirs(RESULTS_DIR_DISCRIM, exist_ok=True)

    print(f"\nVisualizing initial and final states (magnitudes)... Plots saved to {RESULTS_DIR_DISCRIM}/")

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    plot_options = {'cmap': 'viridis'}

    # Plot psi0_1
    im00 = axes[0,0].imshow(np.abs(psi0_1), **plot_options)
    axes[0,0].set_title(r'$|\psi_{0,1}|$')
    axes[0,0].axis('off')
    fig.colorbar(im00, ax=axes[0,0], fraction=0.046, pad=0.04)

    # Plot psiT_1
    im01 = axes[0,1].imshow(np.abs(psiT_1), **plot_options)
    axes[0,1].set_title(r'$|\psi_{T,1}|$ (Evolved from $\psi_{0,1}$)')
    axes[0,1].axis('off')
    fig.colorbar(im01, ax=axes[0,1], fraction=0.046, pad=0.04)

    # Plot psi0_2
    im10 = axes[1,0].imshow(np.abs(psi0_2), **plot_options)
    axes[1,0].set_title(r'$|\psi_{0,2}|$')
    axes[1,0].axis('off')
    fig.colorbar(im10, ax=axes[1,0], fraction=0.046, pad=0.04)

    # Plot psiT_2
    im11 = axes[1,1].imshow(np.abs(psiT_2), **plot_options)
    axes[1,1].set_title(r'$|\psi_{T,2}|$ (Evolved from $\psi_{0,2}$)')
    axes[1,1].axis('off')
    fig.colorbar(im11, ax=axes[1,1], fraction=0.046, pad=0.04)
    
    fig.suptitle(f'Initial and Evolved States for Discrimination ({config_discrimination["hamiltonian_name"]})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    fig_path = os.path.join(RESULTS_DIR_DISCRIM, f"evolved_states_{config_discrimination['hamiltonian_name']}.png")
    plt.savefig(fig_path)
    print(f"State visualization saved to {fig_path}")
    plt.show()

    # If potential is time-independent, visualize it
    if isinstance(V_potential, np.ndarray):
        plt.figure(figsize=(6,5))
        plt.imshow(V_potential, cmap='inferno')
        plt.title(f"Potential V(x,y) for {config_discrimination['hamiltonian_name']}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar(label="Potential Energy")
        plt.tight_layout()
        pot_fig_path = os.path.join(RESULTS_DIR_DISCRIM, f"potential_{config_discrimination['hamiltonian_name']}.png")
        plt.savefig(pot_fig_path)
        print(f"Potential visualization saved to {pot_fig_path}")
        plt.show()

