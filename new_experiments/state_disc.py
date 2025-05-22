import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats # For t-test
from itertools import product # For generating parameter combinations
import multiprocessing # Added for parallel execution

# Imports assuming a modular structure:
from constants import GLOBAL_HBAR, GLOBAL_M
from dataset import (
    get_mesh, 
    free_particle_potential,
    barrier_potential, 
    harmonic_oscillator_potential, 
    random_potential, 
    GRF, 
    paul_trap,
    random_low_order_state 
)
from solvers import solver

def compute_povm_probabilities(psi_real_space, K_measurement_cutoff, N_grid, include_last_bin=True):
    if psi_real_space.shape != (N_grid, N_grid):
        raise ValueError(f"psi_real_space shape {psi_real_space.shape} does not match N_grid ({N_grid},{N_grid})")
    F_psi = np.fft.fft2(psi_real_space)
    kx_eff_all = np.fft.fftfreq(N_grid) * N_grid 
    ky_eff_all = np.fft.fftfreq(N_grid) * N_grid
    probabilities = {}
    sum_probs_within_cutoff = 0.0
    for i_kx in range(N_grid): 
        for i_ky in range(N_grid): 
            kx_eff = int(round(kx_eff_all[i_kx])) 
            ky_eff = int(round(ky_eff_all[i_ky])) 
            if max(abs(kx_eff), abs(ky_eff)) < K_measurement_cutoff:
                prob_k = (np.abs(F_psi[i_kx, i_ky])**2) / (N_grid**2)
                probabilities[(kx_eff, ky_eff)] = prob_k
                sum_probs_within_cutoff += prob_k
    if include_last_bin:
        prob_last_bin = max(0.0, 1.0 - sum_probs_within_cutoff)
        probabilities['last_bin'] = prob_last_bin
        total_prob_check = sum(probabilities.values())
        if not np.isclose(total_prob_check, 1.0, atol=1e-9): pass
    return probabilities

def generate_distinct_single_fourier_modes(num_states, N_grid, K_max_mode):
    initial_states_real_space = []
    chosen_modes = set()
    max_possible_modes = (2 * K_max_mode + 1)**2
    if max_possible_modes < num_states:
        raise ValueError(f"Cannot generate {num_states} distinct Fourier modes with K_max_mode={K_max_mode}. Max possible is {max_possible_modes}.")
    for i in range(num_states):
        freq_array = np.zeros((N_grid, N_grid), dtype=np.complex128)
        def wrap_index(k_val): return k_val % N_grid
        kx_eff, ky_eff = -1, -1; mode_found = False; max_tries_per_state = 1000; tries = 0
        while not mode_found and tries < max_tries_per_state:
            kx_eff = np.random.randint(-K_max_mode, K_max_mode + 1)
            ky_eff = np.random.randint(-K_max_mode, K_max_mode + 1)
            if (kx_eff, ky_eff) not in chosen_modes: mode_found = True
            tries += 1
        if not mode_found:
            for kx_try in range(-K_max_mode, K_max_mode + 1):
                for ky_try in range(-K_max_mode, K_max_mode + 1):
                    if (kx_try, ky_try) not in chosen_modes:
                        kx_eff, ky_eff = kx_try, ky_try; mode_found = True; break
                if mode_found: break
            if not mode_found: raise RuntimeError(f"Could not find a unique Fourier mode for state {i+1}")
        chosen_modes.add((kx_eff, ky_eff)); c = 1.0 + 0.0j 
        freq_array[wrap_index(kx_eff), wrap_index(ky_eff)] = c
        psi_real_space = np.fft.ifft2(freq_array)
        norm_psi = np.linalg.norm(psi_real_space)
        if norm_psi > 1e-14: psi_real_space /= norm_psi
        else: psi_real_space = np.zeros((N_grid,N_grid), dtype=np.complex128); psi_real_space[0,0]=1.0/N_grid
        initial_states_real_space.append(psi_real_space)
    return initial_states_real_space, list(chosen_modes)

def setup_multistate_evolution(config, trial_num=None, process_id=None): 
    N_grid = config['n_grid']; L_domain = config['l_domain']; K_psi0_modes = config['k_psi0_modes'] 
    T_evolution = config['t_evolution']; num_solver_steps = config['num_solver_steps']
    hamiltonian_name = config['hamiltonian_name']; hamiltonian_params = config['hamiltonian_params']
    num_distinct_states = config['num_distinct_states']; dx = L_domain / N_grid 
    log_prefix = ""
    if process_id is not None: log_prefix += f"[P{process_id}]"
    if trial_num is not None: log_prefix += f"[T{trial_num}] "
    print_this_setup = trial_num is None or \
                       (trial_num is not None and trial_num % (config.get('num_trials_per_experiment', 100) // 10 or 1) == 1 or trial_num == 1) or \
                       config.get('num_trials_per_experiment', 100) <= 10 
    if print_this_setup: 
        print(f"{log_prefix}Setting up {num_distinct_states} states for H: {hamiltonian_name}, T={T_evolution:.3f}")
    initial_states_psi0, initial_modes = generate_distinct_single_fourier_modes(num_distinct_states, N_grid, K_psi0_modes)
    evolved_states_psiT = []
    V_potential_arg = None; current_ham_params = hamiltonian_params.copy()
    if hamiltonian_name == 'paul_trap' and 'omega_trap_factor' in current_ham_params:
        current_ham_params['omega_trap'] = current_ham_params['omega_trap_factor'] * (2.0 * np.pi / T_evolution) if T_evolution > 1e-9 else 0
    if hamiltonian_name == 'free_particle': V_potential_arg = free_particle_potential(N_grid)
    elif hamiltonian_name == 'barrier': V_potential_arg = barrier_potential(N_grid, L_domain, **current_ham_params)
    elif hamiltonian_name == 'harmonic_oscillator': V_potential_arg = harmonic_oscillator_potential(N_grid, L_domain, **current_ham_params)
    elif hamiltonian_name == 'random_potential': V_potential_arg = random_potential(N_grid, **current_ham_params) 
    elif hamiltonian_name == 'paul_trap': V_potential_arg = lambda t: paul_trap(N_grid, L_domain, t, **current_ham_params)
    else: raise ValueError(f"Unknown Hamiltonian name: {hamiltonian_name}")
    for i, psi0_j in enumerate(initial_states_psi0):
        psiT_j = psi0_j.copy() if T_evolution == 0 else solver(V_potential_arg, psi0_j, N_grid, dx, T_evolution, num_solver_steps, hbar=GLOBAL_HBAR, m=GLOBAL_M)
        norm_psiT = np.linalg.norm(psiT_j)
        if not np.isclose(norm_psiT, 1.0, atol=1e-5) and norm_psiT > 1e-9: psiT_j /= norm_psiT
        evolved_states_psiT.append(psiT_j)
    return initial_states_psi0, evolved_states_psiT, V_potential_arg, initial_modes

def get_optimal_decision_rule_multistate(list_povm_probs_psi_j, priors_p_j):
    decision_rule = {}; num_states = len(list_povm_probs_psi_j)
    all_measurement_outcomes = set().union(*(probs.keys() for probs in list_povm_probs_psi_j))
    for k_outcome in all_measurement_outcomes:
        max_weighted_prob = -1.0; best_state_idx = 0 
        for j in range(num_states):
            weighted_prob = priors_p_j[j] * list_povm_probs_psi_j[j].get(k_outcome, 0.0)
            if weighted_prob > max_weighted_prob: max_weighted_prob = weighted_prob; best_state_idx = j
        decision_rule[k_outcome] = best_state_idx 
    return decision_rule

def calculate_probability_of_error_multistate(decision_rule, list_povm_probs_psi_j, priors_p_j):
    prob_error = 0.0; num_states = len(list_povm_probs_psi_j)
    for j_true_idx in range(num_states): 
        error_sum_for_state_j = sum(prob_k for k, prob_k in list_povm_probs_psi_j[j_true_idx].items() if decision_rule.get(k,0) != j_true_idx)
        prob_error += priors_p_j[j_true_idx] * error_sum_for_state_j
    return prob_error

def add_spectral_noise_sobolev(psi_real_space, noise_level_base, N_grid, sobolev_order_s):
    """
    Adds complex Gaussian noise to psi_real_space, with noise std dev for mode n
    scaled by noise_level_base / (1 + |n|^2)**(sobolev_order_s / 2).
    """
    F_psi_true = np.fft.fft2(psi_real_space)
    F_psi_noisy = F_psi_true.copy()

    kx_eff_all = np.fft.fftfreq(N_grid) * N_grid 
    ky_eff_all = np.fft.fftfreq(N_grid) * N_grid

    for i_kx in range(N_grid):
        for i_ky in range(N_grid):
            nx = kx_eff_all[i_kx] # Effective integer wavenumbers
            ny = ky_eff_all[i_ky]
            
            # Sobolev weight for amplitude: (1 + |n|^2)^(s/2)
            # Noise std dev for mode n: base_std / weight
            sobolev_weight_factor = (1 + nx**2 + ny**2)**(sobolev_order_s / 2.0)
            if sobolev_weight_factor < 1e-9: # Avoid division by zero if s is very negative for n=0 (though s>=0 here)
                sobolev_weight_factor = 1.0 # Should be 1 for n=0 anyway if s>=0

            mode_noise_std_dev = noise_level_base / sobolev_weight_factor
            
            noise_real_part = np.random.randn() * mode_noise_std_dev
            noise_imag_part = np.random.randn() * mode_noise_std_dev
            F_psi_noisy[i_kx, i_ky] += (noise_real_part + 1j * noise_imag_part)
            
    psi_noisy_real_space = np.fft.ifft2(F_psi_noisy)
    norm_noisy = np.linalg.norm(psi_noisy_real_space)
    if norm_noisy > 1e-9: psi_noisy_real_space /= norm_noisy
    else: psi_noisy_real_space = np.zeros_like(psi_real_space) 
    return psi_noisy_real_space

def apply_error_operator_multistate(psi_state, decision_rule_w, true_state_idx_j, K_measurement_cutoff, N_grid):
    F_psi = np.fft.fft2(psi_state)
    F_O_psi = np.zeros_like(F_psi)
    kx_eff_all = np.fft.fftfreq(N_grid) * N_grid; ky_eff_all = np.fft.fftfreq(N_grid) * N_grid
    is_last_bin_error_inducing = (decision_rule_w.get('last_bin', -1) != true_state_idx_j)
    for i_kx in range(N_grid):
        for i_ky in range(N_grid):
            kx_eff = int(round(kx_eff_all[i_kx])); ky_eff = int(round(ky_eff_all[i_ky]))
            k_mode = (kx_eff, ky_eff)
            is_individual_bin = max(abs(kx_eff), abs(ky_eff)) < K_measurement_cutoff
            apply_projection = False
            if is_individual_bin:
                if decision_rule_w.get(k_mode, -1) != true_state_idx_j: apply_projection = True
            elif is_last_bin_error_inducing: apply_projection = True
            if apply_projection: F_O_psi[i_kx, i_ky] = F_psi[i_kx, i_ky]
    return np.fft.ifft2(F_O_psi)

def find_worst_case_state_multistate_sobolev(psi_nominal_surrogate, decision_rule_w, true_state_idx_j, 
                                             K_measurement_cutoff, N_grid, perturbation_strength_L2, sobolev_order_s):
    """
    Finds worst-case state by perturbing psi_nominal_surrogate.
    The perturbation direction is d_sob(n) ~ d(n) / (1+|n|^2)^s.
    The L2 norm of the added perturbation is controlled by perturbation_strength_L2.
    """
    # 1. Calculate d = O_w,j * psi_nominal_surrogate
    d_operator_action = apply_error_operator_multistate(
        psi_nominal_surrogate, decision_rule_w, true_state_idx_j, K_measurement_cutoff, N_grid
    )
    F_d = np.fft.fft2(d_operator_action) # Fourier transform of d

    # 2. Define Sobolev-scaled perturbation direction in Fourier space
    F_delta_psi_direction = np.zeros_like(F_d)
    kx_eff_all = np.fft.fftfreq(N_grid) * N_grid 
    ky_eff_all = np.fft.fftfreq(N_grid) * N_grid

    for i_kx in range(N_grid):
        for i_ky in range(N_grid):
            nx = kx_eff_all[i_kx]
            ny = ky_eff_all[i_ky]
            # Weight for H^s norm is (1+|n|^2)^s for power, (1+|n|^2)^(s/2) for amplitude
            # Perturbation direction scales as d_hat / (1+|n|^2)^s
            sobolev_scaling_factor = (1 + nx**2 + ny**2)**sobolev_order_s
            if sobolev_scaling_factor < 1e-9 : sobolev_scaling_factor = 1.0 # Avoid division by zero for s < 0, though s >=0 here
            
            F_delta_psi_direction[i_kx, i_ky] = F_d[i_kx, i_ky] / sobolev_scaling_factor

    # 3. Transform direction back to real space and normalize its L2 norm
    delta_psi_direction_real = np.fft.ifft2(F_delta_psi_direction)
    norm_delta_psi_direction_L2 = np.linalg.norm(delta_psi_direction_real)

    if norm_delta_psi_direction_L2 < 1e-9: # If scaled direction is zero
        return psi_nominal_surrogate.copy()

    # 4. Construct the perturbed state, controlling L2 norm of added part
    actual_perturbation_L2 = perturbation_strength_L2 * (delta_psi_direction_real / norm_delta_psi_direction_L2)
    perturbed_state = psi_nominal_surrogate + actual_perturbation_L2
    
    # 5. Normalize the resulting worst-case state
    norm_perturbed = np.linalg.norm(perturbed_state)
    if norm_perturbed > 1e-9:
        worst_case_psi = perturbed_state / norm_perturbed
    else: 
        worst_case_psi = psi_nominal_surrogate.copy() 
    return worst_case_psi

def run_robust_optimization_multistate_sobolev(list_psi_T_nominal_surrogate,
                                              K_measurement_cutoff, N_grid, list_priors_p_j,
                                              perturbation_strength_L2, sobolev_order_s, 
                                              max_robust_iters=10, trial_num=None, num_trials_total_for_logging=100, process_id=None):
    num_states = len(list_psi_T_nominal_surrogate)
    log_prefix = ""
    if process_id is not None: log_prefix += f"[P{process_id}]"
    if trial_num is not None: log_prefix += f"[T{trial_num}] "
    print_details_robust = trial_num is None or \
                           (trial_num is not None and trial_num % (num_trials_total_for_logging // 10 or 1) == 1 or trial_num == 1) or \
                           num_trials_total_for_logging <= 10
    if print_details_robust: print(f"{log_prefix}Sobolev Robust Opt Start (s={sobolev_order_s}, {num_states} states)")
    
    list_povm_probs_nom_surr = [compute_povm_probabilities(psi_surr, K_measurement_cutoff, N_grid) for psi_surr in list_psi_T_nominal_surrogate]
    current_w = get_optimal_decision_rule_multistate(list_povm_probs_nom_surr, list_priors_p_j)
    
    for iter_num in range(max_robust_iters):
        list_psi_T_worst = []
        for j_idx in range(num_states):
            psi_j_worst = find_worst_case_state_multistate_sobolev(
                list_psi_T_nominal_surrogate[j_idx], current_w, j_idx, 
                K_measurement_cutoff, N_grid, perturbation_strength_L2, sobolev_order_s
            )
            list_psi_T_worst.append(psi_j_worst)
        list_povm_probs_worst = [compute_povm_probabilities(psi_worst, K_measurement_cutoff, N_grid) for psi_worst in list_psi_T_worst]
        next_w = get_optimal_decision_rule_multistate(list_povm_probs_worst, list_priors_p_j)
        if current_w == next_w:
            if print_details_robust and max_robust_iters > 1: print(f"{log_prefix}  Sobolev Robust converged iter {iter_num + 1}.")
            current_w = next_w; break
        current_w = next_w
        if iter_num == max_robust_iters - 1 and print_details_robust and max_robust_iters > 1:
            print(f"{log_prefix}  Sobolev Robust reached max iterations.")
    return current_w

def run_single_experiment(config_tuple):
    config, experiment_idx, total_experiments = config_tuple 
    process_id = os.getpid() 
    print(f"[P{process_id}] Exp {experiment_idx+1}/{total_experiments}: Noise={config['noise_level']}, PerturbStr(L2)={config['perturbation_strength_L2']}, NumStates={config['num_distinct_states']}, SobOrderS={config['sobolev_order_s']}")
    num_trials_this_experiment = config['num_trials_per_experiment']
    oracle_errors, nominal_surr_errors, robust_surr_errors = [], [], []
    for i_trial in range(num_trials_this_experiment):
        current_trial_config = config.copy(); current_trial_config['num_trials_total'] = num_trials_this_experiment 
        initial_states_psi0, evolved_states_psiT_true, _, _ = setup_multistate_evolution(current_trial_config, trial_num=i_trial+1, process_id=process_id)
        K_meas_cutoff = current_trial_config['K_measurement_cutoff']; N_grid_config = current_trial_config['n_grid']
        priors_j_config = current_trial_config['prior_p_j']; sobolev_order_s_config = current_trial_config['sobolev_order_s']
        
        list_povm_probs_true = [compute_povm_probabilities(psiT_j, K_meas_cutoff, N_grid_config) for psiT_j in evolved_states_psiT_true]
        oracle_optimal_rule = get_optimal_decision_rule_multistate(list_povm_probs_true, priors_j_config)
        oracle_min_prob_error = calculate_probability_of_error_multistate(oracle_optimal_rule, list_povm_probs_true, priors_j_config)
        oracle_errors.append(oracle_min_prob_error)

        noise_lvl_base = current_trial_config['noise_level'] # This is now base noise for DC mode
        list_psi_T_nominal_surrogate = [add_spectral_noise_sobolev(psiT_j_true, noise_lvl_base, N_grid_config, sobolev_order_s_config) for psiT_j_true in evolved_states_psiT_true]
        
        list_povm_probs_nom_surr = [compute_povm_probabilities(psi_surr, K_meas_cutoff, N_grid_config) for psi_surr in list_psi_T_nominal_surrogate]
        nominal_surrogate_rule = get_optimal_decision_rule_multistate(list_povm_probs_nom_surr, priors_j_config)
        nominal_surrogate_rule_true_perf = calculate_probability_of_error_multistate(nominal_surrogate_rule, list_povm_probs_true, priors_j_config)
        nominal_surr_errors.append(nominal_surrogate_rule_true_perf)

        perturb_strength_L2_config = current_trial_config['perturbation_strength_L2']
        max_iters_robust = current_trial_config['max_robust_iters']
        robust_decision_rule = run_robust_optimization_multistate_sobolev(
            list_psi_T_nominal_surrogate, K_meas_cutoff, N_grid_config, priors_j_config,
            perturb_strength_L2_config, sobolev_order_s_config, max_iters_robust, 
            trial_num=i_trial+1, num_trials_total_for_logging=num_trials_this_experiment, process_id=process_id
        )
        robust_rule_true_perf = calculate_probability_of_error_multistate(robust_decision_rule, list_povm_probs_true, priors_j_config)
        robust_surr_errors.append(robust_rule_true_perf)
        if (i_trial + 1) % (num_trials_this_experiment // 5 or 1) == 0 :
            print(f"    [P{process_id}] Exp {experiment_idx+1}, Trial {i_trial+1}/{num_trials_this_experiment} completed.")
    avg_oracle_err = np.mean(oracle_errors); avg_nominal_err = np.mean(nominal_surr_errors); avg_robust_err = np.mean(robust_surr_errors)
    t_statistic, p_value_one_sided = -1, 1.0 
    if num_trials_this_experiment > 1:
        nom_err_arr = np.array(nominal_surr_errors); rob_err_arr = np.array(robust_surr_errors)
        valid_indices = ~np.isnan(nom_err_arr) & ~np.isnan(rob_err_arr)
        if np.sum(valid_indices) > 1:
            t_statistic, p_value_one_sided = stats.ttest_rel(nom_err_arr[valid_indices], rob_err_arr[valid_indices], alternative='greater', nan_policy='raise')
            if np.isnan(t_statistic): t_statistic = 0.0; p_value_one_sided = 1.0 if np.mean(nom_err_arr[valid_indices] - rob_err_arr[valid_indices]) != 0 else 0.5
        elif np.sum(valid_indices) <= 1 : print(f"    [P{process_id}] Exp {experiment_idx+1}: Not enough valid data points for t-test ({np.sum(valid_indices)} found).")
    print(f"[P{process_id}] Finished Exp {experiment_idx+1}/{total_experiments}")
    return {"noise_level": config['noise_level'], "perturb_strength_L2": config['perturbation_strength_L2'], 
            "num_states": config['num_distinct_states'], "sobolev_order_s": config['sobolev_order_s'],
            "avg_oracle_err": avg_oracle_err, "avg_nominal_err": avg_nominal_err, "avg_robust_err": avg_robust_err,
            "t_statistic": t_statistic, "p_value": p_value_one_sided}

if __name__ == '__main__':
    k_psi0_modes_config_sweep = 8 
    K_MEASUREMENT_CUTOFF_SWEEP = k_psi0_modes_config_sweep + 1 
    N_GRID_DEFAULT_SWEEP = 64 
    T_EVOLUTION_FIXED_SWEEP = 2.0
    CHOSEN_HAMILTONIAN_SWEEP = 'harmonic_oscillator' 
    NUM_TRIALS_PER_EXPERIMENT = 30 # Reduce for quicker testing of sweep, increase for accuracy

    noise_level_base_values = [0.5, 1.0, 1.5] 
    perturbation_strength_L2_values = [0.001, 0.005, 0.0025] 
    num_distinct_states_values = [3] 
    sobolev_order_s_values = [2] # s=0 is L2 like before, s>0 emphasizes low freq more

    print(f"--- Starting Parameter Sweep (Parallel) ---")
    print(f"Hamiltonian: {CHOSEN_HAMILTONIAN_SWEEP}, T_evolution: {T_EVOLUTION_FIXED_SWEEP}")
    print(f"Trials per experiment: {NUM_TRIALS_PER_EXPERIMENT}")

    param_combinations_for_map = []
    base_param_combinations = list(product(noise_level_base_values, perturbation_strength_L2_values, num_distinct_states_values, sobolev_order_s_values))
    total_experiments = len(base_param_combinations)

    for i_exp, (nl_base, ps_l2, n_states, s_order) in enumerate(base_param_combinations):
        current_config = {
            'n_grid': N_GRID_DEFAULT_SWEEP, 'l_domain': 2 * np.pi,
            'k_psi0_modes': k_psi0_modes_config_sweep, 't_evolution': T_EVOLUTION_FIXED_SWEEP, 
            'num_solver_steps': 50, 'hamiltonian_name': CHOSEN_HAMILTONIAN_SWEEP,
            'hamiltonian_params': {}, 'K_measurement_cutoff': K_MEASUREMENT_CUTOFF_SWEEP, 
            'num_distinct_states': n_states, 'prior_p_j': [1.0/n_states] * n_states,
            'noise_level': nl_base, # This is now noise_level_base
            'perturbation_strength_L2': ps_l2, # This is now L2 norm of Sobolev-shaped perturbation
            'sobolev_order_s': s_order, # New parameter
            'max_robust_iters': 10, 'max_initial_overlap_sq': 0.3, 'max_retries_overlap': 20, 
            'num_trials_per_experiment': NUM_TRIALS_PER_EXPERIMENT 
        }
        min_k_for_n_states = 0
        while (2 * min_k_for_n_states + 1)**2 < n_states: min_k_for_n_states += 1
        if current_config['k_psi0_modes'] < min_k_for_n_states:
            current_config['k_psi0_modes'] = min_k_for_n_states
            current_config['K_measurement_cutoff'] = min_k_for_n_states +1 
        if CHOSEN_HAMILTONIAN_SWEEP == 'barrier': current_config['hamiltonian_params'] = {'barrier_height': 100.0, 'slit_width_ratio': 0.15}
        elif CHOSEN_HAMILTONIAN_SWEEP == 'harmonic_oscillator': current_config['hamiltonian_params'] = {'omega': 5.0, 'm_potential': 1.0}
        elif CHOSEN_HAMILTONIAN_SWEEP == 'random_potential': current_config['hamiltonian_params'] = {'alpha': 0.5, 'beta': 0.2, 'gamma': 2.5}
        elif CHOSEN_HAMILTONIAN_SWEEP == 'paul_trap': current_config['hamiltonian_params'] = {'U0': 2.0, 'V0': 10.0, 'omega_trap_factor': 2.0, 'r0_sq_factor': 0.05}
        param_combinations_for_map.append((current_config, i_exp, total_experiments))

    num_processes = 32 # min(multiprocessing.cpu_count(), len(param_combinations_for_map), 4) # Further reduced cap for stability
    print(f"Using {num_processes} processes for parallel execution.")
    with multiprocessing.Pool(processes=num_processes) as pool:
        all_results = pool.map(run_single_experiment, param_combinations_for_map)
    all_results.sort(key=lambda r: (r['noise_level'], r['perturb_strength_L2'], r['num_states'], r['sobolev_order_s']))

    print("\n\n--- Parameter Sweep Results Summary (Sobolev Norm) ---")
    header = "| Noise Lvl | Perturb Str (L2) | Num States | Sob Order S | Avg Oracle Err | Avg Nominal Err | Avg Robust Err | t-statistic | p-value (rob < nom) |"
    print(header)
    print("|" + "-"*(len(header)-2) + "|")
    for res in all_results:
        print(f"| {res['noise_level']:<9.3f} | {res['perturb_strength_L2']:<16.3f} | {res['num_states']:<10d} | {res['sobolev_order_s']:<11d} | "
              f"{res['avg_oracle_err']:<14.4f} | {res['avg_nominal_err']:<15.4f} | {res['avg_robust_err']:<14.4f} | "
              f"{res['t_statistic']:<11.3f} | {res['p_value']:<21.4f} |")

    RESULTS_DIR_SWEEP = "results_parameter_sweep_sobolev" 
    os.makedirs(RESULTS_DIR_SWEEP, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR_SWEEP, f"param_sweep_results_{CHOSEN_HAMILTONIAN_SWEEP}_sobolev_parallel.csv")
    with open(csv_path, 'w') as f:
        f.write("NoiseLevelBase,PerturbationStrengthL2,NumStates,SobolevOrderS,AvgOracleErr,AvgNominalErr,AvgRobustErr,TStatistic,PValue\n")
        for res in all_results:
            f.write(f"{res['noise_level']},{res['perturb_strength_L2']},{res['num_states']},{res['sobolev_order_s']},"
                    f"{res['avg_oracle_err']},{res['avg_nominal_err']},{res['avg_robust_err']},"
                    f"{res['t_statistic']},{res['p_value']}\n")
    print(f"\nResults saved to CSV: {csv_path}")
