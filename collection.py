import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ---------- import-compatible copy of your SNN (must match model.py / calibration.py) ----------
class SimpleSpectralOperatorCNN(nn.Module):
    def __init__(self, K_input_resolution, K_output_resolution, hidden_channels=64, num_hidden_layers=3):
        super().__init__()
        self.K_input_resolution = K_input_resolution
        self.K_output_resolution = K_output_resolution
        if K_output_resolution > K_input_resolution:
            raise ValueError("K_output_resolution cannot exceed K_input_resolution.")
        layers = [nn.Conv2d(2, hidden_channels, kernel_size=3, padding='same'), nn.ReLU()]
        for _ in range(num_hidden_layers):
            layers += [nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding='same'), nn.ReLU()]
        layers += [nn.Conv2d(hidden_channels, 2, kernel_size=3, padding='same')]
        self.cnn_body = nn.Sequential(*layers)

    def forward(self, x):
        y = self.cnn_body(x)
        if self.K_input_resolution == self.K_output_resolution:
            return y
        s = self.K_input_resolution // 2 - self.K_output_resolution // 2
        e = s + self.K_output_resolution
        return y[:, :, s:e, s:e]

# ---------- utils mirroring your helpers ----------
def spectrum_complex_to_channels_torch(spectrum_mat_complex):
    if not isinstance(spectrum_mat_complex, torch.Tensor):
        spectrum_mat_complex = torch.from_numpy(spectrum_mat_complex)
    if not torch.is_complex(spectrum_mat_complex):
        if spectrum_mat_complex.ndim == 3 and spectrum_mat_complex.shape[0] == 2:
            return spectrum_mat_complex.float()
        raise ValueError("Expected complex [K,K] or real [2,K,K].")
    return torch.stack([spectrum_mat_complex.real, spectrum_mat_complex.imag], dim=0).float()

def channels_to_spectrum_complex_torch(channels_mat_real_imag):
    if channels_mat_real_imag.ndim != 3 or channels_mat_real_imag.shape[0] != 2:
        raise ValueError("channels_to_spectrum_complex_torch expects [2,K,K].")
    return torch.complex(channels_mat_real_imag[0], channels_mat_real_imag[1])

def pad_to_full_centered(block_centered, K_full):
    """Embed a centered K_out x K_out block back into a K_full x K_full centered array."""
    K_out = block_centered.shape[0]
    full = np.zeros((K_full, K_full), dtype=block_centered.dtype)
    s = K_full // 2 - K_out // 2
    e = s + K_out
    full[s:e, s:e] = block_centered
    return full

def spec_to_spatial_centered(full_centered):
    return np.fft.ifft2(np.fft.ifftshift(full_centered))

# ---------- geometry & objective ----------
def disk_mask(N, centers, radius_px):
    yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    mask = np.zeros((N, N), dtype=bool)
    for (y, x) in centers:
        mask |= (xx - x)**2 + (yy - y)**2 <= radius_px**2
    return mask

def J_collect(u_spatial_real, mask, dx=1.0):
    return float(u_spatial_real[mask].sum() * dx * dx)

# ---------- dual norm term for robust correction (H^{s-ν} dual) ----------
def sobolev_weight(N, s_minus_nu):
    k = np.fft.fftfreq(N) * N
    kx, ky = np.meshgrid(k, k, indexing='ij')
    k2 = kx**2 + ky**2
    return (1.0 + k2)**(s_minus_nu)

def fft2c(x):
    return np.fft.fftshift(np.fft.fft2(x))

def dual_norm_indicator(mask, s_minus_nu):
    N = mask.shape[0]
    Chi = mask.astype(float)
    ChiHat = fft2c(Chi)
    w = sobolev_weight(N, s_minus_nu)
    # using discrete analog of <·,·> with w(k) as Sobolev weight; divide by N^2 for FFT norming
    val = np.sum(np.abs(ChiHat)**2 / w)
    return float(np.sqrt(val) / (N * N))

# ---------- simple local search for centers ----------
def random_centers(N, K, rng):
    return [(int(rng.integers(0, N)), int(rng.integers(0, N))) for _ in range(K)]

def propose(centers, N, step_px, rng):
    new = list(centers)
    i = rng.integers(0, len(new))
    dy = int(rng.integers(-step_px, step_px + 1))
    dx = int(rng.integers(-step_px, step_px + 1))
    y, x = new[i]
    y = min(max(0, y + dy), N - 1)
    x = min(max(0, x + dx), N - 1)
    new[i] = (y, x)
    return new

# --- change optimize to accept an explicit initializer and rng ---
def optimize(u_tilde_real, K, radius_px, obj_fn, iters, step_px, seed=None, init_centers=None, rng=None, **obj_kwargs):
    if rng is None:
        rng = np.random.default_rng(seed)
    N = u_tilde_real.shape[0]
    centers = init_centers if init_centers is not None else random_centers(N, K, rng)
    best = centers
    best_val = obj_fn(u_tilde_real, best, radius_px, **obj_kwargs)
    for _ in range(iters):
        cand = propose(best, N, step_px, rng)
        val = obj_fn(u_tilde_real, cand, radius_px, **obj_kwargs)
        if val > best_val:
            best, best_val = cand, val
    return best, best_val

# objectives
def nominal_obj(u_tilde_real, centers, radius_px, **_):
    mask = disk_mask(u_tilde_real.shape[0], centers, radius_px)
    return J_collect(u_tilde_real, mask)

def robust_obj(u_tilde_real, centers, radius_px, r_radius, s_minus_nu, **_):
    mask = disk_mask(u_tilde_real.shape[0], centers, radius_px)
    nominal = J_collect(u_tilde_real, mask)
    dual = dual_norm_indicator(mask, s_minus_nu)
    return nominal + r_radius * dual

def _draw_disks(ax, centers, radius_px, edgecolor, label):
    for (y, x) in centers:
        circ = Circle((x, y), radius_px, fill=False, lw=2.0, ec=edgecolor, alpha=0.95)
        ax.add_patch(circ)
        ax.plot([x], [y], marker='o', ms=4, mec='k', mfc=edgecolor, lw=0.5, label=label)
        label = None  # only label first

def visualize_layouts(u_bg, w_nom, w_rob, radius_px, title, out_path, vmin=None, vmax=None):
    """
    u_bg: 2D np.ndarray (spatial domain)
    w_nom, w_rob: list[(y,x)]
    """
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 5.6), dpi=150)
    im = ax.imshow(u_bg, origin='upper', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='intensity')

    _draw_disks(ax, w_nom, radius_px, edgecolor='crimson', label='nominal')
    _draw_disks(ax, w_rob, radius_px, edgecolor='deepskyblue', label='robust')

    ax.set_title(title)
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')
    # build legend from the two first points we plotted
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # keep only unique labels
        seen = set(); H=[]; L=[]
        for h,l in zip(handles, labels):
            if l and l not in seen:
                seen.add(l); H.append(h); L.append(l)
        ax.legend(H, L, loc='upper right', frameon=True)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

def main():
    import argparse, os, json
    import numpy as np
    import torch
    from math import comb, sqrt

    p = argparse.ArgumentParser(description="Facet-1: Robust vs Nominal Collection under Misspecification + stats")
    # match your naming
    p.add_argument('--pde_type', type=str, default='poisson',
                   choices=['poisson','heat_equation','step_index_fiber','grin_fiber'])
    p.add_argument('--n_grid_sim_input_ds', type=int, default=64)  # Nin / full grid
    p.add_argument('--snn_output_res', type=int, default=32)       # Nout used to train
    p.add_argument('--dataset_dir', type=str, default='datasets')
    p.add_argument('--model_dir', type=str, default='trained_snn_models')
    p.add_argument('--calib_results_dir', type=str, default='results_conformal_theorem_validation')
    p.add_argument('--calib_npz_override', type=str, default=None, help='Optional direct path to coverage_data_*.npz')

    # filename suffix params (must match your runs)
    p.add_argument('--grf_alpha', type=float, default=4.0)
    p.add_argument('--grf_tau', type=float, default=1.0)
    p.add_argument('--grf_offset_sigma', type=float, default=0.5)
    p.add_argument('--L_domain', type=float, default=2*np.pi)
    p.add_argument('--fiber_core_radius_factor', type=float, default=0.2)
    p.add_argument('--fiber_potential_depth', type=float, default=0.5)
    p.add_argument('--grin_strength', type=float, default=0.01)
    p.add_argument('--viscosity_nu', type=float, default=0.01)
    p.add_argument('--evolution_time_T', type=float, default=0.1)
    p.add_argument('--solver_num_steps', type=int, default=50)

    # SNN arch
    p.add_argument('--snn_hidden_channels', type=int, default=64)
    p.add_argument('--snn_num_hidden_layers', type=int, default=3)

    # robust params (from paper defaults: s=2, nu=2 -> s-ν = 0)
    p.add_argument('--s_theorem', type=float, default=2.0)
    p.add_argument('--nu_theorem', type=float, default=2.0)
    p.add_argument('--alpha_for_radius', type=float, default=0.10, help='Pick q for this alpha')

    # collection settings
    p.add_argument('--K_facilities', type=int, default=3)
    p.add_argument('--radius_px', type=int, default=6)
    p.add_argument('--iters', type=int, default=800)
    p.add_argument('--step_px', type=int, default=3)
    p.add_argument('--trials', type=int, default=30)
    p.add_argument('--seed', type=int, default=0)

    # bootstrap (optional CI)
    p.add_argument('--bootstrap_samples', type=int, default=0, help='If >0, compute bootstrap CI for mean diff (e.g., 5000)')

    p.add_argument('--viz_trials', type=int, default=6, help='How many trials to save visualizations for')
    p.add_argument('--viz_dir', type=str, default=None, help='Directory to save visualizations (defaults under results_out)')
    p.add_argument('--viz_show_true', action='store_true', help='Also render overlays on TRUE field')
    p.add_argument('--viz_vmax', type=float, default=None, help='Optional vmax for imshow')

    # output
    p.add_argument('--results_out', type=str, default='results_facet1_collection')
    args = p.parse_args()

    # ----- build suffix exactly like your scripts -----
    def calib_suffix(a):
        if a.pde_type == "poisson":
            return f"poisson_grfA{a.grf_alpha:.1f}T{a.grf_tau:.1f}OffS{a.grf_offset_sigma:.1f}"
        elif a.pde_type == "step_index_fiber":
            return (f"fiber_GRFinA{a.grf_alpha:.1f}T{a.grf_tau:.1f}_"
                    f"coreR{a.fiber_core_radius_factor:.1f}_V{a.fiber_potential_depth:.1f}_"
                    f"evoT{a.evolution_time_T:.1e}_steps{a.solver_num_steps}")
        elif a.pde_type == "grin_fiber":
            return (f"grinfiber_GRFinA{a.grf_alpha:.1f}T{a.grf_tau:.1f}_"
                    f"strength{a.grin_strength:.2e}_"
                    f"evoT{a.evolution_time_T:.1e}_steps{a.solver_num_steps}")
        elif a.pde_type == "heat_equation":
            return (f"heat_GRFinA{a.grf_alpha:.1f}T{a.grf_tau:.1f}_"
                    f"nu{a.viscosity_nu:.2e}_evoT{a.evolution_time_T:.1e}")

    suffix = calib_suffix(args)

    # dataset/model paths (same as your conventions)
    dataset_file = os.path.join(
        args.dataset_dir,
        f"dataset_{args.pde_type}_Nin{args.n_grid_sim_input_ds}_Nout{args.snn_output_res}_{suffix}.npz"
    )
    model_file = os.path.join(
        args.model_dir,
        f"snn_PDE{args.pde_type}_Kin{args.n_grid_sim_input_ds}_Kout{args.snn_output_res}_H{args.snn_hidden_channels}_L{args.snn_num_hidden_layers}_{suffix}.pth"
    )

    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset not found: {dataset_file}")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model not found: {model_file}")

    os.makedirs(args.results_out, exist_ok=True)

    # ----- load data -----
    data = np.load(dataset_file)
    # use the calib split for “holdout” in this experiment (consistent with calibration.py)
    Gb = data['gamma_b_calib']                      # (M, Nin, Nin) complex
    Ga_true_full = data['gamma_a_true_full_calib']  # (M, Nin, Nin) complex
    M = Gb.shape[0]
    N = args.n_grid_sim_input_ds

    # ----- load model -----
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    snn = SimpleSpectralOperatorCNN(args.n_grid_sim_input_ds, args.snn_output_res,
                                    hidden_channels=args.snn_hidden_channels,
                                    num_hidden_layers=args.snn_num_hidden_layers)
    snn.load_state_dict(torch.load(model_file, map_location=DEVICE))
    snn.to(DEVICE)
    snn.eval()

    # ----- get r = sqrt(q) for chosen alpha (deterministic paths from your scripts) -----
    # subdirectory (from calibration_sweep.py)
    calib_subdir = (
        f"PDE{args.pde_type}_NinDS{args.n_grid_sim_input_ds}_SNNres{args.snn_output_res}_"
        f"KfullThm{args.n_grid_sim_input_ds}_s{args.s_theorem}_nu{args.nu_theorem}_{suffix}"
    )
    # filename (from calibration.py)
    calib_filename = (
        f"coverage_data_PDE{args.pde_type}_thm_s{args.s_theorem}_nu{args.nu_theorem}_d2"
        f"_Nin{args.n_grid_sim_input_ds}_SNNout{args.snn_output_res}_NfullThm{args.n_grid_sim_input_ds}"
        f"_{suffix}.npz"
    )
    calib_path_1 = os.path.join(args.calib_results_dir, calib_subdir, calib_filename)
    calib_path_2 = os.path.join(args.calib_results_dir, calib_filename)

    if args.calib_npz_override and os.path.exists(args.calib_npz_override):
        calib_npz_path = args.calib_npz_override
    elif os.path.exists(calib_path_1):
        calib_npz_path = calib_path_1
    elif os.path.exists(calib_path_2):
        calib_npz_path = calib_path_2
    else:
        calib_npz_path = None

    if calib_npz_path:
        c = np.load(calib_npz_path)
        nominal_cov = c['nominal_coverages']   # = 1 - alphas used in calibration.py
        alphas_used = 1.0 - nominal_cov
        q_arr = c['quantiles_q_hat_nu']
        idx = int(np.argmin(np.abs(alphas_used - args.alpha_for_radius)))
        q = float(q_arr[idx])
        r_radius = float(np.sqrt(max(q, 0.0)))
        print(f"[calibration] loaded: {calib_npz_path}")
        print(f"[calibration] alpha≈{alphas_used[idx]:.2f} -> q={q:.3e} -> r={r_radius:.3e}")
    else:
        print("WARNING: calibration npz not found at either path; using fallback r=0.05.\n"
              f"  tried:\n    {calib_path_1}\n    {calib_path_2}")
        r_radius = 0.05

    s_minus_nu = float(args.s_theorem - args.nu_theorem)  # = 0 with defaults

    # ---------- optimization helpers ----------
    def disk_mask(N, centers, radius_px):
        yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        mask = np.zeros((N, N), dtype=bool)
        for (y, x) in centers:
            mask |= (xx - x)**2 + (yy - y)**2 <= radius_px**2
        return mask

    def J_collect(u_spatial_real, mask, dx=1.0):
        return float(u_spatial_real[mask].sum() * dx * dx)

    def sobolev_weight(N, s_minus_nu):
        k = np.fft.fftfreq(N) * N
        kx, ky = np.meshgrid(k, k, indexing='ij')
        k2 = kx**2 + ky**2
        return (1.0 + k2)**(s_minus_nu)

    def fft2c(x):
        return np.fft.fftshift(np.fft.fft2(x))

    def dual_norm_indicator(mask, s_minus_nu):
        Nloc = mask.shape[0]
        Chi = mask.astype(float)
        ChiHat = fft2c(Chi)
        w = sobolev_weight(Nloc, s_minus_nu)
        val = np.sum(np.abs(ChiHat)**2 / w)
        return float(np.sqrt(val) / (Nloc * Nloc))

    def nominal_obj(u_tilde_real, centers, radius_px, **_):
        mask = disk_mask(u_tilde_real.shape[0], centers, radius_px)
        return J_collect(u_tilde_real, mask)

    def robust_obj(u_tilde_real, centers, radius_px, r_radius, s_minus_nu, **_):
        mask = disk_mask(u_tilde_real.shape[0], centers, radius_px)
        nominal = J_collect(u_tilde_real, mask)
        dual = dual_norm_indicator(mask, s_minus_nu)
        return nominal + r_radius * dual

    def random_centers(N, K, rng):
        return [(int(rng.integers(0, N)), int(rng.integers(0, N))) for _ in range(K)]

    def propose(centers, N, step_px, rng):
        new = list(centers)
        i = rng.integers(0, len(new))
        dy = int(rng.integers(-step_px, step_px + 1))
        dx = int(rng.integers(-step_px, step_px + 1))
        y, x = new[i]
        y = min(max(0, y + dy), N - 1)
        x = min(max(0, x + dx), N - 1)
        new[i] = (y, x)
        return new

    def optimize(u_tilde_real, K, radius_px, obj_fn, iters, step_px, seed=None, init_centers=None, rng=None, **obj_kwargs):
        if rng is None:
            rng = np.random.default_rng(seed)
        Nloc = u_tilde_real.shape[0]
        centers = init_centers if init_centers is not None else random_centers(Nloc, K, rng)
        best = centers
        best_val = obj_fn(u_tilde_real, best, radius_px, **obj_kwargs)
        for _ in range(iters):
            cand = propose(best, Nloc, step_px, rng)
            val = obj_fn(u_tilde_real, cand, radius_px, **obj_kwargs)
            if val > best_val:
                best, best_val = cand, val
        return best, best_val

    # ---------- spectrum <-> spatial helpers ----------
    def spectrum_complex_to_channels_torch(spectrum_mat_complex):
        if not isinstance(spectrum_mat_complex, torch.Tensor):
            spectrum_mat_complex = torch.from_numpy(spectrum_mat_complex)
        if not torch.is_complex(spectrum_mat_complex):
            if spectrum_mat_complex.ndim == 3 and spectrum_mat_complex.shape[0] == 2:
                return spectrum_mat_complex.float()
            raise ValueError("Expected complex [K,K] or real [2,K,K].")
        return torch.stack([spectrum_mat_complex.real, spectrum_mat_complex.imag], dim=0).float()

    def channels_to_spectrum_complex_torch(channels_mat_real_imag):
        if channels_mat_real_imag.ndim != 3 or channels_mat_real_imag.shape[0] != 2:
            raise ValueError("channels_to_spectrum_complex_torch expects [2,K,K].")
        return torch.complex(channels_mat_real_imag[0], channels_mat_real_imag[1])

    def pad_to_full_centered(block_centered, K_full):
        K_out = block_centered.shape[0]
        full = np.zeros((K_full, K_full), dtype=block_centered.dtype)
        s = K_full // 2 - K_out // 2
        e = s + K_out
        full[s:e, s:e] = block_centered
        return full

    def spec_to_spatial_centered(full_centered):
        return np.fft.ifft2(np.fft.ifftshift(full_centered))

    # ----- per-trial loop -----
    results = []
    rng_master = np.random.default_rng(args.seed)
    idxs = rng_master.choice(M, size=min(args.trials, M), replace=False)

    for t, i in enumerate(idxs):
        # predict truncated spectrum on full input
        gb = Gb[i]  # complex (N,N)
        with torch.no_grad():
            x = spectrum_complex_to_channels_torch(gb).unsqueeze(0).to(DEVICE)  # [1,2,N,N]
            y = snn(x)  # [1,2,Nout,Nout]
            y = y.squeeze(0).cpu()
            y_complex = channels_to_spectrum_complex_torch(y).numpy()  # (Nout,Nout) complex

        # pad to full centered spectrum, inverse FFT -> predicted spatial field
        y_full_centered = pad_to_full_centered(y_complex, N)
        u_tilde = spec_to_spatial_centered(y_full_centered)
        # pick scalar field for objective
        if args.pde_type in ['poisson', 'heat_equation']:
            u_tilde_real = u_tilde.real
        else:
            u_tilde_real = np.abs(u_tilde)

        # true field
        ga_full_centered = Ga_true_full[i]  # (N,N) complex
        u_true = spec_to_spatial_centered(ga_full_centered)
        if args.pde_type in ['poisson', 'heat_equation']:
            u_true_real = u_true.real
        else:
            u_true_real = np.abs(u_true)

        # same RNG and same init centers for both nominal & robust to ensure comparability
        trial_seed = int(rng_master.integers(0, 10_000_000))

        # 1) draw shared initial centers with a dedicated RNG
        rng_init = np.random.default_rng(trial_seed)
        init_centers = random_centers(N, args.K_facilities, rng_init)

        # 2) make two *independent* RNGs with the same seed for identical proposal streams
        rng_nom = np.random.default_rng(trial_seed + 1)
        rng_rob = np.random.default_rng(trial_seed + 1)

        # NOMINAL
        w_nom, _ = optimize(
            u_tilde_real, args.K_facilities, args.radius_px,
            nominal_obj, iters=args.iters, step_px=args.step_px,
            rng=rng_nom, init_centers=init_centers
        )

        # ROBUST (r_radius = 0.0 makes this identical to nominal)
        w_rob, _ = optimize(
            u_tilde_real, args.K_facilities, args.radius_px,
            robust_obj, iters=args.iters, step_px=args.step_px,
            rng=rng_rob, init_centers=init_centers,
            r_radius=0.0, s_minus_nu=s_minus_nu
        )

        # evaluate both on TRUE field
        Jn = J_collect(u_true_real, disk_mask(N, w_nom, args.radius_px))
        Jr = J_collect(u_true_real, disk_mask(N, w_rob, args.radius_px))
        results.append((Jn, Jr))

        # ----- visualization -----
        # decide where to save
        viz_root = args.viz_dir or os.path.join(args.results_out, "viz")
        if t < max(0, int(args.viz_trials)):
            # Use same color scale per trial (optional): clip at 99th percentile for contrast if vmax not provided
            if args.viz_vmax is not None:
                vmax = args.viz_vmax
            else:
                # robust to outliers, good contrast for GRFs and fibers
                vmax = float(np.percentile(np.abs(u_tilde_real), 99.5)) if args.pde_type not in ['poisson','heat_equation'] \
                    else float(np.percentile(u_tilde_real, 99.5))
            vmin = None  # let matplotlib auto-scale floor

            # predicted background
            out_pred = os.path.join(
                viz_root,
                f"trial{t:03d}_pred_PDE{args.pde_type}_N{N}_Nout{args.snn_output_res}_K{args.K_facilities}_rpx{args.radius_px}.png"
            )
            visualize_layouts(
                u_bg=u_tilde_real,
                w_nom=w_nom,
                w_rob=w_rob,
                radius_px=args.radius_px,
                title=f"Predicted field (SNN) — nominal (red) vs robust (blue)",
                out_path=out_pred,
                vmin=vmin, vmax=vmax
            )

            if args.viz_show_true:
                out_true = os.path.join(
                    viz_root,
                    f"trial{t:03d}_true_PDE{args.pde_type}_N{N}_Nout{args.snn_output_res}_K{args.K_facilities}_rpx{args.radius_px}.png"
                )
                # reuse vmax heuristic but on true field for fair visual contrast
                if args.viz_vmax is None:
                    vmax_true = float(np.percentile(np.abs(u_true_real), 99.5)) if args.pde_type not in ['poisson','heat_equation'] \
                                else float(np.percentile(u_true_real, 99.5))
                else:
                    vmax_true = args.viz_vmax

                visualize_layouts(
                    u_bg=u_true_real,
                    w_nom=w_nom,
                    w_rob=w_rob,
                    radius_px=args.radius_px,
                    title=f"TRUE field — nominal (red) vs robust (blue)",
                    out_path=out_true,
                    vmin=None, vmax=vmax_true
                )

    arr = np.array(results)
    diffs = arr[:, 1] - arr[:, 0]
    mean_nom = float(arr[:, 0].mean())
    mean_rob = float(arr[:, 1].mean())
    mean_diff = float(diffs.mean())
    std_diff = float(diffs.std(ddof=1))
    n = len(diffs)
    se_diff = std_diff / sqrt(max(n, 1))
    cohen_dz = mean_diff / std_diff if std_diff > 0 else float('inf')

    # ---------- statistical tests ----------
    # 1) paired t-test (scipy if available)
    t_stat, p_ttest = None, None
    try:
        import scipy.stats as st
        t_stat, p_ttest = st.ttest_rel(arr[:,1], arr[:,0], alternative='greater')
    except Exception:
        # manual t stat (no exact p-value without scipy); we’ll leave p_ttest=None if scipy isn’t available
        if std_diff > 0:
            t_stat = mean_diff / se_diff

    # 2) exact two-sided SIGN TEST (no dependencies)
    # count nonzero diffs and positives
    nz = np.where(np.abs(diffs) > 0)[0]
    n_eff = len(nz)
    k_pos = int(np.sum(diffs[nz] > 0))
    # exact two-sided p under Bin(n_eff, 0.5)
    def binom_cdf(k, n):
        return sum(comb(n, i) for i in range(0, k+1)) / (2**n) if n > 0 else 1.0
    if n_eff > 0:
        cdf = binom_cdf(k_pos, n_eff)
        sf = 1.0 - binom_cdf(k_pos-1, n_eff) if k_pos > 0 else 1.0
        p_sign_two_sided = 2 * min(cdf, sf)
        p_sign_two_sided = min(1.0, p_sign_two_sided)
    else:
        p_sign_two_sided = 1.0

    # 3) optional bootstrap CI for mean diff
    ci_lo, ci_hi = None, None
    if args.bootstrap_samples and args.bootstrap_samples > 0:
        rng_boot = np.random.default_rng(args.seed + 1337)
        boots = []
        for _ in range(args.bootstrap_samples):
            idx = rng_boot.integers(0, n, size=n)
            boots.append(diffs[idx].mean())
        boots = np.array(boots)
        ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    # ---------- print summary ----------
    print("\n=== Facet-1: Robust vs Nominal (TRUE-field evaluation) ===")
    print(f"PDE: {args.pde_type} | N={N} | Nout={args.snn_output_res} | trials={n}")
    print(f"alpha (radius): {args.alpha_for_radius:.2f}  ->  r = {r_radius:.4g}  |  s-ν = {s_minus_nu:.2f}")
    print(f"K={args.K_facilities}, radius_px={args.radius_px}, iters={args.iters}, step={args.step_px}")
    print(f"Nominal mean J_true: {mean_nom:.6f}")
    print(f"Robust  mean J_true: {mean_rob:.6f}")
    print(f"Mean (Rob - Nom):   {mean_diff:.6f}  ± {se_diff:.6f} (s.e.)")
    if ci_lo is not None:
        print(f"Bootstrap 95% CI for mean diff: [{ci_lo:.6f}, {ci_hi:.6f}]")
    print(f"Cohen's d_z (paired): {cohen_dz:.3f}")
    if t_stat is not None:
        print(f"Paired t-test (greater): t={t_stat:.3f}" + (f", p={p_ttest:.4g}" if p_ttest is not None else " (scipy not available)"))
    print(f"Sign test (two-sided): n_eff={n_eff}, k_pos={k_pos}, p={p_sign_two_sided:.4g}")

    # ---------- save outputs ----------
    out_base = os.path.join(
        args.results_out,
        f"facet1_results_PDE{args.pde_type}_N{N}_Nout{args.snn_output_res}_K{args.K_facilities}_rpx{args.radius_px}_alpha{args.alpha_for_radius:.2f}_{suffix}"
    )
    np.savez_compressed(out_base + ".npz",
                        per_trial=np.array(results),
                        diffs=diffs,
                        summary=np.array([mean_nom, mean_rob, mean_diff, std_diff, n]),
                        r_radius=r_radius,
                        s_minus_nu=s_minus_nu)
    stats_payload = {
        "mean_nom": mean_nom,
        "mean_rob": mean_rob,
        "mean_diff": mean_diff,
        "se_diff": se_diff,
        "std_diff": std_diff,
        "n": n,
        "cohen_dz": cohen_dz,
        "t_stat": t_stat,
        "p_ttest_greater": float(p_ttest) if p_ttest is not None else None,
        "sign_test_two_sided_p": p_sign_two_sided,
        "bootstrap_ci_95": [ci_lo, ci_hi] if ci_lo is not None else None,
        "alpha_for_radius": args.alpha_for_radius,
        "r_radius": r_radius
    }
    with open(out_base + "_stats.json", "w") as f:
        json.dump(stats_payload, f, indent=2)
    print(f"Saved per-trial, diffs, and stats to:\n  {out_base+'.npz'}\n  {out_base+'_stats.json'}")

if __name__ == "__main__":
    main()
