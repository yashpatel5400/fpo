import os
import argparse
import subprocess
import multiprocessing
from itertools import product
import json
import numpy as np

# ------------------------- subprocess helper (GPU-aware) -------------------------
def run_script(script_name, args_list, log_prefix="", env_extra=None):
    """Run a python script with arguments; optionally set env vars (e.g., CUDA_VISIBLE_DEVICES)."""
    cmd = ["python", script_name] + [str(a) for a in args_list]
    print(f"{log_prefix}Executing: {' '.join(cmd)}")
    env = os.environ.copy()
    if env_extra:
        env.update({k: str(v) for k, v in env_extra.items()})
    try:
        p = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=10800, env=env)
        if p.stderr:
            low = p.stderr.lower()
            if "traceback" in low or "error" in low:
                print(f"{log_prefix}stderr (possible issues):\n{p.stderr[:1500]}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{log_prefix}ERROR ({script_name}) rc={e.returncode}")
        print(f"STDOUT:\n{e.stdout[:2000]}")
        print(f"STDERR:\n{e.stderr[:2000]}")
        return False
    except subprocess.TimeoutExpired as e:
        print(f"{log_prefix}TIMEOUT after {e.timeout}s")
        return False
    except Exception as ex:
        print(f"{log_prefix}Unexpected error: {ex}")
        return False

# ------------------------- naming helpers to reconstruct files -------------------------
def build_suffix(pde_type, a, grf_alpha):
    if pde_type == "poisson":
        return f"poisson_grfA{grf_alpha:.1f}T{a.grf_tau:.1f}OffS{a.grf_offset_sigma:.1f}"
    elif pde_type == "step_index_fiber":
        return (f"fiber_GRFinA{grf_alpha:.1f}T{a.grf_tau:.1f}_"
                f"coreR{a.fiber_core_radius_factor:.1f}_V{a.fiber_potential_depth:.1f}_"
                f"evoT{a.evolution_time_T:.1e}_steps{a.solver_num_steps}")
    elif pde_type == "grin_fiber":
        return (f"grinfiber_GRFinA{grf_alpha:.1f}T{a.grf_tau:.1f}_"
                f"strength{a.grin_strength:.2e}_"
                f"evoT{a.evolution_time_T:.1e}_steps{a.solver_num_steps}")
    elif pde_type == "heat_equation":
        return (f"heat_GRFinA{grf_alpha:.1f}T{a.grf_tau:.1f}_"
                f"nu{a.viscosity_nu:.2e}_evoT{a.evolution_time_T:.1e}")
    else:
        return ""

def expected_base(results_out, pde_type, Nin, Nout, K, rpx, alpha_for_radius, suffix):
    return os.path.join(
        results_out,
        f"facet1_results_PDE{pde_type}_N{Nin}_Nout{Nout}_K{K}_rpx{rpx}_alpha{alpha_for_radius:.2f}_{suffix}"
    )

# ------------------------- stats helpers -------------------------
def paired_ttest_greater(arr_nom, arr_rob):
    try:
        import scipy.stats as st
        t, p = st.ttest_rel(arr_rob, arr_nom, alternative='greater')
        return float(t), float(p)
    except Exception:
        diffs = arr_rob - arr_nom
        n = len(diffs)
        if n == 0:
            return None, None
        m = float(diffs.mean())
        s = float(diffs.std(ddof=1)) if n > 1 else 0.0
        se = s / np.sqrt(n) if n > 1 else 0.0
        t = m / se if se > 0 else None
        return t, None  # no exact p without scipy

def sign_test_two_sided(arr_nom, arr_rob):
    diffs = arr_rob - arr_nom
    nz = np.where(np.abs(diffs) > 0)[0]
    n_eff = int(len(nz))
    k_pos = int(np.sum(diffs[nz] > 0))
    if n_eff == 0:
        return 1.0
    # exact binomial two-sided p-value
    from math import comb
    def binom_cdf(k, n):
        return sum(comb(n, i) for i in range(0, k+1)) / (2**n)
    cdf = binom_cdf(k_pos, n_eff)
    sf = 1.0 - binom_cdf(k_pos-1, n_eff) if k_pos > 0 else 1.0
    p2 = 2 * min(cdf, sf)
    return float(min(1.0, p2))

# ------------------------- LaTeX table -------------------------
def make_latex_table(results_map, grf_alphas, Nouts, caption, label,
                     value_fmt="Δ={delta:.3f} (p={p:.3g})", use="ttest", bold_sig=0.05):
    header = " & ".join([f"$N_{{out}}={n}$" for n in Nouts])
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{{l|" + f"{'c'*len(Nouts)}" + r"}}",
        r"\toprule",
        rf"$\rho$ & {header} \\ \midrule"
    ]
    for alpha in grf_alphas:
        cells = []
        for nout in Nouts:
            st = results_map.get((alpha, nout))
            if not st:
                cells.append(r"\textemdash")
                continue
            delta = st["mean_diff"]
            pval = st.get("p_ttest_greater") if use == "ttest" else st.get("sign_test_two_sided_p")
            try:
                cell = value_fmt.format(delta=delta, p=pval if pval is not None else float("nan"))
            except Exception:
                cell = f"{delta:.3f}"
            if (pval is not None) and (pval < bold_sig):
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(f"{alpha:.2f} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)

# ------------------------- per-shard worker -------------------------
def worker_run_shard(params):
    (pde_type, k_snn_out, grf_alpha, shard_idx, shard_trials, base_seed,
     args_ns, gpu_id) = params
    a = argparse.Namespace(**vars(args_ns))
    a.grf_alpha = grf_alpha

    env_gpu = {"CUDA_VISIBLE_DEVICES": gpu_id}
    log_prefix = f"[GPU {gpu_id}] [Shard {shard_idx} ρ={grf_alpha:.2f} Nout={k_snn_out}] "

    shard_seed = int(base_seed + 100 * shard_idx)
    combo_tag = f"rho{grf_alpha:.2f}_nout{k_snn_out}"
    shard_out_dir = os.path.join(a.results_out, "shards", combo_tag, f"shard_{shard_idx}")
    os.makedirs(shard_out_dir, exist_ok=True)

    # ---------------- NEW: skip completed shard if stats already exist ----------------
    suffix = build_suffix(pde_type, a, grf_alpha)
    base = expected_base(shard_out_dir, pde_type, a.n_grid_sim_input_ds, k_snn_out,
                         a.K_facilities, a.radius_px, a.alpha_for_radius, suffix)
    stats_json_path = base + "_stats.json"
    if os.path.exists(stats_json_path):
        print(f"{log_prefix}Skipping shard: results already exist at {stats_json_path}")
        return base + ".npz" if os.path.exists(base + ".npz") else None
    # -------------------------------------------------------------------------------

    cmd = [
        "--pde_type", pde_type,
        "--n_grid_sim_input_ds", a.n_grid_sim_input_ds,
        "--snn_output_res", k_snn_out,
        "--dataset_dir", a.dataset_dir,
        "--model_dir", a.model_dir,
        "--calib_results_dir", a.calib_results_dir,
        "--results_out", shard_out_dir,
        "--trials", shard_trials,
        "--iters", a.iters,
        "--K_facilities", a.K_facilities,
        "--radius_px", a.radius_px,
        "--step_px", a.step_px,
        "--seed", shard_seed,
        "--alpha_for_radius", a.alpha_for_radius,
        "--s_theorem", a.s_theorem,
        "--nu_theorem", a.nu_theorem,
        "--snn_hidden_channels", a.snn_hidden_channels,
        "--snn_num_hidden_layers", a.snn_num_hidden_layers,
        "--grf_alpha", grf_alpha,
        "--grf_tau", a.grf_tau,
        "--grf_offset_sigma", a.grf_offset_sigma,
        "--L_domain", a.L_domain,
        "--fiber_core_radius_factor", a.fiber_core_radius_factor,
        "--fiber_potential_depth", a.fiber_potential_depth,
        "--grin_strength", a.grin_strength,
        "--viscosity_nu", a.viscosity_nu,
        "--evolution_time_T", a.evolution_time_T,
        "--solver_num_steps", a.solver_num_steps,
    ]

    ok = run_script(a.collection_script, cmd, log_prefix=log_prefix, env_extra=env_gpu)
    if not ok:
        return None

    shard_npz = base + ".npz"
    if not os.path.exists(shard_npz):
        print(f"{log_prefix}ERROR: Shard npz not found: {shard_npz}")
        return None
    return shard_npz

# ------------------------- aggregate shards for a config -------------------------
def aggregate_config_from_shards(shard_npz_paths):
    """Load all shard .npz files, concatenate per_trial, compute stats dict."""
    per_trials = []
    for p in shard_npz_paths:
        try:
            z = np.load(p)
            if "per_trial" in z:
                per_trials.append(z["per_trial"])
        except Exception as e:
            print(f"Warning: failed to read {p}: {e}")
    if not per_trials:
        return None
    arr = np.vstack(per_trials)  # shape (T, 2) : (Jn, Jr)
    Jn = arr[:, 0]
    Jr = arr[:, 1]
    diffs = Jr - Jn
    mean_nom = float(Jn.mean())
    mean_rob = float(Jr.mean())
    mean_diff = float(diffs.mean())
    std_diff = float(diffs.std(ddof=1)) if diffs.size > 1 else 0.0
    n = int(diffs.size)
    se_diff = std_diff / np.sqrt(n) if n > 1 else 0.0
    dz = mean_diff / std_diff if std_diff > 0 else float("inf")

    t_stat, p_t = paired_ttest_greater(Jn, Jr)
    p_sign = sign_test_two_sided(Jn, Jr)
    return {
        "mean_nom": mean_nom,
        "mean_rob": mean_rob,
        "mean_diff": mean_diff,
        "se_diff": se_diff,
        "std_diff": std_diff,
        "n": n,
        "cohen_dz": dz,
        "t_stat": t_stat,
        "p_ttest_greater": p_t,
        "sign_test_two_sided_p": p_sign
    }

# ------------------------- main -------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GPU-parallel sweep for collection experiment with intra-config sharding + LaTeX table.")
    # sweep axes
    ap.add_argument('--pde_type', type=str, default="poisson",
                    choices=["poisson","step_index_fiber","grin_fiber","heat_equation"])
    ap.add_argument('--k_snn_output_res_values', nargs='+', type=int, default=[32, 48, 64])
    ap.add_argument('--grf_alpha_values', nargs='+', type=float, default=[2.5, 3.0, 3.5, 4.0])
    ap.add_argument('--n_grid_sim_input_ds', type=int, default=64)

    # experiment knobs (match collection_experiment.py)
    ap.add_argument('--dataset_dir', type=str, default='datasets')
    ap.add_argument('--model_dir', type=str, default='trained_snn_models')
    ap.add_argument('--calib_results_dir', type=str, default='results_conformal_theorem_validation')
    ap.add_argument('--results_out', type=str, default='results_facet1_collection_sweep')
    ap.add_argument('--collection_script', type=str, default='collection.py')

    ap.add_argument('--trials', type=int, default=200, help='Total trials per configuration (will be sharded)')
    ap.add_argument('--iters', type=int, default=800)
    ap.add_argument('--K_facilities', type=int, default=3)
    ap.add_argument('--radius_px', type=int, default=6)
    ap.add_argument('--step_px', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)

    ap.add_argument('--alpha_for_radius', type=float, default=0.10)
    ap.add_argument('--s_theorem', type=float, default=2.0)
    ap.add_argument('--nu_theorem', type=float, default=2.0)

    ap.add_argument('--snn_hidden_channels', type=int, default=64)
    ap.add_argument('--snn_num_hidden_layers', type=int, default=3)

    # suffix parameters for filename consistency
    ap.add_argument('--grf_tau', type=float, default=1.0)
    ap.add_argument('--grf_offset_sigma', type=float, default=0.5)
    ap.add_argument('--L_domain', type=float, default=2*np.pi)
    ap.add_argument('--fiber_core_radius_factor', type=float, default=0.2)
    ap.add_argument('--fiber_potential_depth', type=float, default=0.5)
    ap.add_argument('--grin_strength', type=float, default=0.01)
    ap.add_argument('--viscosity_nu', type=float, default=0.01)
    ap.add_argument('--evolution_time_T', type=float, default=0.1)
    ap.add_argument('--solver_num_steps', type=int, default=50)

    # GPU controls
    ap.add_argument('--num_gpus', type=int, default=8, help='If --gpu_ids not provided, use range(num_gpus).')
    ap.add_argument('--gpu_ids', nargs='+', type=str, default=None,
                    help='Explicit GPU ids, e.g. 0 1 2 3 4 5 6 7.')

    # intra-config parallelism
    ap.add_argument('--num_workers_per_job', type=int, default=4,
                    help='How many shards per configuration (>=1).')
    ap.add_argument('--max_concurrent', type=int, default=None,
                    help='Optional cap on total concurrent subprocesses. Defaults to num_gpus * num_workers_per_job.')

    # table options
    ap.add_argument('--use_test', type=str, default='ttest', choices=['ttest','sign'])
    ap.add_argument('--value_fmt', type=str, default="Δ={delta:.3f} (p={p:.3g})")
    ap.add_argument('--caption', type=str, default="Robust vs nominal collection improvement across discretizations and GRF smoothness.")
    ap.add_argument('--label', type=str, default="tab:collection_sweep")
    ap.add_argument('--latex_out', type=str, default='results_collection_sweep_table.tex')

    args = ap.parse_args()

    os.makedirs(args.results_out, exist_ok=True)

    # GPU list
    if args.gpu_ids and len(args.gpu_ids) > 0:
        gpu_ids = [str(g) for g in args.gpu_ids]
    else:
        gpu_ids = [str(i) for i in range(max(1, args.num_gpus))]
    print(f"Using GPUs: {', '.join(gpu_ids)}")

    # Build shard parameter list
    combos = list(product(args.grf_alpha_values, args.k_snn_output_res_values))
    shard_params = []
    for combo_idx, (alpha, nout) in enumerate(combos):
        # compute shard sizes (distribute remainder to first shards)
        base = args.trials // max(1, args.num_workers_per_job)
        rem = args.trials % max(1, args.num_workers_per_job)
        for s in range(args.num_workers_per_job):
            shard_trials = base + (1 if s < rem else 0)
            if shard_trials == 0:
                continue
            gpu_id = gpu_ids[(combo_idx * args.num_workers_per_job + s) % len(gpu_ids)]
            shard_params.append((
                args.pde_type, nout, alpha, s, shard_trials, args.seed,
                args, gpu_id
            ))

    # Decide pool size
    if args.max_concurrent is not None:
        pool_size = int(args.max_concurrent)
    else:
        pool_size = min(len(shard_params), len(gpu_ids) * max(1, args.num_workers_per_job))
    pool_size = max(1, pool_size)

    print(f"Launching pool with {pool_size} processes for {len(shard_params)} shards.")

    # Run shards
    if pool_size > 1:
        with multiprocessing.Pool(processes=pool_size) as pool:
            shard_outputs = pool.map(worker_run_shard, shard_params)
    else:
        shard_outputs = [worker_run_shard(p) for p in shard_params]

    # Group shard outputs by configuration (alpha, nout)
    shards_by_config = {}
    for (alpha, nout) in combos:
        shards_by_config[(alpha, nout)] = []
    for sp, p in zip(shard_outputs, shard_params):
        # shard_params tuple structure:
        # (pde_type, nout, alpha, shard_idx, shard_trials, base_seed, args, gpu_id)
        nout = p[1]
        alpha = p[2]
        if sp is not None:
            # if a shard finished, append its npz path
            shards_by_config.setdefault((alpha, nout), []).append(sp)

    # Aggregate
    results = {}
    for key, paths in shards_by_config.items():
        if not paths:
            continue
        stats = aggregate_config_from_shards(paths)
        if stats:
            results[key] = stats

    # Report missing
    missing = [(a, n) for (a, n) in combos if (a, n) not in results]
    if missing:
        print("\nWARNING: Missing results for combos (no shards completed?):")
        for a, n in missing:
            print(f"  rho={a}, Nout={n}")

    # LaTeX table
    table_tex = make_latex_table(
        results_map=results,
        grf_alphas=args.grf_alpha_values,
        Nouts=args.k_snn_output_res_values,
        caption=args.caption,
        label=args.label,
        value_fmt=args.value_fmt,
        use=args.use_test,
    )
    with open(args.latex_out, "w") as f:
        f.write(table_tex)
    print(f"\nLaTeX table saved to: {args.latex_out}\n")
    print(table_tex)
