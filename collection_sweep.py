import os
import argparse
import subprocess
import multiprocessing
from itertools import product
import json
import numpy as np
from textwrap import dedent

def run_script(script_name, args_list, log_prefix=""):
    """Run a python script with arguments; return True if success."""
    cmd = ["python", script_name] + [str(a) for a in args_list]
    print(f"{log_prefix}Executing: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=10800)
        if proc.stderr:
            lower = proc.stderr.lower()
            if "traceback" in lower or "error" in lower:
                print(f"{log_prefix}Stderr (possible issues):\n{proc.stderr[:1500]}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{log_prefix}Error running {script_name} (return code {e.returncode})")
        print(f"STDOUT:\n{e.stdout[:2000]}")
        print(f"STDERR:\n{e.stderr[:2000]}")
        return False
    except subprocess.TimeoutExpired as e:
        print(f"{log_prefix}Timeout after {e.timeout} seconds.")
        return False
    except Exception as ex:
        print(f"{log_prefix}Unexpected error: {ex}")
        return False

def build_suffix(pde_type, args_like):
    if pde_type == "poisson":
        return f"poisson_grfA{args_like.grf_alpha:.1f}T{args_like.grf_tau:.1f}OffS{args_like.grf_offset_sigma:.1f}"
    elif pde_type == "step_index_fiber":
        return (f"fiber_GRFinA{args_like.grf_alpha:.1f}T{args_like.grf_tau:.1f}_"
                f"coreR{args_like.fiber_core_radius_factor:.1f}_V{args_like.fiber_potential_depth:.1f}_"
                f"evoT{args_like.evolution_time_T:.1e}_steps{args_like.solver_num_steps}")
    elif pde_type == "grin_fiber":
        return (f"grinfiber_GRFinA{args_like.grf_alpha:.1f}T{args_like.grf_tau:.1f}_"
                f"strength{args_like.grin_strength:.2e}_"
                f"evoT{args_like.evolution_time_T:.1e}_steps{args_like.solver_num_steps}")
    elif pde_type == "heat_equation":
        return (f"heat_GRFinA{args_like.grf_alpha:.1f}T{args_like.grf_tau:.1f}_"
                f"nu{args_like.viscosity_nu:.2e}_evoT{args_like.evolution_time_T:.1e}")
    else:
        raise ValueError(f"Unknown pde_type: {pde_type}")

def expected_stats_paths(base_out_dir, pde_type, Nin, Nout, K, rpx, alpha_for_radius, suffix):
    base = os.path.join(
        base_out_dir,
        f"facet1_results_PDE{pde_type}_N{Nin}_Nout{Nout}_K{K}_rpx{rpx}_alpha{alpha_for_radius:.2f}_{suffix}"
    )
    return base + ".npz", base + "_stats.json"

def worker_run_collection(params):
    """
    One sweep worker: runs collection_experiment.py for a (alpha, Nout) combination if needed,
    then loads *_stats.json and returns a summary blob.
    """
    (pde_type, k_snn_out, grf_alpha, args_ns, exp_idx, total_exps) = params
    a = argparse.Namespace(**vars(args_ns))  # clone
    a.grf_alpha = grf_alpha
    suffix = build_suffix(pde_type, a)

    log_prefix = f"[Worker {os.getpid()} {exp_idx+1}/{total_exps} ρ={grf_alpha:.2f}, Nout={k_snn_out}] "

    # expected outputs (created by collection_experiment.py)
    npz_path, stats_json_path = expected_stats_paths(
        a.results_out, pde_type, a.n_grid_sim_input_ds, k_snn_out,
        a.K_facilities, a.radius_px, a.alpha_for_radius, suffix
    )

    need_run = True
    if a.skip_completed and os.path.exists(stats_json_path):
        print(f"{log_prefix}Skipping: found existing stats {stats_json_path}")
        need_run = False

    if need_run:
        # build args for collection_experiment.py
        cmd_args = [
            "--pde_type", pde_type,
            "--n_grid_sim_input_ds", a.n_grid_sim_input_ds,
            "--snn_output_res", k_snn_out,
            "--dataset_dir", a.dataset_dir,
            "--model_dir", a.model_dir,
            "--calib_results_dir", a.calib_results_dir,
            "--alpha_for_radius", a.alpha_for_radius,
            "--K_facilities", a.K_facilities,
            "--radius_px", a.radius_px,
            "--iters", a.iters,
            "--trials", a.trials,
            "--seed", a.seed,
            "--results_out", a.results_out,
            "--s_theorem", a.s_theorem,
            "--nu_theorem", a.nu_theorem,
            "--snn_hidden_channels", a.snn_hidden_channels,
            "--snn_num_hidden_layers", a.snn_num_hidden_layers,
            # suffix knobs (must match dataset/model/calib runs)
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
        # grf_alpha is per-combo
        cmd_args += ["--grf_alpha", grf_alpha]

        ok = run_script(a.collection_script, cmd_args, log_prefix=log_prefix)
        if not ok:
            print(f"{log_prefix}Collection experiment FAILED. Returning None.")
            return None

    # load stats json
    try:
        with open(stats_json_path, "r") as f:
            stats = json.load(f)
        res_key = (grf_alpha, k_snn_out)
        print(f"{log_prefix}Loaded stats: Δ={stats['mean_diff']:.4g}, p_ttest={stats.get('p_ttest_greater')}, p_sign={stats.get('sign_test_two_sided_p')}")
        return res_key, stats
    except Exception as e:
        print(f"{log_prefix}ERROR reading stats at {stats_json_path}: {e}")
        return None

def make_latex_table(results_map, grf_alphas, Nouts, caption, label,
                     value_fmt="Δ={delta:.3f} (p={p:.3g})", use="ttest"):
    """
    results_map[(alpha, Nout)] -> stats dict
    Bold-codes cells where p-value < 0.05.
    'use' can be 'ttest' or 'sign'.
    """
    header_cols = " & ".join([f"$N_{{out}}={n}$" for n in Nouts])
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{{l|" + f"{'c'*len(Nouts)}" + r"}}",
        r"\toprule",
        rf"$\rho$ & {header_cols} \\ \midrule"
    ]

    for alpha in grf_alphas:
        row_elems = []
        for nout in Nouts:
            st = results_map.get((alpha, nout))
            if st is None:
                row_elems.append(r"\textemdash")
            else:
                delta = st["mean_diff"]
                if use == "ttest":
                    p = st.get("p_ttest_greater")
                else:
                    p = st.get("sign_test_two_sided_p")

                try:
                    cell = value_fmt.format(delta=delta, p=p if p is not None else float("nan"))
                except Exception:
                    cell = f"{delta:.3f}"
                cell = f"{p:.3f}"

                # --- bold significant results ---
                try:
                    if p is not None and p < 0.05:
                        cell = r"\textbf{" + cell + "}"
                except Exception:
                    pass

                row_elems.append(cell)

        line = f"{alpha:.2f} & " + " & ".join(row_elems) + r" \\"
        lines.append(line)

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}"
    ]
    return "\n".join(lines)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sweep robust collection experiment across (SNN out res, GRF alpha) and produce LaTeX table.")
    # core sweep
    ap.add_argument('--pde_type', type=str, default="poisson",
                    choices=["poisson","step_index_fiber","grin_fiber","heat_equation"])
    ap.add_argument('--k_snn_output_res_values', nargs='+', type=int, default=[32, 48])
    ap.add_argument('--grf_alpha_values', nargs='+', type=float, default=[2.5, 4.0])
    ap.add_argument('--n_grid_sim_input_ds', type=int, default=64)

    # experiment knobs (mirrors collection_experiment.py)
    ap.add_argument('--dataset_dir', type=str, default='datasets')
    ap.add_argument('--model_dir', type=str, default='trained_snn_models')
    ap.add_argument('--calib_results_dir', type=str, default='results_conformal_theorem_validation')
    ap.add_argument('--results_out', type=str, default='results_facet1_collection_sweep')
    ap.add_argument('--collection_script', type=str, default='collection.py')

    ap.add_argument('--trials', type=int, default=30)
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

    # infra
    ap.add_argument('--num_processes', type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument('--skip_completed', action='store_true')

    # table options
    ap.add_argument('--use_test', type=str, default='ttest', choices=['ttest','sign'], help="Which p-value to show in table")
    ap.add_argument('--value_fmt', type=str, default="Δ={delta:.3f} (p={p:.3g})", help="Cell format; fields: {delta}, {p}")
    ap.add_argument('--caption', type=str, default="Robust vs nominal collection improvement across discretizations and GRF smoothness.")
    ap.add_argument('--label', type=str, default="tab:collection_sweep")
    ap.add_argument('--latex_out', type=str, default='results_collection_sweep_table.tex')

    args = ap.parse_args()

    # dirs
    os.makedirs(args.results_out, exist_ok=True)

    # Prepare param tuples
    combos = list(product(args.grf_alpha_values, args.k_snn_output_res_values))
    total = len(combos)

    params_list = []
    idx = 0
    for grf_alpha in args.grf_alpha_values:
        for nout in args.k_snn_output_res_values:
            params_list.append((
                args.pde_type,
                nout,
                grf_alpha,
                args,
                idx,
                total
            ))
            idx += 1

    # Run
    results = {}
    if args.num_processes > 1 and len(params_list) > 0:
        print(f"Using {args.num_processes} processes.")
        with multiprocessing.Pool(processes=args.num_processes) as pool:
            out_list = pool.map(worker_run_collection, params_list)
    else:
        out_list = [worker_run_collection(p) for p in params_list]

    for item in out_list:
        if item:
            key, stats = item
            results[key] = stats

    # Report missing
    missing = [(a, n) for (a, n) in combos if (a, n) not in results]
    if missing:
        print("\nWARNING: Missing results for these combos (failed or not run):")
        for a, n in missing:
            print(f"  rho={a}, Nout={n}")

    # Build LaTeX table
    table_tex = make_latex_table(
        results_map=results,
        grf_alphas=args.grf_alpha_values,
        Nouts=args.k_snn_output_res_values,
        caption=args.caption,
        label=args.label,
        value_fmt=args.value_fmt,
        use=args.use_test
    )
    with open(args.latex_out, "w") as f:
        f.write(table_tex)
    print(f"\nLaTeX table saved to: {args.latex_out}\n")
    print(table_tex)
