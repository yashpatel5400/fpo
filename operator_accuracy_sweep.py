import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np

from operator_accuracy import build_suffix, order_tag


def parse_float_list(raw):
    if isinstance(raw, (list, tuple)):
        raw = " ".join(str(x) for x in raw)
    return [float(x) for x in raw.replace(",", " ").split() if x.strip()]


def parse_int_list(raw):
    if isinstance(raw, (list, tuple)):
        raw = " ".join(str(x) for x in raw)
    return [int(x) for x in raw.replace(",", " ").split() if x.strip()]


def expected_summary_path(args, grf_alpha, nout):
    suffix = build_suffix(args, grf_alpha=grf_alpha)
    filename = (
        f"operator_accuracy_PDE{args.pde_type}_Nin{args.n_grid_sim_input_ds}_"
        f"Nout{nout}_{suffix}_summary.json"
    )
    return os.path.join(args.results_dir, filename)


def run_one(args, grf_alpha, nout):
    summary_path = expected_summary_path(args, grf_alpha, nout)
    if os.path.exists(summary_path) and not args.force:
        print(f"Skipping existing accuracy summary: {summary_path}")
        return summary_path

    cmd = [
        sys.executable,
        args.accuracy_script,
        "--pde_type", args.pde_type,
        "--n_grid_sim_input_ds", str(args.n_grid_sim_input_ds),
        "--snn_output_res", str(nout),
        "--dataset_dir", args.dataset_dir,
        "--model_dir", args.model_dir,
        "--results_dir", args.results_dir,
        "--snn_hidden_channels", str(args.snn_hidden_channels),
        "--snn_num_hidden_layers", str(args.snn_num_hidden_layers),
        "--batch_size", str(args.batch_size),
        "--calib_split_ratio", str(args.calib_split_ratio),
        "--random_seed", str(args.random_seed),
        "--sobolev_orders", args.sobolev_orders,
        "--grf_alpha", str(grf_alpha),
        "--grf_tau", str(args.grf_tau),
        "--grf_offset_sigma", str(args.grf_offset_sigma),
        "--L_domain", str(args.L_domain),
        "--fiber_core_radius_factor", str(args.fiber_core_radius_factor),
        "--fiber_potential_depth", str(args.fiber_potential_depth),
        "--grin_strength", str(args.grin_strength),
        "--viscosity_nu", str(args.viscosity_nu),
        "--evolution_time_T", str(args.evolution_time_T),
        "--solver_num_steps", str(args.solver_num_steps),
    ]
    if args.max_samples is not None:
        cmd.extend(["--max_samples", str(args.max_samples)])
    if args.no_plot:
        cmd.append("--no_plot")
    if args.cpu:
        cmd.append("--cpu")

    print("Executing:", " ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr)
        raise RuntimeError(f"operator_accuracy.py failed for rho={grf_alpha}, Nout={nout}")
    if result.stderr:
        print(result.stderr)
    return summary_path


def write_csv(rows, output_path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def metric_mean(payload, metric):
    return payload["summary"][metric]["mean"]


def write_latex(rows, args, output_path):
    orders = parse_float_list(args.sobolev_orders)
    h_metrics = [(order, f"rel_H{order_tag(order)}_full") for order in orders]
    with open(output_path, "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n\\small\n")
        f.write("\\begin{tabular}{ll" + "c" * (3 + len(h_metrics)) + "}\n")
        f.write("\\toprule\n")
        headers = ["$\\rho$", "$N_{out}$", "Rel. $L^2_N$", "Rel. full $L^2$", "Rel. phys. $L^2$"]
        headers.extend([f"Rel. full $H^{{{order:g}}}$" for order, _ in h_metrics])
        f.write(" & ".join(headers) + " \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            values = [
                f"{row['grf_alpha']:.2f}",
                f"{row['snn_output_res']}",
                f"{row['rel_l2_truncated_spectral_mean']:.3e}",
                f"{row['rel_l2_full_spectral_mean']:.3e}",
                f"{row['rel_l2_full_spatial_mean']:.3e}",
            ]
            values.extend([f"{row[metric + '_mean']:.3e}" for _, metric in h_metrics])
            f.write(" & ".join(values) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Neural-operator predictive accuracy on held-out test samples.}\n")
        f.write(f"\\label{{tab:operator_accuracy_{args.pde_type}}}\n")
        f.write("\\end{table}\n")


def main():
    parser = argparse.ArgumentParser(description="Sweep neural-operator predictive accuracy across experiment configurations.")
    parser.add_argument("--accuracy_script", type=str, default="operator_accuracy.py")
    parser.add_argument("--pde_type", type=str, required=True,
                        choices=["poisson", "step_index_fiber", "grin_fiber", "heat_equation"])
    parser.add_argument("--k_snn_output_res_values", nargs="+", required=True)
    parser.add_argument("--grf_alpha_values", nargs="+", required=True)
    parser.add_argument("--n_grid_sim_input_ds", type=int, default=64)
    parser.add_argument("--dataset_dir", type=str, default="datasets")
    parser.add_argument("--model_dir", type=str, default="trained_snn_models")
    parser.add_argument("--results_dir", type=str, default="results_operator_accuracy")
    parser.add_argument("--snn_hidden_channels", type=int, default=64)
    parser.add_argument("--snn_num_hidden_layers", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--calib_split_ratio", type=float, default=0.5)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--sobolev_orders", type=str, default="1,2")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--csv_out", type=str, default=None)
    parser.add_argument("--latex_out", type=str, default=None)

    parser.add_argument("--grf_tau", type=float, default=1.0)
    parser.add_argument("--grf_offset_sigma", type=float, default=0.5)
    parser.add_argument("--L_domain", type=float, default=2 * np.pi)
    parser.add_argument("--fiber_core_radius_factor", type=float, default=0.2)
    parser.add_argument("--fiber_potential_depth", type=float, default=1.0)
    parser.add_argument("--grin_strength", type=float, default=0.1)
    parser.add_argument("--viscosity_nu", type=float, default=0.01)
    parser.add_argument("--evolution_time_T", type=float, default=0.1)
    parser.add_argument("--solver_num_steps", type=int, default=50)

    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    grf_values = parse_float_list(args.grf_alpha_values)
    nout_values = parse_int_list(args.k_snn_output_res_values)
    rows = []
    payloads = []
    for grf_alpha in grf_values:
        for nout in nout_values:
            summary_path = run_one(args, grf_alpha, nout)
            with open(summary_path) as f:
                payload = json.load(f)
            payloads.append(payload)
            row = {
                "pde_type": args.pde_type,
                "grf_alpha": grf_alpha,
                "snn_output_res": nout,
                "num_test_samples": payload["num_test_samples"],
                "rel_l2_truncated_spectral_mean": metric_mean(payload, "rel_l2_truncated_spectral"),
                "rel_l2_full_spectral_mean": metric_mean(payload, "rel_l2_full_spectral"),
                "rel_l2_full_spatial_mean": metric_mean(payload, "rel_l2_full_spatial"),
            }
            for order in parse_float_list(args.sobolev_orders):
                metric = f"rel_H{order_tag(order)}_full"
                row[f"{metric}_mean"] = metric_mean(payload, metric)
            rows.append(row)

    summary_out = os.path.join(args.results_dir, f"operator_accuracy_sweep_PDE{args.pde_type}_summary.json")
    with open(summary_out, "w") as f:
        json.dump({"pde_type": args.pde_type, "rows": rows, "payloads": payloads}, f, indent=2)

    csv_out = args.csv_out or os.path.join(args.results_dir, f"operator_accuracy_sweep_PDE{args.pde_type}.csv")
    latex_out = args.latex_out or os.path.join(args.results_dir, f"operator_accuracy_sweep_PDE{args.pde_type}.tex")
    write_csv(rows, csv_out)
    write_latex(rows, args, latex_out)

    print(f"Saved sweep summary to: {summary_out}")
    print(f"Saved CSV table to: {csv_out}")
    print(f"Saved LaTeX table to: {latex_out}")


if __name__ == "__main__":
    main()
