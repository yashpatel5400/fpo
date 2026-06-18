import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from collection import (
    J_collect,
    SimpleSpectralOperatorCNN,
    channels_to_spectrum_complex_torch,
    disk_mask_torus,
    downsample_average,
    optimize,
    optimize_softmask_adam,
    pad_to_full_centered,
    spec_to_spatial_centered,
    spectrum_complex_to_channels_torch,
    transform_resource_field,
    nominal_obj,
)


def poisson_suffix(grf_alpha, grf_tau, grf_offset_sigma):
    return f"poisson_grfA{grf_alpha:.1f}T{grf_tau:.1f}OffS{grf_offset_sigma:.1f}"


def first_existing(candidates, description):
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{description} not found; tried:\n  " + "\n  ".join(candidates))


def load_radius(args, suffix):
    calib_subdir = (
        f"PDE{args.pde_type}_NinDS{args.n_grid_sim_input_ds}_SNNres{args.snn_output_res}_"
        f"KfullThm{args.n_grid_sim_input_ds}_s{args.s_theorem}_nu{args.nu_theorem}_{suffix}"
    )
    calib_filename = (
        f"coverage_data_PDE{args.pde_type}_thm_s{args.s_theorem}_nu{args.nu_theorem}_d2"
        f"_Nin{args.n_grid_sim_input_ds}_SNNout{args.snn_output_res}"
        f"_NfullThm{args.n_grid_sim_input_ds}_{suffix}.npz"
    )
    calib_path = first_existing(
        [
            os.path.join(args.calib_results_dir, calib_subdir, calib_filename),
            os.path.join(args.calib_results_dir, calib_filename),
        ],
        "Calibration result",
    )
    payload = np.load(calib_path)
    alphas = 1.0 - payload["nominal_coverages"]
    q_arr = payload["quantiles_q_hat_nu"]
    idx = int(np.argmin(np.abs(alphas - args.alpha_for_radius)))
    return float(np.sqrt(max(float(q_arr[idx]), 0.0)) * args.collection_radius_scale), calib_path


def load_model(args, model_path, device):
    model = SimpleSpectralOperatorCNN(
        args.n_grid_sim_input_ds,
        args.snn_output_res,
        hidden_channels=args.snn_hidden_channels,
        num_hidden_layers=args.snn_num_hidden_layers,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_spatial_field(model, gb, n_full, device):
    with torch.no_grad():
        x = spectrum_complex_to_channels_torch(gb).unsqueeze(0).to(device)
        y = model(x).squeeze(0).cpu()
        y_complex = channels_to_spectrum_complex_torch(y).numpy()
    return spec_to_spatial_centered(pad_to_full_centered(y_complex, n_full))


def run_multistage(
    u_field,
    args,
    trial_seed,
    r_radius,
    restarts,
    steps,
    init_centers=None,
):
    centers_prev = init_centers
    factor_prev = None
    factors = [int(x) for x in args.multi_stage_factors.split(",") if x.strip()]
    if 1 not in factors:
        factors.append(1)
    factors = sorted(set(factors), reverse=True)

    for factor in factors:
        u_stage = downsample_average(u_field, factor)
        radius_stage = max(1e-6, args.radius_px / factor)
        init_stage = None
        if centers_prev is not None and factor_prev is not None:
            scale = factor_prev / factor
            init_stage = [(y * scale, x * scale) for (y, x) in centers_prev]
        elif centers_prev is not None:
            init_stage = centers_prev

        stage_steps = max(200, int(steps / factor))
        centers_stage, _ = optimize_softmask_adam(
            u_pred_real=u_stage,
            K=args.K_facilities,
            radius_px=radius_stage,
            r_radius=r_radius,
            s_minus_nu=args.s_theorem - args.nu_theorem,
            steps=stage_steps,
            lr=args.collection_lr,
            tau=args.collection_tau,
            restarts=1 if init_stage is not None else restarts,
            seed=trial_seed,
            device=args.device,
            use_robust=(r_radius > 0.0),
            init_centers=init_stage,
        )
        centers_prev = centers_stage
        factor_prev = factor
    return centers_prev


def hard_refine_true(u_true, centers, args, seed, iters):
    if iters <= 0:
        return centers
    refined, _ = optimize(
        u_true,
        args.K_facilities,
        args.radius_px,
        nominal_obj,
        iters=iters,
        step_px=args.hard_refine_step_px,
        init_centers=centers,
        rng=np.random.default_rng(seed),
    )
    return refined


def draw_disks(ax, centers, radius_px, color, label, n_grid, linestyle="-"):
    first = True
    for y, x in centers:
        for dy in (-n_grid, 0, n_grid):
            for dx in (-n_grid, 0, n_grid):
                yy = y + dy
                xx = x + dx
                if -radius_px <= yy < n_grid + radius_px and -radius_px <= xx < n_grid + radius_px:
                    circ = Circle(
                        (xx, yy),
                        radius_px,
                        fill=False,
                        lw=2.2,
                        ec=color,
                        alpha=0.98,
                        linestyle=linestyle,
                    )
                    ax.add_patch(circ)
                    ax.plot(
                        [xx],
                        [yy],
                        marker="o",
                        ms=4.5,
                        mec="white" if color == "black" else "black",
                        mew=0.8,
                        mfc=color,
                        label=label if first else None,
                    )
                    first = False


def plot_trial(record, args, out_path):
    n_grid = record["u_pred"].shape[0]
    vmin = min(
        float(np.percentile(record["u_pred"], 0.5)),
        float(np.percentile(record["u_true"], 0.5)),
    )
    vmax = max(
        float(np.percentile(record["u_pred"], 99.5)),
        float(np.percentile(record["u_true"], 99.5)),
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3), dpi=180, sharex=True, sharey=True)
    panels = [("Predicted field (SNN)", record["u_pred"]), ("True field", record["u_true"])]
    for ax, (title, field) in zip(axes, panels):
        im = ax.imshow(field, origin="upper", interpolation="nearest", vmin=vmin, vmax=vmax)
        draw_disks(ax, record["w_nom"], args.radius_px, "crimson", "nominal", n_grid)
        draw_disks(ax, record["w_rob"], args.radius_px, "deepskyblue", "robust", n_grid)
        draw_disks(ax, record["w_oracle"], args.radius_px, "black", "true-field opt.", n_grid, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("x (pixels)")
        ax.set_xlim(0, n_grid)
        ax.set_ylim(n_grid, 0)
    axes[0].set_ylabel("y (pixels)")

    fig.subplots_adjust(right=0.84, left=0.08, top=0.90, bottom=0.10, wspace=0.15)
    cbar_ax = fig.add_axes([0.855, 0.16, 0.018, 0.68])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("intensity")
    handles = [
        Line2D([0], [0], color="crimson", marker="o", lw=2.0, label="nominal"),
        Line2D([0], [0], color="deepskyblue", marker="o", lw=2.0, label="robust"),
        Line2D([0], [0], color="black", marker="o", lw=2.0, linestyle="--", label="true-field opt."),
    ]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.885, 0.68), frameon=False)
    fig.suptitle(
        f"Collection placements, trial {record['trial_index']} (sample {record['sample_index']})",
        fontsize=12,
        y=0.98,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def format_sci(value):
    if value == 0:
        return "0"
    if abs(value) >= 1000 or abs(value) < 0.01:
        exponent = int(np.floor(np.log10(abs(value))))
        mantissa = value / (10**exponent)
        return f"${mantissa:.3f}\\times10^{{{exponent}}}$"
    return f"{value:.3f}"


def write_latex_table(records, out_path):
    with open(out_path, "w") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n\\small\n")
        f.write("\\caption{True-field collection performance for the visualization trials in \\Cref{fig:rob_nom_collect1,fig:rob_nom_collect2}. The oracle placement is optimized with access to the true field and is included only as a diagnostic upper benchmark.}\n")
        f.write("\\label{tab:collection_viz_performance}\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("Trial & $J_{\\mathrm{nom}}$ & $J_{\\mathrm{rob}}$ & $J_{\\mathrm{oracle}}$ & $J_{\\mathrm{rob}}-J_{\\mathrm{nom}}$ & $J_{\\mathrm{rob}}/J_{\\mathrm{oracle}}$ \\\\\n")
        f.write("\\midrule\n")
        for rec in records:
            ratio = rec["J_rob"] / rec["J_oracle"] if rec["J_oracle"] != 0 else np.nan
            f.write(
                f"{rec['label']} & {format_sci(rec['J_nom'])} & {format_sci(rec['J_rob'])} & "
                f"{format_sci(rec['J_oracle'])} & {format_sci(rec['J_rob'] - rec['J_nom'])} & {ratio:.3f} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def main():
    parser = argparse.ArgumentParser(description="Regenerate Appendix L collection visualizations with oracle overlays.")
    parser.add_argument("--pde_type", default="poisson")
    parser.add_argument("--n_grid_sim_input_ds", type=int, default=64)
    parser.add_argument("--snn_output_res", type=int, default=12)
    parser.add_argument("--grf_alpha", type=float, default=1.0)
    parser.add_argument("--grf_tau", type=float, default=1.0)
    parser.add_argument("--grf_offset_sigma", type=float, default=0.5)
    parser.add_argument("--dataset_dir", default="replication_runs/poisson_nout8_12_16_20260615/functional_coverage/datasets")
    parser.add_argument("--model_dir", default="replication_runs/poisson_nout8_12_16_20260615/functional_coverage/models")
    parser.add_argument("--calib_results_dir", default="replication_runs/poisson_nout8_12_16_20260615/functional_coverage/calibration_poisson_nout8_12_16")
    parser.add_argument("--out_dir", default="replication_runs/poisson_nout8_12_16_20260615/collection/appendix_viz")
    parser.add_argument("--paper_image_dir", default="../images")
    parser.add_argument("--trial_indices", default="10,11")
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2100)
    parser.add_argument("--K_facilities", type=int, default=3)
    parser.add_argument("--radius_px", type=int, default=6)
    parser.add_argument("--iters", type=int, default=800)
    parser.add_argument("--collection_restarts", type=int, default=3)
    parser.add_argument("--oracle_iters", type=int, default=1400)
    parser.add_argument("--oracle_restarts", type=int, default=16)
    parser.add_argument("--oracle_hard_refine_iters", type=int, default=600)
    parser.add_argument("--hard_refine_step_px", type=int, default=2)
    parser.add_argument("--collection_lr", type=float, default=0.15)
    parser.add_argument("--collection_tau", type=float, default=1.5)
    parser.add_argument("--multi_stage_factors", default="4,2,1")
    parser.add_argument("--resource_transform", default="real", choices=["real", "abs", "positive"])
    parser.add_argument("--alpha_for_radius", type=float, default=0.10)
    parser.add_argument("--collection_radius_scale", type=float, default=1.0)
    parser.add_argument("--s_theorem", type=float, default=2.0)
    parser.add_argument("--nu_theorem", type=float, default=2.0)
    parser.add_argument("--snn_hidden_channels", type=int, default=64)
    parser.add_argument("--snn_num_hidden_layers", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    suffix = poisson_suffix(args.grf_alpha, args.grf_tau, args.grf_offset_sigma)
    dataset_path = first_existing(
        [
            os.path.join(
                args.dataset_dir,
                f"dataset_{args.pde_type}_Nin{args.n_grid_sim_input_ds}_Nout{args.snn_output_res}_{suffix}.npz",
            )
        ],
        "Dataset",
    )
    model_path = first_existing(
        [
            os.path.join(
                args.model_dir,
                f"snn_PDE{args.pde_type}_Kin{args.n_grid_sim_input_ds}_Kout{args.snn_output_res}"
                f"_H{args.snn_hidden_channels}_L{args.snn_num_hidden_layers}_{suffix}.pth",
            )
        ],
        "Model",
    )
    r_radius, calib_path = load_radius(args, suffix)

    data = np.load(dataset_path)
    gb_all = data["gamma_b_calib"]
    ga_true_full = data["gamma_a_true_full_calib"]
    n_full = args.n_grid_sim_input_ds
    model = load_model(args, model_path, args.device)

    rng_master = np.random.default_rng(args.seed)
    idxs = rng_master.choice(gb_all.shape[0], size=min(args.trials, gb_all.shape[0]), replace=False)
    requested_trials = [int(x) for x in args.trial_indices.split(",") if x.strip()]
    requested = set(requested_trials)
    records = []

    for trial_index, sample_index in enumerate(idxs):
        trial_seed = int(rng_master.integers(0, 10_000_000))
        if trial_index not in requested:
            continue

        u_pred = transform_resource_field(
            predict_spatial_field(model, gb_all[sample_index], n_full, args.device),
            args.resource_transform,
        )
        u_true = transform_resource_field(
            spec_to_spatial_centered(ga_true_full[sample_index]),
            args.resource_transform,
        )

        w_nom = run_multistage(
            u_pred,
            args,
            trial_seed,
            r_radius=0.0,
            restarts=args.collection_restarts,
            steps=args.iters,
        )
        w_rob = run_multistage(
            u_pred,
            args,
            trial_seed,
            r_radius=r_radius,
            restarts=args.collection_restarts,
            steps=args.iters,
        )
        w_oracle = run_multistage(
            u_true,
            args,
            trial_seed + 123_457,
            r_radius=0.0,
            restarts=args.oracle_restarts,
            steps=args.oracle_iters,
        )
        w_oracle = hard_refine_true(
            u_true,
            w_oracle,
            args,
            trial_seed + 765_431,
            args.oracle_hard_refine_iters,
        )

        vals = {}
        for name, centers in [("nom", w_nom), ("rob", w_rob), ("oracle", w_oracle)]:
            vals[f"J_{name}"] = J_collect(u_true, disk_mask_torus(n_full, centers, args.radius_px))

        record = {
            "label": f"{len(records) + 1}",
            "trial_index": int(trial_index),
            "sample_index": int(sample_index),
            "u_pred": u_pred,
            "u_true": u_true,
            "w_nom": w_nom,
            "w_rob": w_rob,
            "w_oracle": w_oracle,
            **vals,
        }
        records.append(record)

    if len(records) != len(requested_trials):
        raise RuntimeError(f"Only generated {len(records)} records for requested trials {requested_trials}")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.paper_image_dir, exist_ok=True)

    serializable = []
    for idx, rec in enumerate(records, start=1):
        paper_path = os.path.join(args.paper_image_dir, f"collection_viz{idx}.png")
        run_path = os.path.join(args.out_dir, f"collection_viz{idx}.png")
        plot_trial(rec, args, paper_path)
        plot_trial(rec, args, run_path)
        serializable.append(
            {
                k: v
                for k, v in rec.items()
                if k not in {"u_pred", "u_true"}
            }
        )

    table_path = os.path.join(args.out_dir, "collection_viz_performance_table.tex")
    write_latex_table(records, table_path)
    json_path = os.path.join(args.out_dir, "collection_viz_performance.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "dataset_path": dataset_path,
                "model_path": model_path,
                "calib_path": calib_path,
                "r_radius": r_radius,
                "records": serializable,
            },
            f,
            indent=2,
        )

    print(f"Saved paper images to {args.paper_image_dir}/collection_viz1.png and collection_viz2.png")
    print(f"Saved diagnostic images/table/json under {args.out_dir}")
    for rec in serializable:
        print(
            f"trial {rec['trial_index']} sample {rec['sample_index']}: "
            f"J_nom={rec['J_nom']:.3f}, J_rob={rec['J_rob']:.3f}, J_oracle={rec['J_oracle']:.3f}"
        )


if __name__ == "__main__":
    main()
