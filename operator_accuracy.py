import argparse
import json
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(os.getcwd(), ".cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from model import (
    SimpleSpectralOperatorCNN,
    channels_to_spectrum_complex_torch,
    spectrum_complex_to_channels_torch,
)


def build_suffix(args, grf_alpha=None):
    alpha = args.grf_alpha if grf_alpha is None else grf_alpha
    if args.pde_type == "poisson":
        return f"poisson_grfA{alpha:.1f}T{args.grf_tau:.1f}OffS{args.grf_offset_sigma:.1f}"
    if args.pde_type == "step_index_fiber":
        return (
            f"fiber_GRFinA{alpha:.2f}T{args.grf_tau:.2f}_"
            f"coreR{args.fiber_core_radius_factor:.1f}_V{args.fiber_potential_depth:.1f}_"
            f"evoT{args.evolution_time_T:.1e}_steps{args.solver_num_steps}"
        )
    if args.pde_type == "grin_fiber":
        return (
            f"grinfiber_GRFinA{alpha:.2f}T{args.grf_tau:.2f}_"
            f"strength{args.grin_strength:.2e}_"
            f"evoT{args.evolution_time_T:.1e}_steps{args.solver_num_steps}"
        )
    if args.pde_type == "heat_equation":
        return (
            f"heat_GRFinA{alpha:.2f}T{args.grf_tau:.2f}_"
            f"nu{args.viscosity_nu:.2e}_evoT{args.evolution_time_T:.1e}"
        )
    raise ValueError(f"Unknown pde_type: {args.pde_type}")


def build_legacy_suffix(args, grf_alpha=None):
    alpha = args.grf_alpha if grf_alpha is None else grf_alpha
    if args.pde_type == "poisson":
        return build_suffix(args, grf_alpha=alpha)
    if args.pde_type == "step_index_fiber":
        return (
            f"fiber_GRFinA{alpha:.1f}T{args.grf_tau:.1f}_"
            f"coreR{args.fiber_core_radius_factor:.1f}_V{args.fiber_potential_depth:.1f}_"
            f"evoT{args.evolution_time_T:.1e}_steps{args.solver_num_steps}"
        )
    if args.pde_type == "grin_fiber":
        return (
            f"grinfiber_GRFinA{alpha:.1f}T{args.grf_tau:.1f}_"
            f"strength{args.grin_strength:.2e}_"
            f"evoT{args.evolution_time_T:.1e}_steps{args.solver_num_steps}"
        )
    if args.pde_type == "heat_equation":
        return (
            f"heat_GRFinA{alpha:.1f}T{args.grf_tau:.1f}_"
            f"nu{args.viscosity_nu:.2e}_evoT{args.evolution_time_T:.1e}"
        )
    raise ValueError(f"Unknown pde_type: {args.pde_type}")


def suffix_candidates(args, grf_alpha=None):
    primary = build_suffix(args, grf_alpha=grf_alpha)
    legacy = build_legacy_suffix(args, grf_alpha=grf_alpha)
    if legacy == primary:
        return [primary]
    return [primary, legacy]


def first_existing_path(path_templates, description):
    for path in path_templates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{description} not found; tried:\n  " + "\n  ".join(path_templates))


def parse_float_list(raw):
    return [float(x) for x in raw.replace(",", " ").split() if x.strip()]


def order_tag(order):
    text = f"{order:g}"
    return text.replace("-", "m").replace(".", "p")


def extract_center_block_np(full_spectrum_centered_np, k_extract):
    n_full = full_spectrum_centered_np.shape[0]
    if k_extract == n_full:
        return full_spectrum_centered_np
    start = n_full // 2 - k_extract // 2
    end = start + k_extract
    return full_spectrum_centered_np[start:end, start:end]


def pad_to_full_centered(block_centered, n_full):
    k_out = block_centered.shape[0]
    if k_out == n_full:
        return block_centered
    full = np.zeros((n_full, n_full), dtype=block_centered.dtype)
    start = n_full // 2 - k_out // 2
    end = start + k_out
    full[start:end, start:end] = block_centered
    return full


def sobolev_weights(k_grid_size, order):
    if k_grid_size % 2 == 0:
        k = np.arange(-k_grid_size // 2, k_grid_size // 2)
    else:
        k = np.arange(-(k_grid_size - 1) // 2, (k_grid_size - 1) // 2 + 1)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    return (1.0 + kx * kx + ky * ky) ** order


def weighted_norm(spec_centered, order):
    w = sobolev_weights(spec_centered.shape[0], order)
    return float(np.sqrt(np.sum(w * np.abs(spec_centered) ** 2)))


def relative_weighted_error(pred_centered, target_centered, order, eps=1e-12):
    numerator = weighted_norm(pred_centered - target_centered, order)
    denominator = weighted_norm(target_centered, order)
    return numerator / max(denominator, eps)


def relative_l2_spatial(pred_full_centered, target_full_centered, eps=1e-12):
    pred = np.fft.ifft2(np.fft.ifftshift(pred_full_centered))
    target = np.fft.ifft2(np.fft.ifftshift(target_full_centered))
    numerator = np.linalg.norm((pred - target).ravel())
    denominator = np.linalg.norm(target.ravel())
    return float(numerator / max(denominator, eps))


def summary_stats(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "se": float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0,
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def predict_spectra(model, gamma_b_full, batch_size, device):
    preds = []
    with torch.no_grad():
        for start in range(0, len(gamma_b_full), batch_size):
            batch_np = gamma_b_full[start:start + batch_size]
            batch = torch.stack([spectrum_complex_to_channels_torch(x) for x in batch_np]).to(device)
            out = model(batch).cpu()
            for i in range(out.shape[0]):
                preds.append(channels_to_spectrum_complex_torch(out[i]).numpy())
    return np.asarray(preds)


def make_accuracy_plot(summary, output_path):
    metric_order = [
        "rel_l2_truncated_spectral",
        "rel_l2_full_spectral",
        "rel_l2_full_spatial",
    ]
    metric_order.extend(k for k in summary if k.startswith("rel_H") and k.endswith("_full"))
    metric_order = [k for k in metric_order if k in summary]
    means = [summary[k]["mean"] for k in metric_order]
    errors = [summary[k]["se"] for k in metric_order]

    plt.figure(figsize=(max(7, 1.3 * len(metric_order)), 4.5))
    plt.bar(np.arange(len(metric_order)), means, yerr=errors, capsize=3)
    plt.xticks(np.arange(len(metric_order)), metric_order, rotation=30, ha="right")
    plt.ylabel("Relative error")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def evaluate(args):
    suffix = build_suffix(args)
    candidates = suffix_candidates(args)
    dataset_path = first_existing_path(
        [
            os.path.join(
                args.dataset_dir,
                f"dataset_{args.pde_type}_Nin{args.n_grid_sim_input_ds}_"
                f"Nout{args.snn_output_res}_{candidate}.npz",
            )
            for candidate in candidates
        ],
        "Dataset",
    )
    model_path = first_existing_path(
        [
            os.path.join(
                args.model_dir,
                f"snn_PDE{args.pde_type}_Kin{args.n_grid_sim_input_ds}_"
                f"Kout{args.snn_output_res}_H{args.snn_hidden_channels}_"
                f"L{args.snn_num_hidden_layers}_{candidate}.pth",
            )
            for candidate in candidates
        ],
        "Model",
    )

    data = np.load(dataset_path)
    gamma_b_all = data["gamma_b_calib"]
    gamma_a_target_all = data["gamma_a_snn_target_calib"]
    gamma_a_true_full_all = data["gamma_a_true_full_calib"]

    indices = np.arange(len(gamma_b_all))
    test_size = 1.0 - args.calib_split_ratio
    cal_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=args.random_seed,
        shuffle=True,
    )
    del cal_idx
    if args.max_samples is not None:
        test_idx = test_idx[:args.max_samples]

    gamma_b = gamma_b_all[test_idx]
    gamma_a_target = gamma_a_target_all[test_idx]
    gamma_a_true_full = gamma_a_true_full_all[test_idx]

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = SimpleSpectralOperatorCNN(
        K_input_resolution=args.n_grid_sim_input_ds,
        K_output_resolution=args.snn_output_res,
        hidden_channels=args.snn_hidden_channels,
        num_hidden_layers=args.snn_num_hidden_layers,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    pred_target = predict_spectra(model, gamma_b, args.batch_size, device)
    orders = parse_float_list(args.sobolev_orders)

    metrics = {
        "rel_l2_truncated_spectral": [],
        "rel_l2_full_spectral": [],
        "rel_l2_full_spatial": [],
    }
    for order in orders:
        tag = order_tag(order)
        metrics[f"rel_H{tag}_truncated"] = []
        metrics[f"rel_H{tag}_full"] = []

    n_full = args.n_grid_sim_input_ds
    for pred, target, true_full in zip(pred_target, gamma_a_target, gamma_a_true_full):
        pred_full = pad_to_full_centered(pred, n_full)
        metrics["rel_l2_truncated_spectral"].append(relative_weighted_error(pred, target, 0.0))
        metrics["rel_l2_full_spectral"].append(relative_weighted_error(pred_full, true_full, 0.0))
        metrics["rel_l2_full_spatial"].append(relative_l2_spatial(pred_full, true_full))
        for order in orders:
            tag = order_tag(order)
            metrics[f"rel_H{tag}_truncated"].append(relative_weighted_error(pred, target, order))
            metrics[f"rel_H{tag}_full"].append(relative_weighted_error(pred_full, true_full, order))

    summary = {key: summary_stats(values) for key, values in metrics.items()}
    payload = {
        "pde_type": args.pde_type,
        "suffix": suffix,
        "dataset_path": dataset_path,
        "model_path": model_path,
        "n_grid_sim_input_ds": args.n_grid_sim_input_ds,
        "snn_output_res": args.snn_output_res,
        "num_test_samples": int(len(test_idx)),
        "calib_split_ratio": args.calib_split_ratio,
        "random_seed": args.random_seed,
        "sobolev_orders": orders,
        "summary": summary,
    }

    os.makedirs(args.results_dir, exist_ok=True)
    base = os.path.join(
        args.results_dir,
        (
            f"operator_accuracy_PDE{args.pde_type}_Nin{args.n_grid_sim_input_ds}_"
            f"Nout{args.snn_output_res}_{suffix}"
        ),
    )
    np.savez_compressed(
        base + ".npz",
        test_indices=test_idx,
        **{key: np.asarray(values, dtype=float) for key, values in metrics.items()},
    )
    with open(base + "_summary.json", "w") as f:
        json.dump(payload, f, indent=2)
    if not args.no_plot:
        make_accuracy_plot(summary, base + ".png")

    print("\n=== Neural Operator Accuracy ===")
    print(f"PDE: {args.pde_type} | Nout={args.snn_output_res} | n_test={len(test_idx)}")
    for key in ["rel_l2_truncated_spectral", "rel_l2_full_spectral", "rel_l2_full_spatial"]:
        stat = summary[key]
        print(f"{key}: mean={stat['mean']:.4e}, median={stat['median']:.4e}, p95={stat['p95']:.4e}")
    for order in orders:
        key = f"rel_H{order_tag(order)}_full"
        stat = summary[key]
        print(f"{key}: mean={stat['mean']:.4e}, median={stat['median']:.4e}, p95={stat['p95']:.4e}")
    print(f"Saved metrics to:\n  {base + '.npz'}\n  {base + '_summary.json'}")
    if not args.no_plot:
        print(f"Saved plot to:\n  {base + '.png'}")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Report neural-operator predictive accuracy on held-out calibration/test data.")
    parser.add_argument("--pde_type", type=str, default="step_index_fiber",
                        choices=["poisson", "step_index_fiber", "grin_fiber", "heat_equation"])
    parser.add_argument("--n_grid_sim_input_ds", type=int, default=64)
    parser.add_argument("--snn_output_res", type=int, default=32)
    parser.add_argument("--dataset_dir", type=str, default="datasets")
    parser.add_argument("--model_dir", type=str, default="trained_snn_models")
    parser.add_argument("--results_dir", type=str, default="results_operator_accuracy")
    parser.add_argument("--snn_hidden_channels", type=int, default=64)
    parser.add_argument("--snn_num_hidden_layers", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--calib_split_ratio", type=float, default=0.5)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--sobolev_orders", type=str, default="1,2",
                        help="Comma- or space-separated Sobolev orders to report, in addition to L2.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_plot", action="store_true")

    parser.add_argument("--grf_alpha", type=float, default=4.0)
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
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    evaluate(args)


if __name__ == "__main__":
    main()
