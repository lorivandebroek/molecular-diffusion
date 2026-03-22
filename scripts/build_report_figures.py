#!/usr/bin/env python3
"""Build PDF/PNG figures for report/main.tex from artifacts/benchmark_summary.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load_benchmark(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run training and scripts/run_eval.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _series(data: dict) -> tuple[dict[str, float], dict[str, float]]:
    """Return (random_metrics, scaffold_metrics) per model key for ridge, mlp, gnn."""
    by_name = {s["split"]: s["benchmarks"] for s in data["splits"]}
    keys = ("ridge_fp", "mlp", "gnn")
    r = {k: by_name["random"][k] for k in keys}
    s = {k: by_name["scaffold"][k] for k in keys}
    return r, s


def plot_model_comparison(data: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rnd, scf = _series(data)
    labels = ["Ridge\n(FP+T)", "MLP\n(Morgan+T)", "GNN\n(graph+T)"]
    keys = ("ridge_fp", "mlp", "gnn")
    x = np.arange(len(keys))
    w = 0.36

    # Colors: teal / rust — readable in print & screen
    c_rnd = "#1a6b7c"
    c_scf = "#c45c3e"

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))

    rmse_r = [rnd[k]["rmse_log10"] for k in keys]
    rmse_s = [scf[k]["rmse_log10"] for k in keys]
    ax = axes[0]
    ax.bar(x - w / 2, rmse_r, w, label="Random split", color=c_rnd, edgecolor="white", linewidth=0.6)
    ax.bar(x + w / 2, rmse_s, w, label="Scaffold split", color=c_scf, edgecolor="white", linewidth=0.6)
    mb = data["splits"][0]["benchmarks"]["mean_baseline"]["rmse_log10"]
    ax.axhline(mb, color="#888", linestyle="--", linewidth=1, label="Mean baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Test RMSE ($\log_{10} D$)")
    ax.set_title("Error in log space (primary training target)")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.set_ylim(0, max(max(rmse_r), max(rmse_s), mb) * 1.12)

    r2_r = [rnd[k]["r2_log10"] for k in keys]
    r2_s = [scf[k]["r2_log10"] for k in keys]
    ax = axes[1]
    ax.bar(x - w / 2, r2_r, w, label="Random split", color=c_rnd, edgecolor="white", linewidth=0.6)
    ax.bar(x + w / 2, r2_s, w, label="Scaffold split", color=c_scf, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Test $R^2$ ($\log_{10} D$)")
    ax.set_title("Variance explained in log space")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right", fontsize=8)

    fig.suptitle("Held-out test performance (molecule-level splits)", fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"model_comparison.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Second figure: median relative error on D (interpretable)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    med_r = [rnd[k]["median_abs_rel_error_D"] for k in keys]
    med_s = [scf[k]["median_abs_rel_error_D"] for k in keys]
    ax.bar(x - w / 2, med_r, w, label="Random split", color=c_rnd, edgecolor="white", linewidth=0.6)
    ax.bar(x + w / 2, med_s, w, label="Scaffold split", color=c_scf, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Median |relative error| on D")
    ax.set_title("Typical relative error on linear D (test set)")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.set_ylim(0, max(max(med_r), max(med_s)) * 1.15)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"median_rel_error_D.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "artifacts" / "benchmark_summary.json",
    )
    ap.add_argument("--out-dir", type=Path, default=ROOT / "report" / "figures")
    args = ap.parse_args()
    data = _load_benchmark(args.benchmark)
    plot_model_comparison(data, args.out_dir)
    print(f"Wrote figures to {args.out_dir}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
