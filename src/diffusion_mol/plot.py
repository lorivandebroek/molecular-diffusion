"""Parity, residual vs T, and benchmark bar charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_parity_log(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    out_path: str | Path,
    *,
    title: str = "Predicted vs true log10(D)",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true_log, y_pred_log, alpha=0.25, s=8, edgecolors="none")
    lo = min(y_true_log.min(), y_pred_log.min())
    hi = max(y_true_log.max(), y_pred_log.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("True log10(D)")
    ax.set_ylabel("Predicted log10(D)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_residuals_vs_temperature(
    temperature_k: np.ndarray,
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    out_path: str | Path,
    *,
    title: str = "Residual (pred - true) vs temperature",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res = y_pred_log - y_true_log
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(temperature_k, res, alpha=0.25, s=8, edgecolors="none")
    ax.axhline(0.0, color="k", linestyle="--", lw=1)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Residual log10(D)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_benchmark_bars(
    model_to_rmse_log: dict[str, float],
    out_path: str | Path,
    *,
    title: str = "Test RMSE (log10 D)",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(model_to_rmse_log.keys())
    vals = [model_to_rmse_log[k] for k in names]
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.6), 4))
    ax.bar(names, vals, color="steelblue")
    ax.set_ylabel("RMSE log10(D)")
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
