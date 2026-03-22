"""Regression metrics in log10(D) and linear D space."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from diffusion_mol.scaling import inv_log10_d


def regression_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    y_true_log = np.asarray(y_true_log, dtype=np.float64).ravel()
    y_pred_log = np.asarray(y_pred_log, dtype=np.float64).ravel()
    mae_log = float(mean_absolute_error(y_true_log, y_pred_log))
    rmse_log = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))
    r2_log = float(r2_score(y_true_log, y_pred_log))

    d_t = inv_log10_d(y_true_log)
    d_p = inv_log10_d(y_pred_log)
    mae_lin = float(mean_absolute_error(d_t, d_p))
    rmse_lin = float(np.sqrt(mean_squared_error(d_t, d_p)))

    mask = d_t > 1e-12
    rel = np.abs(d_p[mask] - d_t[mask]) / d_t[mask]
    med_rel = float(np.median(rel)) if rel.size else float("nan")

    return {
        "mae_log10": mae_log,
        "rmse_log10": rmse_log,
        "r2_log10": r2_log,
        "mae_D": mae_lin,
        "rmse_D": rmse_lin,
        "median_abs_rel_error_D": med_rel,
    }


def metrics_to_json_serializable(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, float)):
            out[k] = float(v)
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        elif isinstance(v, dict):
            out[k] = metrics_to_json_serializable(v)
        else:
            out[k] = v
    return out
