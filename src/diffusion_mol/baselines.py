"""Constant and Ridge baselines on fingerprints + scaled temperature."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


def predict_constant(mean_logd: float, n: int) -> np.ndarray:
    return np.full((n,), mean_logd, dtype=np.float64)


def fit_eval_ridge(
    X_fp_train: np.ndarray,
    t_train_scaled: np.ndarray,
    y_train_log: np.ndarray,
    X_fp_test: np.ndarray,
    t_test_scaled: np.ndarray,
    *,
    alpha: float = 1.0,
) -> tuple[np.ndarray, Ridge]:
    X_tr = np.concatenate([X_fp_train, t_train_scaled.reshape(-1, 1)], axis=1)
    X_te = np.concatenate([X_fp_test, t_test_scaled.reshape(-1, 1)], axis=1)
    model = Ridge(alpha=alpha)
    model.fit(X_tr, y_train_log)
    pred = model.predict(X_te)
    return pred.astype(np.float64), model
