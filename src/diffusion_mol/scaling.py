"""Temperature scaling (fit on train only) and log10 target helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler


def log10_d(d: np.ndarray | float) -> np.ndarray:
    d = np.asarray(d, dtype=np.float64)
    if np.any(d <= 0):
        raise ValueError("D must be positive for log10")
    return np.log10(d)


def inv_log10_d(logd: np.ndarray | float) -> np.ndarray:
    return np.power(10.0, np.asarray(logd, dtype=np.float64))


class TemperatureScaler:
    """Standardize temperature using sklearn; fit only on training data."""

    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._fitted = False

    def fit(self, temperature_k: np.ndarray) -> "TemperatureScaler":
        self._scaler.fit(np.asarray(temperature_k, dtype=np.float64).reshape(-1, 1))
        self._fitted = True
        return self

    def transform(self, temperature_k: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TemperatureScaler.fit() must be called first")
        t = np.asarray(temperature_k, dtype=np.float64).reshape(-1, 1)
        return self._scaler.transform(t).astype(np.float64).ravel()

    def fit_transform(self, temperature_k: np.ndarray) -> np.ndarray:
        return self.fit(temperature_k).transform(temperature_k)

    def state_dict(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("Not fitted")
        return {
            "mean_": self._scaler.mean_.tolist(),
            "scale_": self._scaler.scale_.tolist(),
            "var_": self._scaler.var_.tolist(),
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self._scaler.mean_ = np.array(d["mean_"], dtype=np.float64)
        self._scaler.scale_ = np.array(d["scale_"], dtype=np.float64)
        self._scaler.var_ = np.array(d["var_"], dtype=np.float64)
        self._scaler.n_features_in_ = 1
        self._scaler.n_samples_seen_ = None
        self._fitted = True

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TemperatureScaler":
        obj = cls()
        obj.load_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        return obj
