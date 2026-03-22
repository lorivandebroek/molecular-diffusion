"""Load a trained checkpoint and predict D from SMILES + temperature (K)."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from diffusion_mol.featurize import EDGE_ATTR_DIM, atom_feature_dim, morgan_fingerprint_bitvect, smiles_to_pyg_data
from diffusion_mol.models.gnn import GNNRegressor
from diffusion_mol.models.mlp import MLPRegressor
from diffusion_mol.scaling import TemperatureScaler, inv_log10_d
from diffusion_mol.train import load_checkpoint


def load_predictor(checkpoint_path: str | Path, device: torch.device | None = None):
    device = device or torch.device("cpu")
    ckpt = load_checkpoint(checkpoint_path, device)
    cfg = ckpt["config"]
    scaler = TemperatureScaler()
    scaler.load_state_dict(ckpt["temp_scaler"])
    model_type = ckpt["model_type"]
    fcfg = cfg["featurize"]

    if model_type == "mlp":
        mcfg = cfg["mlp"]
        model = MLPRegressor(
            fcfg["morgan_n_bits"],
            list(mcfg["hidden_dims"]),
            dropout=mcfg["dropout"],
        ).to(device)
    elif model_type == "gnn":
        gcfg = cfg["gnn"]
        model = GNNRegressor(
            node_dim=atom_feature_dim(),
            edge_dim=EDGE_ATTR_DIM,
            hidden_dim=gcfg["hidden_dim"],
            num_layers=gcfg["num_layers"],
            dropout=gcfg["dropout"],
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type {model_type}")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    t_min = float(cfg.get("data_stats", {}).get("temperature_min", float("nan")))
    t_max = float(cfg.get("data_stats", {}).get("temperature_max", float("nan")))

    return _Predictor(model, scaler, model_type, device, fcfg, t_min, t_max)


class _Predictor:
    def __init__(self, model, scaler, model_type, device, fcfg, t_min, t_max) -> None:
        self.model = model
        self.scaler = scaler
        self.model_type = model_type
        self.device = device
        self.fcfg = fcfg
        self.t_min = t_min
        self.t_max = t_max

    @torch.no_grad()
    def predict_logd(self, smiles: str, temperature_k: float) -> float:
        if not np.isnan(self.t_min) and (temperature_k < self.t_min or temperature_k > self.t_max):
            warnings.warn(
                f"Temperature {temperature_k} K is outside training range "
                f"[{self.t_min:.2f}, {self.t_max:.2f}] K (extrapolation).",
                UserWarning,
                stacklevel=2,
            )
        t_s = self.scaler.transform(np.array([temperature_k], dtype=np.float64))[0]
        t_t = torch.tensor([t_s], dtype=torch.float32, device=self.device)

        if self.model_type == "mlp":
            fp = morgan_fingerprint_bitvect(
                smiles,
                self.fcfg["morgan_radius"],
                self.fcfg["morgan_n_bits"],
            )
            fp_t = torch.from_numpy(fp).unsqueeze(0).to(self.device)
            out = self.model(fp_t, t_t).squeeze().item()
        else:
            data = smiles_to_pyg_data(smiles).to(self.device)
            data.t_cond = t_t.reshape(1)
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=self.device)
            out = self.model(data).squeeze().item()
        return float(out)

    def predict_D(self, smiles: str, temperature_k: float) -> float:
        return float(inv_log10_d(self.predict_logd(smiles, temperature_k)))


def inject_data_stats_into_config(cfg: dict[str, Any], df) -> dict[str, Any]:
    out = dict(cfg)
    out["data_stats"] = {
        "temperature_min": float(df["Temperature"].min()),
        "temperature_max": float(df["Temperature"].max()),
    }
    return out
