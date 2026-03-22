#!/usr/bin/env python3
"""Evaluate baselines + optional trained models; write benchmark metrics and plots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from diffusion_mol.baselines import fit_eval_ridge, predict_constant
from diffusion_mol.datasets import FingerprintDataset, PyGMoleculeDataset, collate_identity
from diffusion_mol.evaluate import metrics_to_json_serializable, regression_metrics
from diffusion_mol.featurize import morgan_fingerprint_bitvect
from diffusion_mol.models.gnn import GNNRegressor
from diffusion_mol.models.mlp import MLPRegressor
from diffusion_mol.plot import plot_benchmark_bars, plot_parity_log, plot_residuals_vs_temperature
from diffusion_mol.scaling import TemperatureScaler
from diffusion_mol.splits import add_split_column
from diffusion_mol.train import load_checkpoint, load_config, predict_gnn, predict_mlp
from diffusion_mol.featurize import EDGE_ATTR_DIM, atom_feature_dim


def _fp_matrix(df: pd.DataFrame, radius: int, n_bits: int) -> np.ndarray:
    return np.stack([morgan_fingerprint_bitvect(s, radius, n_bits) for s in df["smiles"]], axis=0)


def _load_nn(split: str, model: str, art: Path, device: torch.device):
    path = art / "models" / f"{model}_{split}.pt"
    if not path.is_file():
        return None, None, None
    ckpt = load_checkpoint(path, device)
    temp_scaler = TemperatureScaler()
    temp_scaler.load_state_dict(ckpt["temp_scaler"])
    mcfg = ckpt["config"]
    fcfg = mcfg["featurize"]
    if model == "mlp":
        hcfg = mcfg["mlp"]
        net = MLPRegressor(
            fcfg["morgan_n_bits"],
            list(hcfg["hidden_dims"]),
            dropout=hcfg["dropout"],
        ).to(device)
    else:
        gcfg = mcfg["gnn"]
        net = GNNRegressor(
            node_dim=atom_feature_dim(),
            edge_dim=EDGE_ATTR_DIM,
            hidden_dim=gcfg["hidden_dim"],
            num_layers=gcfg["num_layers"],
            dropout=gcfg["dropout"],
        ).to(device)
    net.load_state_dict(ckpt["model_state"])
    return net, temp_scaler, mcfg


def run_for_split(
    df: pd.DataFrame,
    split_name: str,
    cfg: dict,
    art: Path,
    device: torch.device,
) -> dict:
    sp = cfg["split"]
    df = add_split_column(
        df,
        "smiles",
        mode=split_name,
        train_frac=sp["train_frac"],
        val_frac=sp["val_frac"],
        seed=sp["seed"],
    )
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    temp_scaler = TemperatureScaler()
    temp_scaler.fit(train_df["Temperature"].values)

    y_train_log = np.log10(train_df["D"].values)
    y_test_log = np.log10(test_df["D"].values)
    t_test = test_df["Temperature"].values

    bench: dict[str, dict] = {}
    bench["mean_baseline"] = metrics_to_json_serializable(
        regression_metrics(y_test_log, predict_constant(float(np.mean(y_train_log)), len(test_df)))
    )
    bench["median_baseline"] = metrics_to_json_serializable(
        regression_metrics(y_test_log, predict_constant(float(np.median(y_train_log)), len(test_df)))
    )
    X_tr = _fp_matrix(train_df, cfg["featurize"]["morgan_radius"], cfg["featurize"]["morgan_n_bits"])
    X_te = _fp_matrix(test_df, cfg["featurize"]["morgan_radius"], cfg["featurize"]["morgan_n_bits"])
    t_tr = temp_scaler.transform(train_df["Temperature"].values)
    t_te = temp_scaler.transform(test_df["Temperature"].values)
    ridge_p, _ = fit_eval_ridge(X_tr, t_tr, y_train_log, X_te, t_te)
    bench["ridge_fp"] = metrics_to_json_serializable(regression_metrics(y_test_log, ridge_p))

    preds_for_plots: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "ridge_fp": (y_test_log, ridge_p),
    }

    for model_name in ("mlp", "gnn"):
        net, sc, mcfg = _load_nn(split_name, model_name, art, device)
        if net is None or sc is None or mcfg is None:
            continue
        fcfg = mcfg["featurize"]
        batch_size = mcfg["training"]["batch_size"]
        if model_name == "mlp":
            test_ds = FingerprintDataset(
                test_df.reset_index(drop=True),
                "smiles",
                sc.transform(test_df["Temperature"].values),
                y_test_log,
                radius=fcfg["morgan_radius"],
                n_bits=fcfg["morgan_n_bits"],
            )
            loader = DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_identity,
            )
            yt, yp = predict_mlp(net, loader, device)
        else:
            test_ds = PyGMoleculeDataset(
                test_df.reset_index(drop=True),
                "smiles",
                sc.transform(test_df["Temperature"].values),
                y_test_log,
            )
            loader = PyGDataLoader(test_ds, batch_size=batch_size, shuffle=False)
            yt, yp = predict_gnn(net, loader, device)
        bench[model_name] = metrics_to_json_serializable(regression_metrics(yt, yp))
        preds_for_plots[model_name] = (yt, yp)

    fig_dir = art / "figures" / split_name
    fig_dir.mkdir(parents=True, exist_ok=True)
    for name, (yt, yp) in preds_for_plots.items():
        plot_parity_log(yt, yp, fig_dir / f"parity_{name}.png", title=f"{split_name} test: {name}")
        plot_residuals_vs_temperature(
            t_test,
            yt,
            yp,
            fig_dir / f"residuals_T_{name}.png",
            title=f"{split_name} test: {name}",
        )

    rmse_map = {k: float(v["rmse_log10"]) for k, v in bench.items()}
    plot_benchmark_bars(rmse_map, fig_dir / "benchmark_rmse_log10.png", title=f"{split_name} test RMSE log10(D)")

    return {"split": split_name, "benchmarks": bench}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    ap.add_argument("--splits", nargs="+", default=["random", "scaffold"], choices=["random", "scaffold"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    csv_path = Path(paths["processed_csv"])
    if not csv_path.is_file():
        raise SystemExit(f"Missing {csv_path}; run scripts/prepare_data.py first")
    df = pd.read_csv(csv_path)
    art = Path(paths.get("artifacts_dir", "artifacts"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out = {"splits": [], "device": str(device)}
    for s in args.splits:
        out["splits"].append(run_for_split(df, s, cfg, art, device))

    out_path = art / "benchmark_summary.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
