#!/usr/bin/env python3
"""Train MLP or GNN with molecule-level split; save checkpoint and test metrics."""

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
from diffusion_mol.predict import inject_data_stats_into_config
from diffusion_mol.scaling import TemperatureScaler
from diffusion_mol.splits import add_split_column, save_split_manifest
from diffusion_mol.train import (
    load_config,
    predict_gnn,
    predict_mlp,
    save_checkpoint,
    train_gnn,
    train_mlp,
)


def _fp_matrix(df: pd.DataFrame, radius: int, n_bits: int) -> np.ndarray:
    return np.stack([morgan_fingerprint_bitvect(s, radius, n_bits) for s in df["smiles"]], axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    ap.add_argument("--split", choices=["random", "scaffold"], required=True)
    ap.add_argument("--model", choices=["mlp", "gnn"], required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    sp = cfg["split"]
    csv_path = Path(paths["processed_csv"])
    if not csv_path.is_file():
        raise SystemExit(f"Missing {csv_path}; run scripts/prepare_data.py first")
    df = pd.read_csv(csv_path)

    df = add_split_column(
        df,
        "smiles",
        mode=args.split,
        train_frac=sp["train_frac"],
        val_frac=sp["val_frac"],
        seed=sp["seed"],
    )
    art = Path(paths.get("artifacts_dir", "artifacts"))
    split_dir = art / "splits"
    save_split_manifest(
        df,
        "smiles",
        split_dir / f"{args.split}.json",
        mode=args.split,
        seed=sp["seed"],
        train_frac=sp["train_frac"],
        val_frac=sp["val_frac"],
    )

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    temp_scaler = TemperatureScaler()
    temp_scaler.fit(train_df["Temperature"].values)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_full = inject_data_stats_into_config(cfg, train_df)

    if args.model == "mlp":
        model, meta = train_mlp(train_df, val_df, cfg, device, temp_scaler)
        test_ds = FingerprintDataset(
            test_df.reset_index(drop=True),
            "smiles",
            temp_scaler.transform(test_df["Temperature"].values),
            np.log10(test_df["D"].values),
            radius=cfg["featurize"]["morgan_radius"],
            n_bits=cfg["featurize"]["morgan_n_bits"],
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            collate_fn=collate_identity,
        )
        y_true, y_pred = predict_mlp(model, test_loader, device)
    else:
        model, meta = train_gnn(train_df, val_df, cfg, device, temp_scaler)
        test_ds = PyGMoleculeDataset(
            test_df.reset_index(drop=True),
            "smiles",
            temp_scaler.transform(test_df["Temperature"].values),
            np.log10(test_df["D"].values),
        )
        test_loader = PyGDataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)
        y_true, y_pred = predict_gnn(model, test_loader, device)

    metrics_nn = regression_metrics(y_true, y_pred)

    y_train_log = np.log10(train_df["D"].values)
    y_test_log = np.log10(test_df["D"].values)
    mean_pred = predict_constant(float(np.mean(y_train_log)), len(test_df))
    med_pred = predict_constant(float(np.median(y_train_log)), len(test_df))
    metrics_mean = regression_metrics(y_test_log, mean_pred)
    metrics_med = regression_metrics(y_test_log, med_pred)

    X_tr = _fp_matrix(train_df, cfg["featurize"]["morgan_radius"], cfg["featurize"]["morgan_n_bits"])
    X_te = _fp_matrix(test_df, cfg["featurize"]["morgan_radius"], cfg["featurize"]["morgan_n_bits"])
    t_tr = temp_scaler.transform(train_df["Temperature"].values)
    t_te = temp_scaler.transform(test_df["Temperature"].values)
    ridge_pred, _ = fit_eval_ridge(X_tr, t_tr, y_train_log, X_te, t_te)
    metrics_ridge = regression_metrics(y_test_log, ridge_pred)

    out_dir = art / "models"
    ckpt_path = out_dir / f"{args.model}_{args.split}.pt"
    save_checkpoint(
        ckpt_path,
        model_type=args.model,
        model_state=model.state_dict(),
        cfg=cfg_full,
        temp_scaler=temp_scaler,
        train_meta=meta,
        split_mode=args.split,
    )

    summary = {
        "split": args.split,
        "model": args.model,
        "device": str(device),
        "test_metrics": {
            args.model: metrics_to_json_serializable(metrics_nn),
            "mean_baseline": metrics_to_json_serializable(metrics_mean),
            "median_baseline": metrics_to_json_serializable(metrics_med),
            "ridge_fp": metrics_to_json_serializable(metrics_ridge),
        },
        "train_meta": {"best_val_loss": meta.get("best_val_loss"), "epochs": meta.get("epochs")},
    }
    metrics_path = art / f"metrics_{args.model}_{args.split}.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["test_metrics"], indent=2))
    print(f"Checkpoint: {ckpt_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
