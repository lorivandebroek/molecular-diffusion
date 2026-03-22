"""Train MLP or GNN for log10(D) prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
import yaml

from diffusion_mol.datasets import FingerprintDataset, PyGMoleculeDataset, collate_identity
from diffusion_mol.featurize import atom_feature_dim, EDGE_ATTR_DIM
from diffusion_mol.models.gnn import GNNRegressor
from diffusion_mol.models.mlp import MLPRegressor
from diffusion_mol.scaling import TemperatureScaler, log10_d


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_arrays(df, smiles_col: str = "smiles"):
    y_log = log10_d(df["D"].values)
    return y_log


@torch.no_grad()
def predict_mlp(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: list[float] = []
    preds: list[float] = []
    for fp, t, y in loader:
        fp = fp.to(device)
        t = t.to(device).squeeze(-1)
        out = model(fp, t).squeeze(-1)
        preds.extend(out.cpu().numpy().tolist())
        ys.extend(y.squeeze(-1).cpu().numpy().tolist())
    return np.array(ys), np.array(preds)


@torch.no_grad()
def predict_gnn(model: nn.Module, loader: PyGDataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: list[float] = []
    preds: list[float] = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch).squeeze(-1)
        ys.extend(batch.y.view(-1).cpu().numpy().tolist())
        preds.extend(out.cpu().numpy().tolist())
    return np.array(ys), np.array(preds)


def train_mlp(
    train_df,
    val_df,
    cfg: dict[str, Any],
    device: torch.device,
    temp_scaler: TemperatureScaler,
) -> tuple[MLPRegressor, dict[str, Any]]:
    fcfg = cfg["featurize"]
    tcfg = cfg["training"]
    mcfg = cfg["mlp"]

    y_tr = prepare_arrays(train_df)
    y_va = prepare_arrays(val_df)
    t_tr = temp_scaler.transform(train_df["Temperature"].values)
    t_va = temp_scaler.transform(val_df["Temperature"].values)

    train_ds = FingerprintDataset(
        train_df.reset_index(drop=True),
        "smiles",
        t_tr,
        y_tr,
        radius=fcfg["morgan_radius"],
        n_bits=fcfg["morgan_n_bits"],
    )
    val_ds = FingerprintDataset(
        val_df.reset_index(drop=True),
        "smiles",
        t_va,
        y_va,
        radius=fcfg["morgan_radius"],
        n_bits=fcfg["morgan_n_bits"],
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        collate_fn=collate_identity,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tcfg["batch_size"],
        shuffle=False,
        collate_fn=collate_identity,
    )

    fp_dim = fcfg["morgan_n_bits"]
    model = MLPRegressor(fp_dim, list(mcfg["hidden_dims"]), dropout=mcfg["dropout"]).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["lr"],
        weight_decay=tcfg["weight_decay"],
    )
    criterion = nn.HuberLoss(delta=tcfg.get("huber_delta", 1.0))

    best_state = None
    best_val = float("inf")
    patience = tcfg.get("patience", 25)
    bad = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(tcfg.get("max_epochs", 200)):
        model.train()
        losses = []
        for fp, t, y in train_loader:
            fp = fp.to(device)
            t = t.to(device).squeeze(-1)
            y = y.to(device).squeeze(-1)
            opt.zero_grad()
            pred = model(fp, t).squeeze(-1)
            loss = criterion(pred, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        val_losses = []
        with torch.no_grad():
            for fp, t, y in val_loader:
                fp = fp.to(device)
                t = t.to(device).squeeze(-1)
                y = y.to(device).squeeze(-1)
                pred = model(fp, t).squeeze(-1)
                val_losses.append(criterion(pred, y).item())
        tr_m = float(np.mean(losses)) if losses else 0.0
        va_m = float(np.mean(val_losses)) if val_losses else 0.0
        history["train_loss"].append(tr_m)
        history["val_loss"].append(va_m)
        if va_m < best_val - 1e-6:
            best_val = va_m
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    meta = {"best_val_loss": best_val, "epochs": len(history["train_loss"]), "history": history}
    return model, meta


def train_gnn(
    train_df,
    val_df,
    cfg: dict[str, Any],
    device: torch.device,
    temp_scaler: TemperatureScaler,
) -> tuple[GNNRegressor, dict[str, Any]]:
    tcfg = cfg["training"]
    gcfg = cfg["gnn"]

    y_tr = prepare_arrays(train_df)
    y_va = prepare_arrays(val_df)
    t_tr = temp_scaler.transform(train_df["Temperature"].values)
    t_va = temp_scaler.transform(val_df["Temperature"].values)

    train_ds = PyGMoleculeDataset(train_df.reset_index(drop=True), "smiles", t_tr, y_tr)
    val_ds = PyGMoleculeDataset(val_df.reset_index(drop=True), "smiles", t_va, y_va)
    train_loader = PyGDataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True)
    val_loader = PyGDataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False)

    node_dim = atom_feature_dim()
    model = GNNRegressor(
        node_dim=node_dim,
        edge_dim=EDGE_ATTR_DIM,
        hidden_dim=gcfg["hidden_dim"],
        num_layers=gcfg["num_layers"],
        dropout=gcfg["dropout"],
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["lr"],
        weight_decay=tcfg["weight_decay"],
    )
    criterion = nn.HuberLoss(delta=tcfg.get("huber_delta", 1.0))

    best_state = None
    best_val = float("inf")
    patience = tcfg.get("patience", 25)
    bad = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(tcfg.get("max_epochs", 200)):
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(device)
            y = batch.y.view(-1)
            opt.zero_grad()
            pred = model(batch).view(-1)
            loss = criterion(pred, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                y = batch.y.view(-1)
                pred = model(batch).view(-1)
                val_losses.append(criterion(pred, y).item())
        tr_m = float(np.mean(losses)) if losses else 0.0
        va_m = float(np.mean(val_losses)) if val_losses else 0.0
        history["train_loss"].append(tr_m)
        history["val_loss"].append(va_m)
        if va_m < best_val - 1e-6:
            best_val = va_m
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    meta = {"best_val_loss": best_val, "epochs": len(history["train_loss"]), "history": history}
    return model, meta


def save_checkpoint(
    path: str | Path,
    *,
    model_type: str,
    model_state: dict,
    cfg: dict[str, Any],
    temp_scaler: TemperatureScaler,
    train_meta: dict[str, Any],
    split_mode: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": model_type,
        "model_state": {k: v.cpu() for k, v in model_state.items()},
        "config": cfg,
        "temp_scaler": temp_scaler.state_dict(),
        "train_meta": train_meta,
        "split_mode": split_mode,
    }
    torch.save(payload, path)
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_type": model_type,
                "split_mode": split_mode,
                "best_val_loss": train_meta.get("best_val_loss"),
                "epochs": train_meta.get("epochs"),
            },
            f,
            indent=2,
        )


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)
