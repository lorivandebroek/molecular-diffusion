import pandas as pd
import pytest
import torch

torch_geometric = pytest.importorskip("torch_geometric")

from diffusion_mol.scaling import TemperatureScaler
from diffusion_mol.train import load_config, train_mlp, train_gnn


@pytest.fixture
def tiny_cfg():
    return {
        "featurize": {"morgan_radius": 2, "morgan_n_bits": 128},
        "training": {
            "batch_size": 8,
            "lr": 0.01,
            "weight_decay": 0.0,
            "max_epochs": 3,
            "patience": 5,
            "huber_delta": 1.0,
        },
        "mlp": {"hidden_dims": [32, 16], "dropout": 0.0},
        "gnn": {"hidden_dim": 32, "num_layers": 2, "dropout": 0.0},
    }


def _tiny_df():
    smiles = ["CC", "CCC", "CCCC", "CCO", "O", "CO"]
    rows = []
    for s in smiles:
        for t in (273.0, 298.0):
            rows.append({"smiles": s, "Temperature": t, "D": 1e-5 * (1 + 0.01 * hash(s) % 5)})
    return pd.DataFrame(rows)


def test_mlp_runs_short_training(tiny_cfg):
    df = _tiny_df()
    tr = df.iloc[:8].copy()
    va = df.iloc[8:12].copy()
    scaler = TemperatureScaler()
    scaler.fit(tr["Temperature"].values)
    device = torch.device("cpu")
    model, meta = train_mlp(tr, va, tiny_cfg, device, scaler)
    assert meta["epochs"] >= 1
    assert model is not None


def test_gnn_runs_short_training(tiny_cfg):
    df = _tiny_df()
    tr = df.iloc[:8].copy()
    va = df.iloc[8:12].copy()
    scaler = TemperatureScaler()
    scaler.fit(tr["Temperature"].values)
    device = torch.device("cpu")
    model, meta = train_gnn(tr, va, tiny_cfg, device, scaler)
    assert meta["epochs"] >= 1
