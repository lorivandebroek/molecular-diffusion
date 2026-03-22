"""PyTorch datasets for fingerprint-MLP and PyG batches."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from diffusion_mol.featurize import morgan_fingerprint_bitvect, smiles_to_pyg_data


class FingerprintDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        smiles_col: str,
        t_scaled: np.ndarray,
        y_logd: np.ndarray,
        *,
        radius: int,
        n_bits: int,
    ) -> None:
        self.smiles = df[smiles_col].tolist()
        self.t = torch.tensor(t_scaled, dtype=torch.float32)
        self.y = torch.tensor(y_logd, dtype=torch.float32)
        self.radius = radius
        self.n_bits = n_bits
        self._cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.smiles)

    def _fp(self, smi: str) -> np.ndarray:
        if smi not in self._cache:
            self._cache[smi] = morgan_fingerprint_bitvect(smi, self.radius, self.n_bits)
        return self._cache[smi]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fp = torch.from_numpy(self._fp(self.smiles[idx]))
        t = self.t[idx].reshape(1)
        y = self.y[idx].reshape(1)
        return fp, t, y


class PyGMoleculeDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        smiles_col: str,
        t_scaled: np.ndarray,
        y_logd: np.ndarray,
    ) -> None:
        self.smiles = df[smiles_col].tolist()
        self.t = torch.tensor(t_scaled, dtype=torch.float32)
        self.y = torch.tensor(y_logd, dtype=torch.float32)
        self._graphs: dict[str, Data] = {}

    def __len__(self) -> int:
        return len(self.smiles)

    def _graph(self, smi: str) -> Data:
        if smi not in self._graphs:
            self._graphs[smi] = smiles_to_pyg_data(smi)
        return self._graphs[smi].clone()

    def __getitem__(self, idx: int) -> Data:
        data = self._graph(self.smiles[idx])
        data.t_cond = self.t[idx].reshape(1)
        data.y = self.y[idx].reshape(1)
        return data


def collate_identity(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    fps = torch.stack([b[0] for b in batch])
    ts = torch.stack([b[1] for b in batch])
    ys = torch.stack([b[2] for b in batch])
    return fps, ts, ys


def make_pyg_loader(dataset: PyGMoleculeDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
