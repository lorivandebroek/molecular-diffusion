"""Molecule-level random split and Bemis–Murcko scaffold split (no scaffold leakage)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

SplitName = Literal["train", "val", "test"]


def scaffold_smiles_from_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    core = MurckoScaffold.GetScaffoldForMol(mol)
    if core is None or core.GetNumAtoms() == 0:
        return smiles
    return Chem.MolToSmiles(core)


def _three_way_counts(n: int, train_frac: float, val_frac: float) -> tuple[int, int, int]:
    test_frac = 1.0 - train_frac - val_frac
    if not np.isclose(train_frac + val_frac + test_frac, 1.0, atol=1e-6):
        raise ValueError("train_frac + val_frac must be <= 1 and sum with test to 1")
    if n <= 0:
        return 0, 0, 0
    n_train = int(np.floor(train_frac * n))
    n_val = int(np.floor(val_frac * n))
    n_test = n - n_train - n_val
    if n_test <= 0 and n >= 3:
        n_test = 1
        if n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
    if n_train <= 0 and n >= 1:
        n_train = 1
        n_val = min(n_val, max(0, n - n_train - 1))
        n_test = n - n_train - n_val
    return n_train, n_val, n_test


def assign_molecule_random_split(
    unique_smiles: list[str],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> dict[str, SplitName]:
    rng = np.random.default_rng(seed)
    shuffled = list(unique_smiles)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train, n_val, _n_test = _three_way_counts(n, train_frac, val_frac)
    splits: dict[str, SplitName] = {}
    for i, smi in enumerate(shuffled):
        if i < n_train:
            splits[smi] = "train"
        elif i < n_train + n_val:
            splits[smi] = "val"
        else:
            splits[smi] = "test"
    return splits


def assign_scaffold_split(
    unique_smiles: list[str],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> dict[str, SplitName]:
    """Assign each unique scaffold to exactly one split; map molecules by scaffold."""
    scaffold_for_smiles: dict[str, str] = {}
    scaffold_to_smiles: dict[str, list[str]] = {}
    for smi in unique_smiles:
        sc = scaffold_smiles_from_smiles(smi)
        scaffold_for_smiles[smi] = sc
        scaffold_to_smiles.setdefault(sc, []).append(smi)

    scaffolds = list(scaffold_to_smiles.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(scaffolds)
    n = len(scaffolds)
    if n == 0:
        return {}

    n_train, n_val, _n_test = _three_way_counts(n, train_frac, val_frac)
    train_sc = set(scaffolds[:n_train])
    val_sc = set(scaffolds[n_train : n_train + n_val])
    test_sc = set(scaffolds[n_train + n_val :])
    if not test_sc and n >= 2:
        move = next(iter(train_sc), None)
        if move is not None and len(train_sc) > 1:
            train_sc.remove(move)
            test_sc.add(move)

    splits: dict[str, SplitName] = {}
    for smi in unique_smiles:
        sc = scaffold_for_smiles[smi]
        if sc in train_sc:
            splits[smi] = "train"
        elif sc in val_sc:
            splits[smi] = "val"
        else:
            splits[smi] = "test"
    return splits


def add_split_column(
    df: pd.DataFrame,
    smiles_col: str,
    mode: Literal["random", "scaffold"],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> pd.DataFrame:
    unique = df[smiles_col].unique().tolist()
    if mode == "random":
        mapping = assign_molecule_random_split(unique, train_frac, val_frac, seed)
    else:
        mapping = assign_scaffold_split(unique, train_frac, val_frac, seed)
    out = df.copy()
    out["split"] = out[smiles_col].map(mapping)
    if out["split"].isna().any():
        raise RuntimeError("Some rows missing split assignment")
    return out


def save_split_manifest(
    df: pd.DataFrame,
    smiles_col: str,
    path: str | Path,
    *,
    mode: str,
    seed: int,
    train_frac: float,
    val_frac: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    unique = df[[smiles_col, "split"]].drop_duplicates(subset=[smiles_col])
    manifest: dict[str, Any] = {
        "mode": mode,
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": 1.0 - train_frac - val_frac,
        "n_molecules": int(unique.shape[0]),
        "assignments": unique.set_index(smiles_col)["split"].astype(str).to_dict(),
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_split_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
