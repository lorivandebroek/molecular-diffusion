"""Morgan fingerprints and RDKit -> PyG molecular graphs."""

from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")

EDGE_ATTR_DIM = 4


def morgan_fingerprint_bitvect(smiles: str, radius: int, n_bits: int) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(vec, arr)
    return arr


def _one_hot(idx: int, size: int) -> list[float]:
    v = [0.0] * size
    if 0 <= idx < size:
        v[idx] = 1.0
    return v


def mol_to_pyg_data(mol: Chem.Mol) -> Data:
    """Atom/bond features suitable for GINEConv (edge_attr dim = 4 bond type one-hot)."""
    mol = Chem.AddHs(mol)

    atom_feats: list[list[float]] = []
    for atom in mol.GetAtoms():
        hyb = atom.GetHybridization()
        hyb_order = [
            Chem.HybridizationType.S,
            Chem.HybridizationType.SP,
            Chem.HybridizationType.SP2,
            Chem.HybridizationType.SP3,
            Chem.HybridizationType.SP3D,
            Chem.HybridizationType.SP3D2,
            Chem.HybridizationType.UNSPECIFIED,
        ]
        try:
            hyb_idx = hyb_order.index(hyb)
        except ValueError:
            hyb_idx = len(hyb_order) - 1
        feat = [
            float(atom.GetAtomicNum()) / 100.0,
            float(atom.GetTotalDegree()) / 6.0,
            float(atom.GetFormalCharge()),
            float(atom.GetTotalNumHs()) / 4.0,
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
        ] + _one_hot(hyb_idx, len(hyb_order))
        atom_feats.append(feat)

    x = torch.tensor(atom_feats, dtype=torch.float)

    row: list[int] = []
    col: list[int] = []
    edge_attr: list[list[float]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        if bt == Chem.BondType.SINGLE:
            b = [1.0, 0.0, 0.0, 0.0]
        elif bt == Chem.BondType.DOUBLE:
            b = [0.0, 1.0, 0.0, 0.0]
        elif bt == Chem.BondType.TRIPLE:
            b = [0.0, 0.0, 1.0, 0.0]
        elif bt == Chem.BondType.AROMATIC:
            b = [0.0, 0.0, 0.0, 1.0]
        else:
            b = [0.25, 0.25, 0.25, 0.25]
        row += [i, j]
        col += [j, i]
        edge_attr += [b, b]

    if len(row) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        ea = torch.zeros((0, EDGE_ATTR_DIM), dtype=torch.float)
    else:
        edge_index = torch.tensor([row, col], dtype=torch.long)
        ea = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=ea)


def smiles_to_pyg_data(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol_to_pyg_data(mol)


def atom_feature_dim() -> int:
    return smiles_to_pyg_data("CC").x.size(1)
