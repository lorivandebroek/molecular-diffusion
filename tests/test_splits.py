import pandas as pd

from diffusion_mol.splits import (
    add_split_column,
    assign_molecule_random_split,
    assign_scaffold_split,
    scaffold_smiles_from_smiles,
)


def test_random_split_partitions_molecules():
    unique = ["CC", "CCC", "CCCC", "CCO", "O"]
    m = assign_molecule_random_split(unique, 0.6, 0.2, seed=0)
    assert set(m.keys()) == set(unique)
    parts = set(m.values())
    assert parts <= {"train", "val", "test"}


def test_scaffold_benzene_toluene_same_split_component():
    bz = "c1ccccc1"
    tol = "Cc1ccccc1"
    sc_bz = scaffold_smiles_from_smiles(bz)
    sc_tol = scaffold_smiles_from_smiles(tol)
    assert sc_bz == sc_tol
    m = assign_scaffold_split([bz, tol, "CC"], 0.34, 0.33, seed=42)
    assert m[bz] == m[tol]


def test_add_split_column_no_nan():
    df = pd.DataFrame(
        {
            "smiles": ["CC"] * 3 + ["O"] * 3,
            "Temperature": [273, 298, 310] * 2,
            "D": [1e-5] * 6,
        }
    )
    out = add_split_column(df, "smiles", "random", 0.5, 0.25, seed=1)
    assert not out["split"].isna().any()
