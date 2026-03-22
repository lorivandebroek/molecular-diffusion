"""Load Table S2 from the Han et al. supporting information Excel file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TABLE_S2_SHEET = "Table S2"
SMILES_COL_RAW = "SMILSES"


def load_table_s2(excel_path: str | Path) -> pd.DataFrame:
    path = Path(excel_path)
    if not path.is_file():
        raise FileNotFoundError(f"Excel not found: {path.resolve()}")
    df = pd.read_excel(path, sheet_name=TABLE_S2_SHEET, engine="openpyxl")
    if SMILES_COL_RAW not in df.columns:
        raise ValueError(
            f"Expected column {SMILES_COL_RAW!r} in sheet {TABLE_S2_SHEET!r}; got {list(df.columns)}"
        )
    df = df.rename(columns={SMILES_COL_RAW: "smiles_raw"})
    return df
