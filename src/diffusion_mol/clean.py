"""Canonicalize SMILES, drop invalid structures, deduplicate (smiles, T)."""

from __future__ import annotations

from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def canonicalize_smiles(smiles: str | None) -> str | None:
    if smiles is None or (isinstance(smiles, str) and not str(smiles).strip()):
        return None
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def clean_dataframe(
    df: pd.DataFrame,
    *,
    dedupe_agg: str = "mean",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Expect columns: smiles_raw, Temperature, D (plus optional metadata).

    Deduplicate (canonical_smiles, Temperature): default aggregate D by mean.
    Returns (cleaned_df, report_dict).
    """
    required = {"smiles_raw", "Temperature", "D"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    work = df.copy()
    work["canonical_smiles"] = work["smiles_raw"].map(canonicalize_smiles)
    before = len(work)
    invalid = work["canonical_smiles"].isna().sum()
    work = work.dropna(subset=["canonical_smiles"])
    work = work.drop(columns=["smiles_raw"])
    work = work.rename(columns={"canonical_smiles": "smiles"})

    numeric_cols = [c for c in work.columns if c not in ("smiles",)]
    for c in numeric_cols:
        if c in ("Temperature", "D"):
            work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=["Temperature", "D"])
    if dedupe_agg == "mean":
        gb = work.groupby(["smiles", "Temperature"], as_index=False)
        d_part = gb["D"].mean()
        others = [c for c in work.columns if c not in ("smiles", "Temperature", "D")]
        if others:
            first_part = gb[others].first()
            grouped = d_part.merge(first_part, on=["smiles", "Temperature"])
        else:
            grouped = d_part
    elif dedupe_agg == "first":
        grouped = work.drop_duplicates(subset=["smiles", "Temperature"], keep="first")
    else:
        raise ValueError(f"Unknown dedupe_agg: {dedupe_agg}")

    after_dedupe = len(grouped)
    report: dict[str, Any] = {
        "rows_in": int(before),
        "rows_invalid_smiles": int(invalid),
        "rows_after_drop_invalid": int(len(work)),
        "rows_after_dedupe": int(after_dedupe),
        "dedupe_pairs_collapsed": int(len(work) - after_dedupe),
        "temperature_min": float(grouped["Temperature"].min()),
        "temperature_max": float(grouped["Temperature"].max()),
        "n_unique_smiles": int(grouped["smiles"].nunique()),
        "dedupe_policy": dedupe_agg,
    }
    return grouped.reset_index(drop=True), report
