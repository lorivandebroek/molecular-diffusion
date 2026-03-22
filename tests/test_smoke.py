"""Import and minimal clean pipeline."""

import pandas as pd

from diffusion_mol.clean import clean_dataframe


def test_clean_drops_invalid_and_dedupes():
    df = pd.DataFrame(
        {
            "smiles_raw": ["CC", "not_a_valid_smiles_string", "CC"],
            "Temperature": [298.0, 298.0, 298.0],
            "D": [1e-5, 1e-5, 2e-5],
        }
    )
    out, rep = clean_dataframe(df, dedupe_agg="mean")
    assert rep["rows_invalid_smiles"] >= 1
    assert "smiles" in out.columns
    assert (out["D"] > 0).all()
