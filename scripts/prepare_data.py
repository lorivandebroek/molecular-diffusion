#!/usr/bin/env python3
"""Load Excel Table S2, clean, write data/processed/cleaned.csv + data_report.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from diffusion_mol.clean import clean_dataframe
from diffusion_mol.io_excel import load_table_s2


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--excel", type=Path, default=ROOT / "jp5c01881_si_002.xlsx")
    p.add_argument("--out-csv", type=Path, default=ROOT / "data" / "processed" / "cleaned.csv")
    p.add_argument("--out-report", type=Path, default=ROOT / "data" / "processed" / "data_report.json")
    args = p.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    raw = load_table_s2(args.excel)
    cleaned, report = clean_dataframe(raw, dedupe_agg="mean")
    cleaned.to_csv(args.out_csv, index=False)
    args.out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_csv} ({len(cleaned)} rows)")
    print(f"Wrote {args.out_report}")


if __name__ == "__main__":
    main()
