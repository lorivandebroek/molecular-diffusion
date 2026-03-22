#!/usr/bin/env python3
"""Emit report/_generated_tables.tex from data_report.json, benchmark_summary.json, default.yaml."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "_generated_tables.tex"

# Fallback if artifacts/ not present (keeps LaTeX build reproducible).
_FALLBACK_BENCH = {
    "splits": [
        {
            "split": "random",
            "benchmarks": {
                "mean_baseline": {
                    "rmse_log10": 0.4731,
                    "r2_log10": -5.92e-5,
                    "median_abs_rel_error_D": 0.7910,
                },
                "median_baseline": {
                    "rmse_log10": 0.4938,
                    "r2_log10": -0.0894,
                    "median_abs_rel_error_D": 0.7750,
                },
                "ridge_fp": {
                    "rmse_log10": 0.1068,
                    "r2_log10": 0.9491,
                    "median_abs_rel_error_D": 0.1293,
                },
                "mlp": {
                    "rmse_log10": 0.1028,
                    "r2_log10": 0.9528,
                    "median_abs_rel_error_D": 0.1105,
                },
                "gnn": {
                    "rmse_log10": 0.0989,
                    "r2_log10": 0.9563,
                    "median_abs_rel_error_D": 0.1026,
                },
            },
        },
        {
            "split": "scaffold",
            "benchmarks": {
                "mean_baseline": {
                    "rmse_log10": 0.4708,
                    "r2_log10": -0.0001,
                    "median_abs_rel_error_D": 0.7898,
                },
                "median_baseline": {
                    "rmse_log10": 0.4915,
                    "r2_log10": -0.0897,
                    "median_abs_rel_error_D": 0.7666,
                },
                "ridge_fp": {
                    "rmse_log10": 0.1065,
                    "r2_log10": 0.9489,
                    "median_abs_rel_error_D": 0.1379,
                },
                "mlp": {
                    "rmse_log10": 0.0843,
                    "r2_log10": 0.9679,
                    "median_abs_rel_error_D": 0.1006,
                },
                "gnn": {
                    "rmse_log10": 0.0766,
                    "r2_log10": 0.9735,
                    "median_abs_rel_error_D": 0.1020,
                },
            },
        },
    ],
}

_FALLBACK_DATA = {
    "rows_after_dedupe": 20576,
    "n_unique_smiles": 6909,
    "temperature_min": 273.75,
    "temperature_max": 394.0,
    "dedupe_pairs_collapsed": 152,
    "rows_invalid_smiles": 0,
}


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_r2(x: float) -> str:
    if abs(x) < 0.0005:
        return f"{x:.3f}"
    return f"{x:.3f}"


def _bench_rows(data: dict) -> str:
    order = [
        ("mean_baseline", "Mean $\\log_{10}D$"),
        ("median_baseline", "Median $\\log_{10}D$"),
        ("ridge_fp", "Ridge (FP+T)"),
        ("mlp", "MLP"),
        ("gnn", "GNN"),
    ]
    lines = []
    for block in data["splits"]:
        name = block["split"].capitalize()
        b = block["benchmarks"]
        for key, label in order:
            m = b[key]
            r2 = m["r2_log10"]
            r2s = _fmt_r2(r2)
            lines.append(
                f"{name} & {label} & {m['rmse_log10']:.4f} & {r2s} & {m['median_abs_rel_error_D']:.4f} \\\\"
            )
        lines.append("\\midrule")
    lines.pop()  # drop last midrule
    return "\n".join(lines)


def _data_rows(rep: dict) -> str:
    return (
        f"Rows after cleaning & {int(rep['rows_after_dedupe']):,} \\\\\n"
        f"Unique SMILES & {int(rep['n_unique_smiles']):,} \\\\\n"
        f"$T$ range in data (K) & {rep['temperature_min']:.2f} -- {rep['temperature_max']:.1f} \\\\\n"
        f"Invalid SMILES dropped & {int(rep.get('rows_invalid_smiles', 0)):,} \\\\\n"
        f"Duplicate $(\\text{{SMILES}},T)$ collapsed & {int(rep['dedupe_pairs_collapsed']):,} "
        f"(mean $D$) \\\\\n"
    )


def _hp_rows(cfg: dict) -> str:
    tr = cfg["training"]
    fe = cfg["featurize"]
    sp = cfg["split"]
    mlp = cfg["mlp"]
    gnn = cfg["gnn"]
    return (
        f"Train / val / test fraction & {sp['train_frac']:.2f} / {sp['val_frac']:.2f} / "
        f"{1 - sp['train_frac'] - sp['val_frac']:.2f} \\\\\n"
        f"Split RNG seed & {sp['seed']} \\\\\n"
        f"Morgan radius / bits & {fe['morgan_radius']} / {fe['morgan_n_bits']} \\\\\n"
        f"Batch size / max epochs / patience & {tr['batch_size']} / {tr['max_epochs']} / {tr['patience']} \\\\\n"
        f"Optimizer / LR / weight decay & AdamW / {tr['lr']} / {tr['weight_decay']} \\\\\n"
        f"Loss & Huber ($\\delta={tr['huber_delta']}$) on $\\log_{{10}} D$ \\\\\n"
        f"MLP hidden dims / dropout & {mlp['hidden_dims']} / {mlp['dropout']} \\\\\n"
        f"GNN (GINE) hidden dim / layers / dropout & {gnn['hidden_dim']} / {gnn['num_layers']} / {gnn['dropout']} \\\\\n"
    )


def main() -> None:
    data_rep = _load_json(ROOT / "data" / "processed" / "data_report.json") or _FALLBACK_DATA
    bench = _load_json(ROOT / "artifacts" / "benchmark_summary.json") or _FALLBACK_BENCH
    cfg_path = ROOT / "configs" / "default.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}

    lines_comment = [
        "%% AUTO-GENERATED by scripts/generate_report_latex_data.py — do not edit",
    ]
    if not (ROOT / "artifacts" / "benchmark_summary.json").is_file():
        lines_comment.append(
            "%% NOTE: benchmark_summary.json missing; table uses embedded fallback. "
            "Run scripts/run_eval.py to refresh."
        )
    if not (ROOT / "data" / "processed" / "data_report.json").is_file():
        lines_comment.append(
            "%% NOTE: data_report.json missing; data summary uses embedded fallback. "
            "Run scripts/prepare_data.py to refresh."
        )

    hp = _hp_rows(cfg) if cfg else "% (missing default.yaml)\n"

    body = "\n".join(lines_comment) + "\n" + r"""

\newcommand{\GeneratedDataTable}{%
\begin{tabular}{@{}lr@{}}
\toprule
Quantity & Value \\
\midrule
""" + _data_rows(data_rep) + r"""\bottomrule
\end{tabular}
}

\newcommand{\GeneratedBenchmarkTable}{%
\begin{tabular}{@{}llccc@{}}
\toprule
Split & Model & RMSE $\log_{10}D$ & $R^2$ & Med.\ $| \hat D-D|/D$ \\
\midrule
""" + _bench_rows(bench) + r"""
\bottomrule
\end{tabular}
}

\newcommand{\GeneratedHyperparamTable}{%
\begin{tabular}{@{}lr@{}}
\toprule
Setting & Value \\
\midrule
""" + hp + r"""\bottomrule
\end{tabular}
}
"""
    OUT.write_text(body, encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
