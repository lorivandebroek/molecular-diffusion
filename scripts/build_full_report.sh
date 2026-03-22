#!/usr/bin/env zsh
# Build report/full_report.pdf (figures + generated tables + LaTeX).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
python scripts/generate_report_latex_data.py
python scripts/build_report_figures.py
TECT="$ROOT/.tools/tectonic/tectonic"
if [[ ! -x "$TECT" ]]; then
  echo "Missing $TECT — install Tectonic or use: cd report && pdflatex full_report.tex" >&2
  exit 1
fi
cd "$ROOT/report"
"$TECT" full_report.tex
echo "Wrote $ROOT/report/full_report.pdf"
