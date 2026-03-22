#!/usr/bin/env zsh
# Run FastAPI on :8000 (foreground). In another terminal: cd web && pnpm dev
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
exec uvicorn demo_api.main:app --reload --host 127.0.0.1 --port 8000
