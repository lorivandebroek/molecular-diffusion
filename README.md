# Diffusion coefficient prediction (water)

Predict **aqueous diffusion coefficient** \(D\) from **SMILES** and **temperature (K)** using the dataset in `jp5c01881_si_002.xlsx` (Table S2). See [approach.md](approach.md) for the scientific workflow.

## Setup (venv)

```bash
cd molecular-diffusion   # repository root
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
# Optional: local web demo API
pip install -e ".[demo]"
```

**PyTorch Geometric:** if `pip install -e .` fails on `torch-geometric`, install PyTorch first from [pytorch.org](https://pytorch.org), then follow [PyG installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for your platform (CPU/CUDA). On macOS CPU, often:

```bash
pip install torch torch-geometric
```

**RDKit:** included via PyPI (`rdkit`). If wheels are missing for your platform, use **conda-forge**: `conda install -c conda-forge rdkit`.

## Data preparation

Place `jp5c01881_si_002.xlsx` in the project root (or pass `--excel`).

```bash
python scripts/prepare_data.py
```

Writes `data/processed/cleaned.csv` and `data/processed/data_report.json`.

## Training

```bash
python scripts/run_train.py --split random --model mlp
python scripts/run_train.py --split scaffold --model mlp
python scripts/run_train.py --split random --model gnn
python scripts/run_train.py --split scaffold --model gnn
```

Checkpoints: `artifacts/models/{mlp,gnn}_{random,scaffold}.pt`  
Metrics: `artifacts/metrics_{model}_{split}.json`

Hyperparameters: [configs/default.yaml](configs/default.yaml).

## Evaluation and plots

After training (checkpoints optional for NN rows; baselines always run):

```bash
python scripts/run_eval.py --splits random scaffold
```

Writes `artifacts/benchmark_summary.json` and figures under `artifacts/figures/{random,scaffold}/`.

## Tests

```bash
pytest
```

## Inference (Python API)

```python
from diffusion_mol.predict import load_predictor

p = load_predictor("artifacts/models/mlp_random.pt")
D = p.predict_D("CC", 298.15)
```

Training temperature range is stored in the checkpoint config for extrapolation warnings.

## Web demo (Vite + FastAPI)

Requires a trained checkpoint (default: `artifacts/models/gnn_scaffold.pt`). Override with `DIFFUSION_MODEL_PATH`.

**Terminal 1 — API** (from repository root, venv activated):

```bash
./scripts/run_demo.sh
# or: uvicorn demo_api.main:app --reload --port 8000
```

**Terminal 2 — UI**:

```bash
cd web && pnpm install && pnpm dev
```

Open [http://localhost:5173](http://localhost:5173). The dev server proxies `/api` to `http://127.0.0.1:8000`.

Optional: `DEMO_CORS_ORIGINS` (comma-separated) if you serve the UI from another origin.

## Interpretation

- Metrics are reported for **log10(D)** (primary) and linear **D** after back-transform.
- **Random** vs **scaffold** splits measure different generalization axes; see the research note in your project plan.
- Model is **water-only**; **T** outside the training range is extrapolation.
