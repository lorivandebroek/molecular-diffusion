"""FastAPI app: predict log10(D) and D from SMILES + temperature; RDKit SVG depict."""

from __future__ import annotations

import math
import os
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from diffusion_mol.predict import load_predictor
from diffusion_mol.scaling import inv_log10_d

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = ROOT / "artifacts" / "models" / "gnn_scaffold.pt"
MAX_SWEEP_STEPS = 100


def _checkpoint_path() -> Path:
    raw = os.environ.get("DIFFUSION_MODEL_PATH", str(DEFAULT_CKPT))
    return Path(raw).expanduser().resolve()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ckpt = _checkpoint_path()
    if not ckpt.is_file():
        raise RuntimeError(
            f"Checkpoint not found: {ckpt}. Train the model or set DIFFUSION_MODEL_PATH to a .pt file."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.state.predictor = load_predictor(ckpt, device=device)
    app.state.checkpoint_path = str(ckpt)
    yield


app = FastAPI(title="diffusion-mol demo", lifespan=lifespan)

_origins = os.environ.get("DEMO_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    smiles: str = Field(..., min_length=1)
    temperature_k: float = Field(..., gt=0, lt=5000)


class PredictResponse(BaseModel):
    log10_D: float
    D: float
    extrapolated: bool
    message: str | None = None


class SweepRequest(BaseModel):
    smiles: str = Field(..., min_length=1)
    t_min_k: float = Field(..., gt=0, lt=5000)
    t_max_k: float = Field(..., gt=0, lt=5000)
    steps: int = Field(50, ge=2, le=MAX_SWEEP_STEPS)


class SweepPoint(BaseModel):
    T: float
    log10_D: float
    D: float


class SweepResponse(BaseModel):
    points: list[SweepPoint]


class DepictRequest(BaseModel):
    smiles: str = Field(..., min_length=1)
    width: int = Field(220, ge=80, le=600)
    height: int = Field(180, ge=80, le=600)


class DepictResponse(BaseModel):
    svg: str


def _json_float(x: float) -> float | None:
    return None if (isinstance(x, float) and math.isnan(x)) else x


@app.get("/api/health")
def health():
    pred = app.state.predictor
    return {
        "ok": True,
        "model_type": pred.model_type,
        "checkpoint": app.state.checkpoint_path,
        "temperature_min_k": _json_float(pred.t_min),
        "temperature_max_k": _json_float(pred.t_max),
    }


def _predict_with_warning_capture(pred, smiles: str, temperature_k: float) -> tuple[float, bool, str | None]:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always", UserWarning)
        logd = pred.predict_logd(smiles.strip(), float(temperature_k))
    extrapolated = any(w.category is UserWarning for w in rec)
    msg = str(rec[0].message) if rec else None
    return logd, extrapolated, msg


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    pred = app.state.predictor
    try:
        logd, extrapolated, msg = _predict_with_warning_capture(
            pred, req.smiles, req.temperature_k
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}") from e
    d = float(inv_log10_d(logd))
    return PredictResponse(
        log10_D=logd,
        D=d,
        extrapolated=extrapolated,
        message=msg,
    )


@app.post("/api/sweep", response_model=SweepResponse)
def sweep(req: SweepRequest):
    if req.t_max_k < req.t_min_k:
        raise HTTPException(status_code=400, detail="t_max_k must be >= t_min_k")
    pred = app.state.predictor
    temps = np.linspace(req.t_min_k, req.t_max_k, int(req.steps))
    points: list[SweepPoint] = []
    for t in temps:
        try:
            logd, _, _ = _predict_with_warning_capture(pred, req.smiles, float(t))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"At T={t}: {e}") from e
        points.append(
            SweepPoint(T=float(t), log10_D=logd, D=float(inv_log10_d(logd)))
        )
    return SweepResponse(points=points)


@app.post("/api/depict", response_model=DepictResponse)
def depict(req: DepictRequest):
    from rdkit import Chem
    from rdkit.Chem import Draw

    mol = Chem.MolFromSmiles(req.smiles.strip())
    if mol is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES")
    svg = Draw.MolToSVG(mol, width=req.width, height=req.height)
    return DepictResponse(svg=svg)
