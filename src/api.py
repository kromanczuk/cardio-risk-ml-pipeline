"""FastAPI app serving heart-disease risk predictions.

Run locally:
    uvicorn src.api:app --reload                      # model from gs://<bucket>/models/latest
    MODEL_URI=artifacts uvicorn src.api:app --reload  # model from a local folder

The model (model.joblib + metadata.json) is loaded once at startup from MODEL_URI.
If it can't be loaded, the API still starts: /health reports model_loaded=false
and /predict returns 503 until a model is available and the API is restarted.
"""

import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse

from src.config import load_config
from src.schemas import HealthResponse, PatientFeatures, PredictionResponse

logger = logging.getLogger("uvicorn.error")

ARTIFACT_FILES = ("model.joblib", "metadata.json")
THRESHOLD = 0.5  # probability >= THRESHOLD -> prediction 1
LOW_RISK_BELOW = 0.3
HIGH_RISK_FROM = 0.6


# ---------------------------------------------------------------- model loading
def default_model_uri() -> str:
    """gs://<bucket>/<models_prefix>/latest, built from config.yaml."""
    gcs = load_config()["gcs"]
    return f"gs://{gcs['bucket']}/{gcs.get('models_prefix', 'models')}/latest"


def fetch_artifacts(uri: str, download_dir: Path) -> Path:
    """Return a local folder holding model.joblib + metadata.json.

    `uri` is either gs://bucket/path/to/folder (downloaded into download_dir)
    or a local folder path (used as is).
    """
    if not uri.startswith("gs://"):
        return Path(uri)

    from google.cloud import storage

    bucket_name, _, prefix = uri.removeprefix("gs://").partition("/")
    bucket = storage.Client().bucket(bucket_name)
    for name in ARTIFACT_FILES:
        blob = f"{prefix.strip('/')}/{name}" if prefix.strip("/") else name
        bucket.blob(blob).download_to_filename(str(download_dir / name))
    return download_dir


def load_artifacts(uri: str) -> tuple[object, dict]:
    """Load (model, metadata) from a gs:// or local folder."""
    with tempfile.TemporaryDirectory() as tmp:
        folder = fetch_artifacts(uri, Path(tmp))
        model = joblib.load(folder / "model.joblib")
        metadata = json.loads((folder / "metadata.json").read_text())
    return model, metadata


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once at startup: load the model, or start without one."""
    app.state.model, app.state.metadata = None, {}
    uri = os.environ.get("MODEL_URI")
    try:
        uri = uri or default_model_uri()
        app.state.model, app.state.metadata = load_artifacts(uri)
        logger.info("Loaded model %s from %s", app.state.metadata.get("model_version"), uri)
    except Exception:
        # Deliberately broad: a missing/broken model must not stop the API from starting.
        logger.exception("No model loaded from %s; /predict will return 503", uri)
    yield


app = FastAPI(
    title="Heart Disease Risk API",
    description="Predicts the probability of heart disease from 13 clinical features "
    "(UCI Heart Disease, Cleveland). Educational project, not medical advice.",
    version="1.0.0",
    lifespan=lifespan,
)


def risk_level(probability: float) -> str:
    if probability < LOW_RISK_BELOW:
        return "low"
    if probability < HIGH_RISK_FROM:
        return "medium"
    return "high"


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=request.app.state.model is not None,
        model_version=request.app.state.metadata.get("model_version"),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={503: {"description": "No model is loaded"}},
)
def predict(features: PatientFeatures, request: Request) -> PredictionResponse:
    model = request.app.state.model
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="No model is loaded. Train and upload one, then restart the API.",
        )

    X = pd.DataFrame([features.model_dump()])
    # sklearn requires the same column order as in training.
    if hasattr(model, "feature_names_in_"):
        X = X[list(model.feature_names_in_)]

    probability = float(model.predict_proba(X)[0, 1])
    return PredictionResponse(
        probability=probability,
        prediction=int(probability >= THRESHOLD),
        risk_level=risk_level(probability),
        model_version=request.app.state.metadata.get("model_version", "unknown"),
    )
