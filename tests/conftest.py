"""Shared fixtures: a tiny local model so API tests run without GCS or MLflow."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.api import app
from src.config import feature_columns, load_config
from src.preprocess import build_preprocessor

DUMMY_VERSION = "test-dummy-v1"

# Valid values for each feature, matching src/schemas.py.
FEATURE_VALUES = {
    "age": range(29, 78),
    "sex": [0, 1],
    "cp": [1, 2, 3, 4],
    "trestbps": range(94, 201),
    "chol": range(126, 565),
    "fbs": [0, 1],
    "restecg": [0, 1, 2],
    "thalach": range(71, 203),
    "exang": [0, 1],
    "oldpeak": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
    "slope": [1, 2, 3],
    "ca": [0, 1, 2, 3],
    "thal": [3, 6, 7],
}


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    """Train a small pipeline on random data and save it like train.py does."""
    cfg = load_config()
    cols = feature_columns(cfg)  # same column order as training
    rng = np.random.default_rng(0)
    X = pd.DataFrame({c: rng.choice(list(FEATURE_VALUES[c]), size=200) for c in cols})
    y = rng.integers(0, 2, size=200)

    model = Pipeline(
        [("preprocess", build_preprocessor(cfg)), ("model", LogisticRegression(max_iter=1000))]
    ).fit(X, y)

    out = tmp_path_factory.mktemp("model")
    joblib.dump(model, out / "model.joblib")
    (out / "metadata.json").write_text(json.dumps({"model_version": DUMMY_VERSION}))
    return out


@pytest.fixture
def client(model_dir, monkeypatch):
    """API client with the dummy model loaded."""
    monkeypatch.setenv("MODEL_URI", str(model_dir))
    with TestClient(app) as c:  # `with` runs the startup (lifespan) code
        yield c


@pytest.fixture
def client_no_model(tmp_path, monkeypatch):
    """API client started with an empty model folder."""
    monkeypatch.setenv("MODEL_URI", str(tmp_path))
    with TestClient(app) as c:
        yield c
