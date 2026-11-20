"""Tests for /health, /predict, input validation and the no-model fallback."""

import pytest

from src.api import risk_level
from src.schemas import EXAMPLE_PATIENT
from tests.conftest import DUMMY_VERSION


# ---------------------------------------------------------------- with a model
def test_health_reports_loaded_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "model_version": DUMMY_VERSION}


def test_predict_returns_consistent_result(client):
    r = client.post("/predict", json=EXAMPLE_PATIENT)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability"] <= 1
    assert body["prediction"] == int(body["probability"] >= 0.5)
    assert body["risk_level"] == risk_level(body["probability"])
    assert body["model_version"] == DUMMY_VERSION


def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/docs"


# ---------------------------------------------------------------- bad input -> 422
@pytest.mark.parametrize(
    "change",
    [
        {"cp": 9},  # not an allowed code
        {"age": 150},  # out of range
        {"oldpeak": -1},  # out of range
        {"chol": "high"},  # wrong type
        {"chool": 250},  # unknown field (typo)
    ],
)
def test_predict_rejects_invalid_input(client, change):
    r = client.post("/predict", json={**EXAMPLE_PATIENT, **change})
    assert r.status_code == 422


def test_predict_rejects_missing_field(client):
    patient = {k: v for k, v in EXAMPLE_PATIENT.items() if k != "thal"}
    assert client.post("/predict", json=patient).status_code == 422


# ---------------------------------------------------------------- without a model
def test_health_without_model(client_no_model):
    r = client_no_model.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": False, "model_version": None}


def test_predict_without_model_returns_503(client_no_model):
    assert client_no_model.post("/predict", json=EXAMPLE_PATIENT).status_code == 503


# ---------------------------------------------------------------- risk levels
@pytest.mark.parametrize(
    "p, expected",
    [(0.0, "low"), (0.29, "low"), (0.3, "medium"), (0.59, "medium"), (0.6, "high"), (1.0, "high")],
)
def test_risk_level_boundaries(p, expected):
    assert risk_level(p) == expected
