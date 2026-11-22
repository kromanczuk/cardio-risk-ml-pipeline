"""Tests for src/train.py: tiny fake data, MLflow in a temp folder, no GCS upload."""

import copy
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import feature_columns, load_config
from src.train import MODEL_TYPES, train

REQUIRED_METADATA = {
    "model_version",
    "model_type",
    "trained_at",
    "mlflow_run_id",
    "metrics",
    "features",
}
REQUIRED_METRICS = {"roc_auc", "f1", "recall", "accuracy"}


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(load_config())
    c["mlflow"]["tracking_uri"] = f"sqlite:///{tmp_path / 'mlflow.db'}"
    c["output"]["local_dir"] = str(tmp_path / "artifacts")
    c["split"]["cv_folds"] = 3
    c["model"]["params"]["random_forest"]["n_estimators"] = 20  # keep tests fast
    c["model"]["params"]["xgboost"]["n_estimators"] = 20
    return c


@pytest.fixture
def fake_df():
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame(
        {
            "age": rng.integers(29, 78, n),
            "trestbps": rng.integers(94, 200, n),
            "chol": rng.integers(126, 564, n),
            "thalach": rng.integers(71, 202, n),
            "oldpeak": rng.uniform(0, 6.2, n).round(1),
            "ca": rng.integers(0, 4, n),
            "sex": rng.integers(0, 2, n),
            "fbs": rng.integers(0, 2, n),
            "exang": rng.integers(0, 2, n),
            "cp": np.resize([1, 2, 3, 4], n),
            "restecg": np.resize([0, 1, 2], n),
            "slope": np.resize([1, 2, 3], n),
            "thal": np.resize([3, 6, 7], n),
        }
    )
    df["target"] = ((df["exang"] == 1) | (df["oldpeak"] > 3)).astype(int)
    return df


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_train_writes_contract_artifacts(cfg, fake_df, tmp_path, model_type):
    meta = train(cfg, fake_df, model_type=model_type, upload=False)

    # metadata.json has every field from the team brief
    saved = json.loads((tmp_path / "artifacts" / "metadata.json").read_text())
    assert saved == meta
    assert REQUIRED_METADATA <= meta.keys()
    assert REQUIRED_METRICS <= meta["metrics"].keys()
    assert meta["model_type"] == model_type
    assert meta["features"] == feature_columns(cfg)

    # model.joblib is the full pipeline: the API's exact call works on raw columns
    model = joblib.load(tmp_path / "artifacts" / "model.joblib")
    one_row = fake_df[feature_columns(cfg)].iloc[[0]]
    proba = model.predict_proba(one_row)[:, 1]
    assert proba.shape == (1,) and 0.0 <= proba[0] <= 1.0


def test_unknown_model_type_raises(cfg, fake_df):
    with pytest.raises(ValueError):
        train(cfg, fake_df, model_type="svm", upload=False)
