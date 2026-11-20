# Tests for the preprocessing pipeline.
"""Tests for src/preprocess.py. They use a small hand-made table: no GCS, no W&B."""
import numpy as np
import pandas as pd
import pytest

from src.config import feature_columns, load_config
from src.preprocess import build_preprocessor

CFG = load_config()

# Every category value from the UCI contract, so the encoder sees all of them.
CATEGORY_VALUES = {"cp": [1, 2, 3, 4], "restecg": [0, 1, 2], "slope": [1, 2, 3], "thal": [3, 6, 7]}
N_ONE_HOT = sum(len(v) for v in CATEGORY_VALUES.values())  # 13


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame({
        "age": rng.integers(29, 78, n),
        "trestbps": rng.integers(94, 200, n),
        "chol": rng.integers(126, 564, n),
        "thalach": rng.integers(71, 202, n),
        "oldpeak": rng.uniform(0, 6.2, n).round(1),
        "ca": rng.integers(0, 4, n),
        "sex": rng.integers(0, 2, n),
        "fbs": rng.integers(0, 2, n),
        "exang": rng.integers(0, 2, n),
    })
    for col, values in CATEGORY_VALUES.items():
        df[col] = np.resize(values, n)  # cycle through every value
    return df


def test_config_lists_the_13_contract_features():
    expected = {"age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalach", "exang", "oldpeak", "slope", "ca", "thal"}
    cols = feature_columns(CFG)
    assert set(cols) == expected
    assert len(cols) == len(set(cols)) == 13


def test_output_shape(sample_df):
    out = build_preprocessor(CFG).fit_transform(sample_df)
    n_numeric = len(CFG["data"]["numeric_features"])
    n_binary = len(CFG["data"]["binary_features"])
    assert out.shape == (len(sample_df), n_numeric + N_ONE_HOT + n_binary)  # 6 + 13 + 3 = 22
    assert not np.isnan(out).any()


def test_numeric_columns_are_scaled(sample_df):
    out = build_preprocessor(CFG).fit_transform(sample_df)
    numeric = out[:, : len(CFG["data"]["numeric_features"])]
    np.testing.assert_allclose(numeric.mean(axis=0), 0, atol=1e-8)
    np.testing.assert_allclose(numeric.std(axis=0), 1, atol=1e-8)


def test_binary_columns_pass_through_unchanged(sample_df):
    out = build_preprocessor(CFG).fit_transform(sample_df)
    binary = out[:, -len(CFG["data"]["binary_features"]):]
    np.testing.assert_array_equal(binary, sample_df[CFG["data"]["binary_features"]].to_numpy())


def test_unseen_category_does_not_crash(sample_df):
    pre = build_preprocessor(CFG).fit(sample_df)
    row = sample_df.iloc[[0]].copy()
    row["cp"] = 99  # not a valid chest-pain code
    out = pre.transform(row)
    assert out.shape[0] == 1 and not np.isnan(out).any()


def test_missing_column_raises(sample_df):
    pre = build_preprocessor(CFG).fit(sample_df)
    with pytest.raises(ValueError):
        pre.transform(sample_df.drop(columns=["chol"]))
