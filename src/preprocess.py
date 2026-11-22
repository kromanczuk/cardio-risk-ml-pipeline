# Builds the ColumnTransformer (scaling + one-hot encoding).
"""Preprocessing for the 13 UCI features.

The ColumnTransformer returned here becomes the first step of the saved model
Pipeline (see src/train.py), so the API can pass raw feature values straight in.

    numeric      -> StandardScaler
    categorical  -> OneHotEncoder (unseen values are ignored, not an error)
    binary       -> passed through unchanged (already 0/1)
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import load_config


def build_preprocessor(cfg: dict | None = None) -> ColumnTransformer:
    """Return an unfitted ColumnTransformer built from the feature lists in config.yaml."""
    cfg = cfg or load_config()
    d = cfg["data"]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), d["numeric_features"]),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                d["categorical_features"],
            ),
            ("bin", "passthrough", d["binary_features"]),
        ],
        remainder="drop",  # ignore any extra columns the caller sends
    )
