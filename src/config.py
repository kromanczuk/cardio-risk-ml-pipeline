# Loads config.yaml into a settings object.
"""Shared config loader (used by both the data/ML code and the API).

Usage:
    from src.config import load_config
    cfg = load_config()
    cfg["gcs"]["bucket"]
"""

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Read config.yaml (or CONFIG_PATH) and apply environment overrides."""
    path = path or os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Environment variables win over the file, so secrets/bucket names stay out of git.
    if os.environ.get("GCS_BUCKET"):
        cfg.setdefault("gcs", {})["bucket"] = os.environ["GCS_BUCKET"]
    return cfg


def feature_columns(cfg: dict) -> list[str]:
    """The 13 model input columns, in a fixed order."""
    d = cfg["data"]
    return d["numeric_features"] + d["binary_features"] + d["categorical_features"]
