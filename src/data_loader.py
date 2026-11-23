# Downloads heart_clean.parquet from GCS.
"""Load the cleaned dataset from GCS (path comes from config.yaml).

Usage:
    from src.data_loader import load_data
    df = load_data()                 # downloads gs://<bucket>/data/heart_clean.parquet
    df = load_data(local_path="x.parquet")   # skip GCS, e.g. for tests
"""

import tempfile
from pathlib import Path

import pandas as pd

from src.config import feature_columns, load_config


def download_from_gcs(bucket: str, blob: str, dest: str | Path) -> Path:
    from google.cloud import storage

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    storage.Client().bucket(bucket).blob(blob).download_to_filename(str(dest))
    return dest


def load_data(cfg: dict | None = None, local_path: str | None = None) -> pd.DataFrame:
    """Return the clean DataFrame: 13 feature columns + target."""
    cfg = cfg or load_config()
    if local_path:
        df = pd.read_parquet(local_path)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            path = download_from_gcs(
                cfg["gcs"]["bucket"], cfg["gcs"]["data_path"], Path(tmp) / "heart_clean.parquet"
            )
            df = pd.read_parquet(path)

    expected = feature_columns(cfg) + [cfg["data"]["target"]]
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"Data is missing expected columns: {sorted(missing)}")
    return df[expected]
