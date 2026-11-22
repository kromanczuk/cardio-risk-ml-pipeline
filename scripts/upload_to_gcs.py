# One-time: fetch UCI id 45, clean, binarize target, upload parquet to GCS.
"""Fetch UCI Heart Disease (Cleveland), clean it, and upload it to GCS as parquet.

Usage:
    python scripts/upload_to_gcs.py                         # fetch, clean, upload
    python scripts/upload_to_gcs.py --local out.parquet     # fetch + clean only, no upload
    python scripts/upload_to_gcs.py --raw-file processed.cleveland.data
                                       # read a downloaded UCI file instead of fetching

Requires `gcloud auth application-default login` for the upload step.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import feature_columns, load_config


def fetch_raw(uci_id: int) -> pd.DataFrame:
    """Download the dataset from the UCI repository: 13 features + `num` (0-4)."""
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=uci_id)
    return pd.concat([ds.data.features, ds.data.targets], axis=1)


def read_raw_file(path: str) -> pd.DataFrame:
    """Read UCI's `processed.cleveland.data` (no header, '?' for missing values)."""
    uci_order = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "num",
    ]
    return pd.read_csv(path, header=None, names=uci_order, na_values="?")


def clean(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply the team contract: 13 features with original UCI encodings + binary `target`."""
    features = feature_columns(cfg)
    target = cfg["data"]["target"]

    missing = set(features + ["num"]) - set(raw.columns)
    if missing:
        raise ValueError(f"Raw data is missing columns: {sorted(missing)}")

    df = raw[features + ["num"]].copy()
    df = df.replace("?", pd.NA)  # older copies of the file use '?'
    df[features] = df[features].apply(pd.to_numeric)
    df = df.dropna().reset_index(drop=True)  # ~6 rows with missing ca/thal

    df[target] = (df["num"] > 0).astype(int)  # 0 = no disease, 1-4 = disease
    df = df.drop(columns=["num"])

    # Whole-number columns back to int (NaNs forced them to float).
    int_cols = [c for c in features if c != "oldpeak"]
    df[int_cols] = df[int_cols].astype(int)
    return df


def upload(local_path: str, bucket: str, blob: str) -> None:
    from google.cloud import storage

    storage.Client().bucket(bucket).blob(blob).upload_from_filename(local_path)
    print(f"Uploaded -> gs://{bucket}/{blob}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", metavar="PATH", help="save parquet here and skip the upload")
    parser.add_argument(
        "--raw-file",
        metavar="PATH",
        help="use a downloaded processed.cleveland.data instead of fetching from UCI",
    )
    args = parser.parse_args()

    cfg = load_config()
    raw = read_raw_file(args.raw_file) if args.raw_file else fetch_raw(cfg["data"]["uci_id"])
    df = clean(raw, cfg)

    counts = df[cfg["data"]["target"]].value_counts().sort_index().to_dict()
    print(f"Raw rows: {len(raw)}  ->  clean rows: {len(df)}")
    print(f"Target counts (0 = no disease, 1 = disease): {counts}")

    if args.local:
        df.to_parquet(args.local, index=False)
        print(f"Saved {args.local} (no upload)")
        return

    bucket = cfg["gcs"]["bucket"]
    if not bucket or bucket.startswith("REPLACE"):
        sys.exit("Set gcs.bucket in config.yaml or the GCS_BUCKET env var first.")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "heart_clean.parquet"
        df.to_parquet(path, index=False)
        upload(str(path), bucket, cfg["gcs"]["data_path"])


if __name__ == "__main__":
    main()
