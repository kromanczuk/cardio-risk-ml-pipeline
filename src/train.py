# Splits, trains, evaluates, logs to MLflow, uploads model + metadata to GCS.
"""Train the heart-disease model.

Usage:
    python -m src.train                                  # model type from config.yaml
    python -m src.train --model-type random_forest       # override the model type
    python -m src.train --local-data heart_clean.parquet --no-upload   # offline run

Steps: load data -> stratified train/val/test split -> CV on train -> fit ->
evaluate on val + test -> log to MLflow -> save model.joblib + metadata.json
-> upload to gs://<bucket>/models/<version>/ and gs://<bucket>/models/latest/.

The saved model is the full sklearn Pipeline (preprocessing + classifier), so the
API can call model.predict_proba() on raw 13-column input.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src.config import feature_columns, load_config  # noqa: E402
from src.data_loader import load_data  # noqa: E402
from src.preprocess import build_preprocessor  # noqa: E402

MODEL_TYPES = ("logreg", "random_forest", "xgboost")


# ---------------------------------------------------------------- building blocks
def build_classifier(model_type: str, params: dict, seed: int):
    if model_type == "logreg":
        return LogisticRegression(random_state=seed, **params)
    if model_type == "random_forest":
        return RandomForestClassifier(random_state=seed, **params)
    if model_type == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(random_state=seed, eval_metric="logloss", **params)
    raise ValueError(f"Unknown model type {model_type!r}; choose from {MODEL_TYPES}")


def build_pipeline(cfg: dict, model_type: str) -> Pipeline:
    params = cfg["model"]["params"].get(model_type, {})
    return Pipeline([
        ("preprocess", build_preprocessor(cfg)),
        ("model", build_classifier(model_type, params, cfg["seed"])),
    ])


def split_data(df: pd.DataFrame, cfg: dict):
    """Stratified train/val/test split; sizes are fractions of the full dataset."""
    target, s, seed = cfg["data"]["target"], cfg["split"], cfg["seed"]
    X, y = df[feature_columns(cfg)], df[target]
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=s["test_size"], stratify=y, random_state=seed)
    val_frac = s["val_size"] / (1 - s["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_frac, stratify=y_rest, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y, proba),
        "f1": f1_score(y, pred),
        "recall": recall_score(y, pred),
        "accuracy": accuracy_score(y, pred),
    }


def log_plots(model, X_test, y_test) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["no disease", "disease"], ax=ax, colorbar=False)
    ax.set_title("Confusion matrix (test)")
    mlflow.log_figure(fig, "plots/confusion_matrix.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_title("ROC curve (test)")
    mlflow.log_figure(fig, "plots/roc_curve.png")
    plt.close(fig)


def upload_artifacts(cfg: dict, local_dir: Path, version: str) -> list[str]:
    from google.cloud import storage

    bucket = storage.Client().bucket(cfg["gcs"]["bucket"])
    prefix = cfg["gcs"].get("models_prefix", "models")
    uris = []
    for folder in (version, "latest"):
        for name in ("model.joblib", "metadata.json"):
            blob = f"{prefix}/{folder}/{name}"
            bucket.blob(blob).upload_from_filename(str(local_dir / name))
            uris.append(f"gs://{cfg['gcs']['bucket']}/{blob}")
    return uris


# ---------------------------------------------------------------- main routine
def train(cfg: dict, df: pd.DataFrame, model_type: str | None = None,
          upload: bool = True) -> dict:
    """Run one training job end to end. Returns the metadata dict."""
    model_type = model_type or cfg["model"]["type"]
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type {model_type!r}; choose from {MODEL_TYPES}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, cfg)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=model_type) as run:
        params = cfg["model"]["params"].get(model_type, {})
        mlflow.log_params({
            "model_type": model_type,
            **{f"model__{k}": v for k, v in params.items()},
            "seed": cfg["seed"],
            "test_size": cfg["split"]["test_size"],
            "val_size": cfg["split"]["val_size"],
            "cv_folds": cfg["split"]["cv_folds"],
            "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
        })

        # Cross-validation on the training set only
        cv = StratifiedKFold(n_splits=cfg["split"]["cv_folds"], shuffle=True,
                             random_state=cfg["seed"])
        cv_res = cross_validate(build_pipeline(cfg, model_type), X_train, y_train, cv=cv,
                                scoring=["roc_auc", "f1", "recall", "accuracy"])
        cv_metrics = {f"cv_{m}": float(cv_res[f"test_{m}"].mean())
                      for m in ("roc_auc", "f1", "recall", "accuracy")}

        # Final fit on the training set, then validation + test scores
        model = build_pipeline(cfg, model_type).fit(X_train, y_train)
        val_metrics = evaluate(model, X_val, y_val)
        test_metrics = evaluate(model, X_test, y_test)

        mlflow.log_metrics({
            **cv_metrics,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        })
        log_plots(model, X_test, y_test)
        mlflow.log_artifact(str(Path(__file__).resolve().parent.parent / "config.yaml"))
        # cloudpickle = same trust model as model.joblib (we only load models we trained)
        mlflow.sklearn.log_model(model, name="model", input_example=X_train.head(3),
                                 serialization_format="cloudpickle")

        # model.joblib + metadata.json (format agreed in the team brief)
        now = datetime.now(timezone.utc)
        version = f"{now:%Y%m%d-%H%M%S}-{run.info.run_id[:8]}"
        metadata = {
            "model_version": version,
            "model_type": model_type,
            "trained_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "mlflow_run_id": run.info.run_id,
            "metrics": {k: round(float(v), 4) for k, v in test_metrics.items()},
            "features": feature_columns(cfg),
        }
        out_dir = Path(cfg["output"]["local_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out_dir / "model.joblib")
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        mlflow.log_artifact(str(out_dir / "metadata.json"))

        if upload:
            for uri in upload_artifacts(cfg, out_dir, version):
                print(f"Uploaded {uri}")
        mlflow.set_tag("uploaded_to_gcs", upload)

    print(f"\n{model_type}  (version {version})")
    print(f"  CV   : " + "  ".join(f"{k[3:]}={v:.3f}" for k, v in cv_metrics.items()))
    print(f"  val  : " + "  ".join(f"{k}={v:.3f}" for k, v in val_metrics.items()))
    print(f"  test : " + "  ".join(f"{k}={v:.3f}" for k, v in test_metrics.items()))
    print(f"  saved {out_dir / 'model.joblib'} and {out_dir / 'metadata.json'}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the heart-disease model.")
    parser.add_argument("--model-type", choices=MODEL_TYPES, help="overrides model.type in config")
    parser.add_argument("--local-data", metavar="PATH", help="read this parquet instead of GCS")
    parser.add_argument("--no-upload", action="store_true", help="don't upload to GCS")
    args = parser.parse_args()

    cfg = load_config()
    df = load_data(cfg, local_path=args.local_data)
    train(cfg, df, model_type=args.model_type, upload=not args.no_upload)


if __name__ == "__main__":
    main()
