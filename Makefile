# Shortcuts: make data, train, api, ui, test, deploy.

.PHONY: data train train-all mlflow-ui

# Fetch UCI Heart Disease, clean it, upload data/heart_clean.parquet to GCS.
# Needs `gcloud auth application-default login` and GCS_BUCKET (or gcs.bucket in config.yaml).
data:
	python scripts/upload_to_gcs.py

# Train one model, log it to MLflow, upload model.joblib + metadata.json to GCS.
#   make train                       (model.type from config.yaml)
#   make train MODEL=random_forest   (logreg | random_forest | xgboost)
train:
	python -m src.train $(if $(MODEL),--model-type $(MODEL))

# One run per model type, for the MLflow comparison. The last one becomes models/latest/.
train-all:
	python -m src.train --model-type logreg
	python -m src.train --model-type random_forest
	python -m src.train --model-type xgboost

# Open the MLflow dashboard at http://127.0.0.1:5000
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db
