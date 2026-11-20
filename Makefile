# Shortcuts: make data, train, api, ui, test, deploy.

.PHONY: data

# Fetch UCI Heart Disease, clean it, upload data/heart_clean.parquet to GCS.
# Needs `gcloud auth application-default login` and GCS_BUCKET (or gcs.bucket in config.yaml).
data:
	python scripts/upload_to_gcs.py
