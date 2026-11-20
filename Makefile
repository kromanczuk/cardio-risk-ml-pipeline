# Shortcuts: make data, train, api, ui, test, deploy.

.PHONY: data train train-all mlflow-ui api test \
        docker-build docker-build-train docker-run docker-run-local docker-train

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

# ---------------------------------------------------------------- API (local)
# API with auto-reload at http://127.0.0.1:8000 (model from MODEL_URI, else GCS latest).
api:
	uvicorn src.api:app --reload

test:
	pytest

# ---------------------------------------------------------------- Docker
API_IMAGE   ?= heart-api
TRAIN_IMAGE ?= heart-train
GCP_PROJECT ?= $(shell awk '/^gcp:/{f=1} f && /project:/{print $$2; exit}' config.yaml)
# Your local gcloud login (ADC), mounted read-only so containers can reach GCS.
# The project must be passed explicitly: containers don't see your gcloud config.
GCLOUD_CREDS = -v $(HOME)/.config/gcloud:/gcloud:ro \
               -e GOOGLE_APPLICATION_CREDENTIALS=/gcloud/application_default_credentials.json \
               -e GOOGLE_CLOUD_PROJECT=$(GCP_PROJECT)

docker-build:
	docker build -t $(API_IMAGE) .

docker-build-train:
	docker build --target train -t $(TRAIN_IMAGE) .

# API at http://127.0.0.1:8080, model from gs://<bucket>/models/latest.
docker-run: docker-build
	docker run --rm -p 8080:8080 $(GCLOUD_CREDS) $(API_IMAGE)

# API at http://127.0.0.1:8080, model from ./artifacts (output of `make train`).
docker-run-local: docker-build
	docker run --rm -p 8080:8080 -v $(PWD)/artifacts:/model:ro -e MODEL_URI=/model $(API_IMAGE)

# Train inside the container. The image provides the environment; the repo is mounted
# at the same path so mlflow.db, mlruns/ and artifacts/ land in your repo as usual.
#   make docker-train                 (model.type from config.yaml)
#   make docker-train MODEL=xgboost
docker-train: docker-build-train
	docker run --rm --user $$(id -u):$$(id -g) -e HOME=/tmp -e USER=$$(whoami) \
		-e GIT_PYTHON_REFRESH=quiet $(GCLOUD_CREDS) \
		-v $(PWD):$(PWD) -w $(PWD) $(TRAIN_IMAGE) $(if $(MODEL),--model-type $(MODEL))
