#!/usr/bin/env bash
# Build, push to Artifact Registry, and deploy to Cloud Run.
#
# Usage:
#   deploy/deploy.sh api                          # serves serving.model_version from config.yaml
#   MODEL_VERSION=latest deploy/deploy.sh api     # override the model folder to serve
#
# Prerequisites (one-time): gcloud auth login, the Cloud Run / Artifact Registry APIs enabled,
# the Artifact Registry repo created, and `gcloud auth configure-docker <region>-docker.pkg.dev`.
# See the README "Deploy to Google Cloud" section.
set -euo pipefail
cd "$(dirname "$0")/.."

# Read `key` from the top-level `section` of config.yaml (simple two-level YAML only).
cfg() {
  awk -v sec="$1" -v key="$2" '
    $0 ~ "^" sec ":" { inside = 1; next }
    /^[^ #]/         { inside = 0 }
    inside && $1 == key ":" { print $2; exit }
  ' config.yaml
}

PROJECT="${PROJECT:-$(cfg gcp project)}"
REGION="${REGION:-$(cfg gcp region)}"
REPO="${REPO:-$(cfg gcp artifact_repo)}"
BUCKET="${BUCKET:-$(cfg gcs bucket)}"
MODELS_PREFIX="$(cfg gcs models_prefix)"
REGISTRY="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}"

# Tag images with the git commit, so every deployed image traces back to the code.
TAG="$(git rev-parse --short HEAD)"
if [[ -n "$(git status --porcelain)" ]]; then TAG="${TAG}-dirty"; fi

build_and_push() {  # $1 = image name, remaining args = extra docker build args
  local image="${REGISTRY}/$1:${TAG}"; shift
  # Cloud Run runs linux/amd64; this also works on Apple Silicon (emulated build).
  docker build --platform linux/amd64 -t "$image" "$@" >&2
  docker push "$image" >&2
  echo "$image"
}

deploy_api() {
  local service sa version image
  service="$(cfg serving api_service)"
  sa="$(cfg serving api_service_account)@${PROJECT}.iam.gserviceaccount.com"
  version="${MODEL_VERSION:-$(cfg serving model_version)}"

  echo "==> Building API image (tag ${TAG})"
  image="$(build_and_push heart-api --target api .)"

  echo "==> Deploying ${service} with model gs://${BUCKET}/${MODELS_PREFIX}/${version}"
  gcloud run deploy "$service" \
    --project "$PROJECT" --region "$REGION" \
    --image "$image" \
    --service-account "$sa" \
    --set-env-vars "MODEL_URI=gs://${BUCKET}/${MODELS_PREFIX}/${version}" \
    --allow-unauthenticated \
    --port 8080 --memory 1Gi --cpu 1 --cpu-boost \
    --min-instances 0 --max-instances 2 \
    --quiet

  local url
  url="$(gcloud run services describe "$service" --project "$PROJECT" --region "$REGION" \
    --format 'value(status.url)')"
  echo "==> API live at ${url}  (docs: ${url}/docs)"
}

case "${1:-}" in
  api) deploy_api ;;
  *) echo "Usage: $0 api" >&2; exit 1 ;;
esac
