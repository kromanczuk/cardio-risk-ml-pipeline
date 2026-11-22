#!/usr/bin/env bash
# Build, push to Artifact Registry, and deploy to Cloud Run.
#
# Usage:
#   deploy/deploy.sh api                          # serves serving.model_version from config.yaml
#   MODEL_VERSION=latest deploy/deploy.sh api     # override the model folder to serve
#   deploy/deploy.sh ui                           # Streamlit UI, pointed at the deployed API
#   deploy/deploy.sh all                          # API, then UI
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

deploy_ui() {
  local service sa api_url image
  service="$(cfg serving ui_service)"
  sa="$(cfg serving ui_service_account)@${PROJECT}.iam.gserviceaccount.com"

  if ! gcloud iam service-accounts describe "$sa" --project "$PROJECT" >/dev/null 2>&1; then
    echo "Service account ${sa} not found. Create it once with:" >&2
    echo "  gcloud iam service-accounts create $(cfg serving ui_service_account) --project ${PROJECT}" >&2
    exit 1
  fi

  # The UI talks to the deployed API; look up its URL unless API_URL is given.
  api_url="${API_URL:-$(gcloud run services describe "$(cfg serving api_service)" \
    --project "$PROJECT" --region "$REGION" --format 'value(status.url)')}"
  if [[ -z "$api_url" ]]; then
    echo "API service not found. Deploy it first: deploy/deploy.sh api" >&2
    exit 1
  fi

  echo "==> Building UI image (tag ${TAG})"
  image="$(build_and_push heart-ui frontend)"

  echo "==> Deploying ${service} -> API ${api_url}"
  # Streamlit keeps a websocket per browser session: --session-affinity keeps a user on
  # the same instance when Cloud Run scales to more than one.
  gcloud run deploy "$service" \
    --project "$PROJECT" --region "$REGION" \
    --image "$image" \
    --service-account "$sa" \
    --set-env-vars "API_URL=${api_url}" \
    --allow-unauthenticated \
    --port 8080 --memory 1Gi --cpu 1 --cpu-boost \
    --min-instances 0 --max-instances 2 --session-affinity \
    --quiet

  local url
  url="$(gcloud run services describe "$service" --project "$PROJECT" --region "$REGION" \
    --format 'value(status.url)')"
  echo "==> UI live at ${url}"
}

case "${1:-}" in
  api) deploy_api ;;
  ui) deploy_ui ;;
  all) deploy_api && deploy_ui ;;
  *) echo "Usage: $0 {api|ui|all}" >&2; exit 1 ;;
esac
