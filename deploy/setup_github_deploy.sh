#!/usr/bin/env bash
# One-time setup: let GitHub Actions deploy to Cloud Run without any stored keys
# (Workload Identity Federation). Safe to re-run: existing resources are kept.
#
# Creates:
#   - service account heart-mlops-deployer: can deploy Cloud Run, push images, and run
#     services as the runtime account. It has NO access to the GCS bucket.
#   - workload identity pool "github" + OIDC provider "github-actions" that only accepts
#     tokens from this repository's main branch.
#
# Usage:  bash deploy/setup_github_deploy.sh            (needs Owner on the project)
set -euo pipefail
cd "$(dirname "$0")/.."

cfg() {
  awk -v sec="$1" -v key="$2" '
    $0 ~ "^" sec ":" { inside = 1; next }
    /^[^ #]/         { inside = 0 }
    inside && $1 == key ":" { print $2; exit }
  ' config.yaml
}

PROJECT="$(cfg gcp project)"
REGION="$(cfg gcp region)"
REPO_AR="$(cfg gcp artifact_repo)"
RUNNER_SA="$(cfg serving service_account)@${PROJECT}.iam.gserviceaccount.com"
DEPLOYER="heart-mlops-deployer"
DEPLOYER_SA="${DEPLOYER}@${PROJECT}.iam.gserviceaccount.com"
POOL="github"
PROVIDER="github-actions"
# owner/name of the GitHub repo, from the git remote (works for SSH and HTTPS URLs).
GITHUB_REPO="$(git remote get-url origin | sed -E 's#(git@github.com:|https://github.com/)##; s#\.git$##')"
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format 'value(projectNumber)')"

echo "Project ${PROJECT} (${PROJECT_NUMBER}), GitHub repo ${GITHUB_REPO}"

echo "==> Enabling APIs"
gcloud services enable iam.googleapis.com iamcredentials.googleapis.com sts.googleapis.com \
  --project "$PROJECT"

echo "==> Deployer service account"
gcloud iam service-accounts describe "$DEPLOYER_SA" --project "$PROJECT" >/dev/null 2>&1 ||
  gcloud iam service-accounts create "$DEPLOYER" --project "$PROJECT" \
    --display-name "GitHub Actions deployer"

# Deploy and manage Cloud Run services (incl. making them public).
gcloud projects add-iam-policy-binding "$PROJECT" --condition=None --quiet \
  --member "serviceAccount:${DEPLOYER_SA}" --role roles/run.admin >/dev/null
# Push images to our Artifact Registry repo only.
gcloud artifacts repositories add-iam-policy-binding "$REPO_AR" --location "$REGION" \
  --project "$PROJECT" --member "serviceAccount:${DEPLOYER_SA}" \
  --role roles/artifactregistry.writer >/dev/null
# Deploy services that run as the runtime account (and only that account).
gcloud iam service-accounts add-iam-policy-binding "$RUNNER_SA" --project "$PROJECT" \
  --member "serviceAccount:${DEPLOYER_SA}" --role roles/iam.serviceAccountUser >/dev/null

echo "==> Workload identity pool + GitHub provider"
gcloud iam workload-identity-pools describe "$POOL" --location global \
  --project "$PROJECT" >/dev/null 2>&1 ||
  gcloud iam workload-identity-pools create "$POOL" --location global --project "$PROJECT" \
    --display-name "GitHub Actions"

gcloud iam workload-identity-pools providers describe "$PROVIDER" --location global \
  --workload-identity-pool "$POOL" --project "$PROJECT" >/dev/null 2>&1 ||
  gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" --location global \
    --workload-identity-pool "$POOL" --project "$PROJECT" \
    --display-name "GitHub Actions OIDC" \
    --issuer-uri "https://token.actions.githubusercontent.com" \
    --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
    --attribute-condition "assertion.repository == '${GITHUB_REPO}' && assertion.ref == 'refs/heads/main'"

# Workflows from this repo (main branch, enforced above) may act as the deployer.
gcloud iam service-accounts add-iam-policy-binding "$DEPLOYER_SA" --project "$PROJECT" \
  --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/attribute.repository/${GITHUB_REPO}" \
  >/dev/null

cat <<EOF

Done. The workflow (.github/workflows/ci.yml) expects these values:
  GCP_WIF_PROVIDER = projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/providers/${PROVIDER}
  GCP_DEPLOYER_SA  = ${DEPLOYER_SA}
EOF
