#!/usr/bin/env bash
# MlangCast — one-shot Cloud Run deploy (Day 4).
#
# Prereqs (once):
#   gcloud auth login && gcloud config set project <YOUR_PROJECT>
#   gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
#                          secretmanager.googleapis.com texttospeech.googleapis.com
#   ./deploy/deploy.sh --create-secrets      # stores GOOGLE_API_KEY + MONGODB_URI in Secret Manager
#
# Then deploy any time with:
#   ./deploy/deploy.sh
#
# Config via env vars (with sensible defaults):
set -euo pipefail

SERVICE="${SERVICE:-mlangcast}"
REGION="${REGION:-us-central1}"
GEMINI_MODEL="${GEMINI_MODEL:-gemini-3-pro}"
DEFAULT_LANGUAGE="${DEFAULT_LANGUAGE:-es-ES}"
REPLAY_SPEED="${REPLAY_SPEED:-30}"

# Optionally create the two secrets from your local environment, then exit.
if [[ "${1:-}" == "--create-secrets" ]]; then
  : "${GOOGLE_API_KEY:?export GOOGLE_API_KEY before --create-secrets}"
  : "${MONGODB_URI:?export MONGODB_URI before --create-secrets}"
  printf '%s' "$GOOGLE_API_KEY" | gcloud secrets create GOOGLE_API_KEY --data-file=- 2>/dev/null \
    || printf '%s' "$GOOGLE_API_KEY" | gcloud secrets versions add GOOGLE_API_KEY --data-file=-
  printf '%s' "$MONGODB_URI" | gcloud secrets create MONGODB_URI --data-file=- 2>/dev/null \
    || printf '%s' "$MONGODB_URI" | gcloud secrets versions add MONGODB_URI --data-file=-
  echo "Secrets stored. Now run ./deploy/deploy.sh"
  exit 0
fi

# Deploy straight from source (Cloud Build uses the Dockerfile in repo root).
gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --set-env-vars "GEMINI_MODEL=${GEMINI_MODEL},DEFAULT_LANGUAGE=${DEFAULT_LANGUAGE},REPLAY_SPEED=${REPLAY_SPEED}" \
  --set-secrets "GOOGLE_API_KEY=GOOGLE_API_KEY:latest,MONGODB_URI=MONGODB_URI:latest"

echo "Deployed. URL above. The UI defaults to the offline 'Fast demo' toggle so it"
echo "works even before you turn on Gemini/TTS."
