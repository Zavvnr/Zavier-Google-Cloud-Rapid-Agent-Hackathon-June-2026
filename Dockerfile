# MlangCast — container image for Google Cloud Run (Day 4 deploy).
#
# Quick deploy (build happens in the cloud, no local Docker needed):
#   gcloud run deploy mlangcast --source . --region us-central1 \
#       --allow-unauthenticated \
#       --set-env-vars GEMINI_MODEL=gemini-3-pro,DEFAULT_LANGUAGE=es-ES \
#       --set-secrets GOOGLE_API_KEY=GOOGLE_API_KEY:latest,MONGODB_URI=MONGODB_URI:latest
#
# See deploy/cloudrun.md for the full walkthrough (secrets, etc.).

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so this layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app (see .dockerignore: .env, .venv, caches, tests are excluded;
# data/cache IS included so the deployed demo can use the real cached match).
COPY . .

# Cloud Run injects $PORT (defaults to 8080). The gthread worker streams
# Server-Sent Events; --timeout 0 stops gunicorn cutting off a long commentary
# stream mid-match.
ENV PORT=8080
EXPOSE 8080
CMD exec gunicorn -b :$PORT -k gthread --workers 1 --threads 8 --timeout 0 web.app:app
