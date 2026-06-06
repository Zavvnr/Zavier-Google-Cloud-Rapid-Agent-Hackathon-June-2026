# Deploying MlangCast to Google Cloud Run (Day 4)

The app is a single Flask service (`web.app:app`) containerized by the repo-root
`Dockerfile`. Secrets are injected as env vars at runtime — **nothing secret is
baked into the image** (`.env` is excluded by `.dockerignore` and `.gcloudignore`).

## 1. One-time setup

```bash
gcloud auth login
gcloud config set project <YOUR_PROJECT_ID>

# Enable the APIs the app + build need:
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com \
  texttospeech.googleapis.com
```

## 2. Store secrets in Secret Manager

The app reads `GOOGLE_API_KEY` (Gemini + Cloud TTS) and `MONGODB_URI` (Atlas).
Put them in Secret Manager instead of env files:

```bash
export GOOGLE_API_KEY=...           # your key
export MONGODB_URI='mongodb+srv://...'
./deploy/deploy.sh --create-secrets
```

(Behind the scenes that runs `gcloud secrets create … --data-file=-`.)

Grant the Cloud Run runtime service account access if prompted:

```bash
PROJECT_NUMBER=$(gcloud projects describe "$(gcloud config get-value project)" --format='value(projectNumber)')
for S in GOOGLE_API_KEY MONGODB_URI; do
  gcloud secrets add-iam-policy-binding "$S" \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
done
```

## 3. Deploy

```bash
./deploy/deploy.sh
```

or directly:

```bash
gcloud run deploy mlangcast --source . --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_MODEL=gemini-3-pro,DEFAULT_LANGUAGE=es-ES,REPLAY_SPEED=30 \
  --set-secrets GOOGLE_API_KEY=GOOGLE_API_KEY:latest,MONGODB_URI=MONGODB_URI:latest
```

`gcloud` prints the public HTTPS URL — that's your live demo link.

## 4. Verify

- Open the URL → the UI loads with **Fast demo (offline)** ticked, so it streams
  the bundled sample even before Gemini/TTS are on. Untick it for real Gemini
  commentary; tick **Speak audio** for Google TTS.
- `data/cache/` is uploaded (see `.gcloudignore`), so the cached WC2022 final
  appears in the match picker too.

## Notes / tuning

- **Request timeout:** commentary streams for the length of the match ÷ speed.
  Keep `REPLAY_SPEED` high enough that a stream finishes within Cloud Run's
  request timeout (default 300s; raise with `--timeout` up to 3600s if needed).
- **Concurrency:** SSE holds a worker thread per active viewer. The container
  runs `gunicorn -k gthread --threads 8`; bump threads / instances for more
  simultaneous viewers.
- **Cost control:** add `--min-instances 0` (scale to zero) and
  `--max-instances 2` for a hackathon demo.
