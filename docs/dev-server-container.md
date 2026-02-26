# Dev Server Container Runbook

This document explains how to run the translator API on a dev server using Docker.

## What This Runs

- FastAPI app from `api.py`
- HTTP port `8010`
- Translation backend: Ollama endpoint configured by env vars

## Files Used

- `Dockerfile.api`
- `requirements-api.txt`
- `.dockerignore`

## Prerequisites

- Docker installed on the dev server
- Ollama reachable from the container
- Model available in Ollama (for example `mixtral` or your trained model alias)

## 1. Build the API Image

From repo root:

```bash
cd /path/to/latin-rag-translator
docker build -f Dockerfile.api -t latin-rag-translator-api:dev .
```

## 2. Choose Ollama Connectivity

### Option A (Recommended): Ollama on host machine

Use host gateway address from container:

```bash
docker run -d --name latin-rag-api \
  -p 8010:8010 \
  -e OLLAMA_MODEL=latin-en-trained \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  --add-host=host.docker.internal:host-gateway \
  latin-rag-translator-api:dev
```

If you are on Docker Desktop (Mac/Windows), `host.docker.internal` usually works without `--add-host`.

### Option B: Ollama on another server/service

```bash
docker run -d --name latin-rag-api \
  -p 8010:8010 \
  -e OLLAMA_MODEL=latin-en-trained \
  -e OLLAMA_BASE_URL=http://<ollama-host>:11434 \
  latin-rag-translator-api:dev
```

## 3. Verify API is Up

```bash
curl -s http://localhost:8010/health
curl -s http://localhost:8010/model-info
```

Expected response includes:

- `"status": "ok"`
- `"backend": "ollama"`
- `"model": "<your model name>"`

## 4. Smoke Test Translation

```bash
curl -s http://localhost:8010/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"In principio erat Verbum","direction":"lat-en"}'
```

## 5. Logs and Lifecycle

View logs:

```bash
docker logs -f latin-rag-api
```

Restart:

```bash
docker restart latin-rag-api
```

Stop/remove:

```bash
docker rm -f latin-rag-api
```

## 6. API Docs for Integrators

- Swagger UI: `http://localhost:8010/docs`
- ReDoc: `http://localhost:8010/redoc`
- OpenAPI JSON: `http://localhost:8010/openapi.json`
- Handoff guide: `docs/api-handoff.md`

## Environment Variables

- `OLLAMA_MODEL`: Ollama model name (default from app: `mixtral`)
- `OLLAMA_BASE_URL`: Ollama HTTP endpoint (default from app: `http://localhost:11434`)

## Troubleshooting

### `500` on `/translate`

Common cause: API cannot reach Ollama.

Check:

- `OLLAMA_BASE_URL` is reachable from container
- model exists in Ollama (`OLLAMA_MODEL`)

### Container runs, but responses are very slow

- First request may be slower while model warms up
- Larger models need more CPU/RAM/GPU resources

### Port already in use

If `8010` is occupied, map another host port:

```bash
docker run -d --name latin-rag-api \
  -p 8011:8010 \
  -e OLLAMA_MODEL=latin-en-trained \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  --add-host=host.docker.internal:host-gateway \
  latin-rag-translator-api:dev
```

Then call `http://localhost:8011`.
