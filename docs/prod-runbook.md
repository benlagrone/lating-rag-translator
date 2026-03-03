# Production Publish Runbook

This runbook covers releasing and deploying the translator API to production using Docker on a Linux server.

## Scope

- App: FastAPI service (`api.py`)
- Container: `Dockerfile.api`
- Port: container `8010`
- Inference backend: Ollama (`OLLAMA_MODEL`, `OLLAMA_BASE_URL`)

## Release Inputs

Before deploying, confirm:

- target Git commit SHA
- target Docker image tag
- target Ollama model name in production
- target Ollama base URL reachable from API container

## 1. Pre-Deploy Checklist

Run from your local repo:

```bash
cd /path/to/lating-rag-translator

# Optional but recommended checks
python3 -m py_compile api.py main.py
```

Validate runtime files exist:

- `Dockerfile.api`
- `requirements-api.txt`
- `deploy/prod.env.example`

## 2. Build and Push Release Image

Set your registry and tag format (example uses commit SHA):

```bash
cd /path/to/lating-rag-translator

GIT_SHA=$(git rev-parse --short HEAD)
IMAGE=ghcr.io/benlagrone/lating-rag-translator-api:${GIT_SHA}

docker build -f Dockerfile.api -t ${IMAGE} .
docker push ${IMAGE}

echo "Published ${IMAGE}"
```

Record the exact `${IMAGE}` value in your release notes.

## 3. Prepare Production Server

On production host:

```bash
mkdir -p /opt/lating-rag-translator
cd /opt/lating-rag-translator
```

Create env file once:

```bash
cat > prod.env <<'ENV'
OLLAMA_MODEL=latin-en-trained
OLLAMA_BASE_URL=http://host.docker.internal:11434
APP_ENV=production
APP_NAME=latin-rag-translator-api
ENV
```

Adjust `OLLAMA_BASE_URL` to your real production Ollama endpoint.

## 4. Deploy Container

Use the published image from step 2.

```bash
IMAGE=ghcr.io/benlagrone/lating-rag-translator-api:<release-tag>

# Stop and remove existing container if present
docker rm -f latin-rag-api 2>/dev/null || true

# Pull target image
docker pull ${IMAGE}

# Run new container
docker run -d --name latin-rag-api \
  --restart unless-stopped \
  -p 8010:8010 \
  --env-file /opt/lating-rag-translator/prod.env \
  --add-host=host.docker.internal:host-gateway \
  ${IMAGE}
```

## 5. Post-Deploy Verification

Run smoke checks from the server:

```bash
curl -s http://localhost:8010/health
curl -s http://localhost:8010/model-info
curl -s http://localhost:8010/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"In principio erat Verbum","direction":"lat-en"}'
```

Expected:

- `/health` returns `status=ok`
- `/model-info` returns expected model name
- `/translate` returns non-empty `translation`

## 6. Rollback Procedure

If deploy is unhealthy, roll back immediately to previous known-good image:

```bash
# Example previous image tag
PREV_IMAGE=ghcr.io/benlagrone/lating-rag-translator-api:<previous-tag>

docker rm -f latin-rag-api 2>/dev/null || true
docker pull ${PREV_IMAGE}

docker run -d --name latin-rag-api \
  --restart unless-stopped \
  -p 8010:8010 \
  --env-file /opt/lating-rag-translator/prod.env \
  --add-host=host.docker.internal:host-gateway \
  ${PREV_IMAGE}

curl -s http://localhost:8010/health
```

## 7. Operational Commands

```bash
# logs
docker logs --tail=200 -f latin-rag-api

# container status
docker ps --filter name=latin-rag-api

# restart
docker restart latin-rag-api
```

## 8. Production Traffic Front Door (Recommended)

Put a reverse proxy/load balancer in front of `:8010`:

- TLS termination
- request size/time limits
- access logs
- optional auth/rate limiting

## 9. Change Record Template

For each prod publish, record:

- date/time (UTC)
- operator
- git SHA
- image tag
- env changes
- verification results
- rollback needed (yes/no)

## 10. Known Failure Modes

- `500` from API: Ollama unreachable or model missing
- startup crash: dependency drift in image build
- empty response/slow response: model cold start or resource pressure

## Related Docs

- `docs/dev-server-container.md`
- `docs/api.md`
- `docs/api-handoff.md`
