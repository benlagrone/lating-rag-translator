# API Guide

## What You Have Now

Your trained artifact is a LoRA adapter under:

- `training/checkpoints/latin-en-lora/adapter/`

The current API runtime (`api.py`) serves translation through Ollama.

## Start the API

```bash
uvicorn api:app --host 0.0.0.0 --port 8010 --reload
```

Interactive docs:

- Swagger UI: `http://localhost:8010/docs`
- ReDoc: `http://localhost:8010/redoc`
- OpenAPI JSON: `http://localhost:8010/openapi.json`

## Endpoints

### `GET /health`

Returns API status and backend model config.

Example response:

```json
{
  "status": "ok",
  "backend": "ollama",
  "model": "mixtral",
  "llm_initialized": false
}
```

### `GET /model-info`

Alias for backend/model visibility.

### `POST /translate`

Translate one text block.

Request:

```json
{
  "text": "In principio erat Verbum",
  "direction": "lat-en"
}
```

Directions:

- `lat-en`: Latin -> English
- `en-lat`: English -> Latin

Example:

```bash
curl -s http://localhost:8010/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"In principio erat Verbum","direction":"lat-en"}'
```

### `POST /translate-file`

Translate a `.txt` file path on the server.

Request:

```json
{
  "path": "data/sample_latin.txt",
  "direction": "lat-en"
}
```

Example:

```bash
curl -s http://localhost:8010/translate-file \
  -H "Content-Type: application/json" \
  -d '{"path":"data/sample_latin.txt","direction":"lat-en"}'
```

## Using Your Trained Model with the API

Right now the API backend is Ollama. To use your trained adapter in this API path:

1. Merge adapter into a standalone HF model:

```bash
python3 scripts/merge_lora.py \
  --adapter-dir training/checkpoints/latin-en-lora/adapter \
  --output-dir training/checkpoints/latin-en-merged
```

2. Convert merged HF model to GGUF and import into Ollama (using your `llama.cpp` tooling).
3. Start API with that Ollama model name:

```bash
export OLLAMA_MODEL=latin-en-trained
uvicorn api:app --host 0.0.0.0 --port 8010 --reload
```

4. Confirm active model:

```bash
curl -s http://localhost:8010/model-info
```

You should see `"model": "latin-en-trained"`.

## Common Errors

- `404` on `/translate-file`: file path does not exist on server.
- `400` on `/translate-file`: input is not a `.txt` file.
- `500`: backend model error (e.g., Ollama not running or model not pulled/created).


## App Integration Handoff

For external app integration patterns and client examples, see:

- `docs/api-handoff.md`


## Dev Server Container

For Dockerized dev server deployment, see:

- `docs/dev-server-container.md`
