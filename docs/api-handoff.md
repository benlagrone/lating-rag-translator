# API Handoff Guide for Other Apps

This guide is for teams integrating external apps/services with this translator API.

## Base URL

Use an environment variable in each client app:

- `TRANSLATOR_API_URL` (example: `https://translator.yourdomain.com`)

Local default:

- `http://localhost:8010`

## API Surface

- `GET /health`: service health and active backend/model
- `GET /model-info`: alias for backend/model visibility
- `POST /translate`: translate one text payload
- `POST /translate-file`: translate a server-side `.txt` file path

For cross-app integrations, prefer `POST /translate`.

`POST /translate-file` is only appropriate when the API server can directly read the file path you pass.

## Request/Response Contract

### `POST /translate`

Request JSON:

```json
{
  "text": "In principio erat Verbum",
  "direction": "lat-en"
}
```

- `text`: required string
- `direction`: required enum
- `lat-en`: Latin -> English
- `en-lat`: English -> Latin

Success response (`200`):

```json
{
  "translation": "In the beginning was the Word",
  "direction": "lat-en"
}
```

### `POST /translate-file`

Request JSON:

```json
{
  "path": "data/sample_latin.txt",
  "direction": "lat-en"
}
```

Success response (`200`) uses same shape as `/translate`.

## Errors

- `400`: invalid request, or non-`.txt` path for `/translate-file`
- `404`: file not found for `/translate-file`
- `500`: backend/model runtime error

Error body (FastAPI default):

```json
{
  "detail": "error message"
}
```

## Integration Pattern (Recommended)

Use this pattern in consumer apps:

1. Build text chunks in your app (paragraph/sentence units).
2. Call `POST /translate` for each chunk.
3. Reassemble translated chunks in original order.
4. Retry only on retryable failures (`5xx`, timeouts).

## Timeouts and Retries

Suggested client defaults:

- request timeout: `120s`
- max retries: `2`
- retry backoff: `0.5s`, `1.5s`
- retry only on: network errors, `502`, `503`, `504`

Avoid retrying `400` and `404`.

## JavaScript Example (Node/Browser)

```ts
const BASE_URL = process.env.TRANSLATOR_API_URL ?? "http://localhost:8010";

type Direction = "lat-en" | "en-lat";

type TranslateResponse = {
  translation: string;
  direction: Direction;
};

export async function translateText(text: string, direction: Direction = "lat-en"): Promise<TranslateResponse> {
  const res = await fetch(`${BASE_URL}/translate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, direction }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(`Translator API ${res.status}: ${err.detail ?? "Unknown error"}`);
  }

  return (await res.json()) as TranslateResponse;
}
```

## Python Example

```python
import os
import requests

BASE_URL = os.getenv("TRANSLATOR_API_URL", "http://localhost:8010")


def translate_text(text: str, direction: str = "lat-en") -> dict:
    resp = requests.post(
        f"{BASE_URL}/translate",
        json={"text": text, "direction": direction},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()
```

## Quick Validation for Integrators

```bash
curl -s "$TRANSLATOR_API_URL/health"
curl -s "$TRANSLATOR_API_URL/model-info"
curl -s "$TRANSLATOR_API_URL/translate" \
  -H "Content-Type: application/json" \
  -d '{"text":"In principio erat Verbum","direction":"lat-en"}'
```

## OpenAPI for Client Generation

- `GET /openapi.json`

You can generate strongly typed clients from this spec in other apps.

## Current Limitations to Communicate in Handoff

- No built-in auth/rate limiting in API yet.
- `/translate-file` depends on server-local filesystem visibility.
- Response latency depends on Ollama model size and hardware.

## Deployment Handoff Checklist

- API URL and environment variable documented in consumer app
- Health check wired (`/health`)
- Request timeout + retry policy set
- Direction mapping tested (`lat-en`, `en-lat`)
- Error handling for `400/404/500` implemented
- Observability added (request id, latency, failure count)
