# Latin RAG Translator with Ollama + Mixtral

This project provides two translation workflows:

1. Inference with Ollama (`main.py` and `api.py`)
2. Real LoRA fine-tuning workflow (`scripts/` + `training/`)

## Features

- Latin -> English and English -> Latin direction support
- File and phrase translation via CLI
- HTTP API via FastAPI (`/translate`, `/translate-file`, `/health`)
- Optional RAG context from local corpus
- Real model fine-tuning pipeline (LoRA/QLoRA)

## Inference Requirements

```bash
python3 -m pip install langchain chromadb sentence-transformers pypdf fastapi uvicorn
```

Install and run Ollama model:

```bash
curl https://ollama.com/install.sh | sh
ollama run mixtral
```

## CLI Usage

Default (Latin -> English):

```bash
python3 main.py --file De_Genesi_ad_Litteram.txt
```

English -> Latin phrase:

```bash
python3 main.py --direction en-lat --phrase "In the beginning was the Word"
```

Use retrieval context (requires usable corpus in `data/`):

```bash
python3 main.py --use-rag --file De_Genesi_ad_Litteram.txt --direction lat-en
```

## HTTP API

Start server:

```bash
uvicorn api:app --reload
```

Interactive API docs:

- Swagger UI: `http://localhost:8010/docs`
- ReDoc: `http://localhost:8010/redoc`
- OpenAPI JSON: `http://localhost:8010/openapi.json`

Detailed API guide:

- `docs/api.md`
- `docs/api-handoff.md`
- `docs/dev-server-container.md`
- `docs/prod-runbook.md`

Phrase translate:

```bash
curl -X POST http://localhost:8010/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"In principio erat Verbum","direction":"lat-en"}'
```

File translate:

```bash
curl -X POST http://localhost:8010/translate-file \
  -H "Content-Type: application/json" \
  -d '{"path":"De_Genesi_ad_Litteram.txt","direction":"lat-en"}'
```

## Use Trained Model In API

Set the Ollama model name at runtime (no code edit needed):

```bash
export OLLAMA_MODEL=latin-en-trained
uvicorn api:app --reload
```

Then verify:

```bash
curl -s http://localhost:8010/model-info
```

## Real Model Training (LoRA)

Training quickstart with full Vulgate + Douay-Rheims parallel verses:

```bash
python3 -m pip install -r requirements-training.txt
python3 scripts/pull_source_texts.py --english-column drc --output-prefix vulgate
python3 scripts/build_sft_dataset.py \
  --latin-file data/vulgate_latin.txt \
  --english-file data/vulgate_english.txt \
  --bilingual \
  --output-dir training/datasets/vulgate
python3 scripts/train_lora.py \
  --train-file training/datasets/vulgate/train.jsonl \
  --eval-file training/datasets/vulgate/eval.jsonl \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --gradient-checkpointing
```

Detailed instructions:

- `training/README.md`

## Corpus Format

For parallel corpus ingestion, add line-aligned file pairs:

- `*_latin.txt`
- `*_english.txt`

You can generate a full pair automatically:

```bash
python3 scripts/pull_source_texts.py --english-column drc --output-prefix vulgate
```

This writes:

- `data/vulgate_latin.txt`
- `data/vulgate_english.txt`
- `data/vulgate_refs.txt`

## License

MIT
