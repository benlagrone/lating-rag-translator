# Real Model Training (LoRA)

This project includes a full fine-tuning path for Latin <-> English translation.

## 1. Pull Real Parallel Corpus (Vulgate + English)

Use the built-in source puller:

```bash
python3 scripts/pull_source_texts.py --english-column drc --output-prefix vulgate
```

Default source dataset:

- `jam963/parallel-catholic-bible-versions` (Hugging Face)

Generated files:

- `data/vulgate_latin.txt`
- `data/vulgate_english.txt`
- `data/vulgate_refs.txt`
- `data/vulgate_summary.json`

Optional filters/examples:

```bash
# Use CPDV instead of Douay-Rheims
python3 scripts/pull_source_texts.py --english-column cpdv --output-prefix vulgate_cpdv

# Only New Testament
python3 scripts/pull_source_texts.py --english-column drc --testament nt --output-prefix vulgate_nt
```

## 2. Install Training Dependencies

```bash
python3 -m pip install -r requirements-training.txt
```

## 3. Build SFT Dataset

```bash
python3 scripts/build_sft_dataset.py \
  --latin-file data/vulgate_latin.txt \
  --english-file data/vulgate_english.txt \
  --bilingual \
  --output-dir training/datasets/vulgate
```

Outputs:

- `training/datasets/vulgate/train.jsonl`
- `training/datasets/vulgate/eval.jsonl`
- `training/datasets/vulgate/all.jsonl`
- `training/datasets/vulgate/summary.json`

## 4. Train LoRA Adapter

Compatible/default (CPU/MPS/CUDA):

```bash
python3 scripts/train_lora.py \
  --train-file training/datasets/vulgate/train.jsonl \
  --eval-file training/datasets/vulgate/eval.jsonl \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --gradient-checkpointing
```

CUDA QLoRA (faster/cheaper VRAM):

```bash
python3 scripts/train_lora.py \
  --train-file training/datasets/vulgate/train.jsonl \
  --eval-file training/datasets/vulgate/eval.jsonl \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --load-in-4bit \
  --gradient-checkpointing
```

Outputs:

- `training/checkpoints/latin-en-lora/adapter/`
- `training/checkpoints/latin-en-lora/training_metadata.json`

## 5. Merge Adapter (Optional)

```bash
python3 scripts/merge_lora.py \
  --adapter-dir training/checkpoints/latin-en-lora/adapter \
  --output-dir training/checkpoints/latin-en-merged
```

## 6. Serving Note

Current `main.py` and `api.py` use Ollama (`Config.LLM_MODEL`).

The LoRA output is a Hugging Face adapter/merged model. To use it in Ollama, you need an extra conversion path (typically GGUF + Ollama model import) or add a Hugging Face inference backend to this app.
