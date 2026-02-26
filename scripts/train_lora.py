#!/usr/bin/env python3
"""Fine-tune a causal LM for Latin<->English translation using LoRA.

This script expects JSONL built by scripts/build_sft_dataset.py with a `text`
field containing full instruction/response examples.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Latin/English translator")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train-file", default="training/datasets/train.jsonl")
    parser.add_argument("--eval-file", default="training/datasets/eval.jsonl")
    parser.add_argument("--output-dir", default="training/checkpoints/latin-en-lora")

    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1024)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")

    return parser.parse_args()


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def resolve_target_modules(model) -> List[str]:
    candidates = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "query_key_value",
        "dense",
        "fc1",
        "fc2",
        "wq",
        "wk",
        "wv",
        "wo",
    }

    found = set()
    for name, _module in model.named_modules():
        suffix = name.split(".")[-1]
        if suffix in candidates:
            found.add(suffix)

    if not found:
        raise ValueError(
            "Could not auto-detect LoRA target modules. "
            "Set a model with standard attention/MLP layer names."
        )

    return sorted(found)


def load_base_model(base_model: str, use_4bit: bool, dtype: torch.dtype):
    model_kwargs = {}
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"
        return AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    # Prefer modern key (`dtype`) and fall back for older transformers.
    model_kwargs["dtype"] = dtype
    try:
        return AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    except TypeError:
        model_kwargs.pop("dtype", None)
        model_kwargs["torch_dtype"] = dtype
        return AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)


def make_training_args(
    args: argparse.Namespace,
    output_dir: Path,
    has_eval: bool,
    use_bf16: bool,
    use_fp16: bool,
    use_4bit: bool,
) -> TrainingArguments:
    kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        bf16=use_bf16,
        fp16=use_fp16,
        lr_scheduler_type="cosine",
        report_to="none",
        seed=args.seed,
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
    )

    params = inspect.signature(TrainingArguments.__init__).parameters
    eval_value = "steps" if has_eval else "no"
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = eval_value
    else:
        kwargs["evaluation_strategy"] = eval_value

    return TrainingArguments(**kwargs)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = detect_device()
    dtype = choose_dtype(device)
    use_4bit = bool(args.load_in_4bit and device == "cuda")

    if args.load_in_4bit and not use_4bit:
        print("--load-in-4bit requested, but CUDA is unavailable; continuing without 4-bit quantization.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_base_model(args.base_model, use_4bit=use_4bit, dtype=dtype)

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules = resolve_target_modules(model)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.config.use_cache = False

    model.print_trainable_parameters()

    train_path = Path(args.train_file)
    eval_path = Path(args.eval_file)

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")

    data_files = {"train": str(train_path)}
    if eval_path.exists() and eval_path.stat().st_size > 0:
        data_files["eval"] = str(eval_path)

    dataset = load_dataset("json", data_files=data_files)
    has_eval = "eval" in dataset

    def tokenize_batch(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_len,
            padding=False,
        )

    remove_columns = dataset["train"].column_names
    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=remove_columns,
        desc="Tokenizing dataset",
    )

    use_bf16 = device == "cuda" and dtype == torch.bfloat16
    use_fp16 = device == "cuda" and dtype == torch.float16

    training_args = make_training_args(
        args=args,
        output_dir=output_dir,
        has_eval=has_eval,
        use_bf16=use_bf16,
        use_fp16=use_fp16,
        use_4bit=use_4bit,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("eval") if has_eval else None,
        data_collator=collator,
    )

    trainer.train()

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    metadata = {
        "base_model": args.base_model,
        "device": device,
        "dtype": str(dtype),
        "used_4bit": use_4bit,
        "target_modules": target_modules,
        "train_file": str(train_path),
        "eval_file": str(eval_path) if eval_path.exists() else None,
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("Training complete.")
    print(f"Adapter saved to: {adapter_dir}")


if __name__ == "__main__":
    main()
