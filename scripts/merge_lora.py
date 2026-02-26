#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model and save a standalone model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--adapter-dir", default="training/checkpoints/latin-en-lora/adapter")
    parser.add_argument("--output-dir", default="training/checkpoints/latin-en-merged")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def main() -> None:
    args = parse_args()
    adapter_dir = Path(args.adapter_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter dir not found: {adapter_dir}")

    dtype = resolve_dtype(args.dtype)

    model = AutoPeftModelForCausalLM.from_pretrained(
        str(adapter_dir),
        torch_dtype=dtype,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), use_fast=True)

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    print(f"Merged model saved to: {output_dir}")


if __name__ == "__main__":
    main()
