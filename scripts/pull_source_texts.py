#!/usr/bin/env python3
"""Download a full Latin/English Bible parallel corpus into local text files.

Default source:
  jam963/parallel-catholic-bible-versions (Hugging Face)

Outputs (line-aligned):
  data/vulgate_latin.txt
  data/vulgate_english.txt
  data/vulgate_refs.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull parallel Latin/English source texts into local data/*.txt files."
    )
    parser.add_argument(
        "--dataset",
        default="jam963/parallel-catholic-bible-versions",
        help="Hugging Face dataset name",
    )
    parser.add_argument("--split", default="train", help="Dataset split to load")
    parser.add_argument(
        "--latin-column",
        default="vulgate",
        help="Column to use as Latin source text",
    )
    parser.add_argument(
        "--english-column",
        default="drc",
        help="Column to use as English target text (e.g. drc or cpdv)",
    )
    parser.add_argument(
        "--testament",
        choices=["all", "ot", "nt"],
        default="all",
        help="Optional testament filter",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory to write output files",
    )
    parser.add_argument(
        "--output-prefix",
        default="vulgate",
        help="Output prefix, producing <prefix>_latin.txt and <prefix>_english.txt",
    )
    parser.add_argument(
        "--max-verses",
        type=int,
        default=0,
        help="Optional cap for debugging (0 = all rows)",
    )
    return parser.parse_args()


def _int_or(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _keep_row(row: Dict[str, Any], testament_filter: str) -> bool:
    if testament_filter == "all":
        return True
    value = str(row.get("testament", "")).strip().lower()
    return value == testament_filter


def iter_parallel_rows(
    rows: Iterable[Dict[str, Any]],
    latin_column: str,
    english_column: str,
    testament_filter: str,
) -> Iterable[Tuple[str, str, str]]:
    for row in rows:
        if not _keep_row(row, testament_filter):
            continue

        latin = _clean_text(row.get(latin_column))
        english = _clean_text(row.get(english_column))
        if not latin or not english:
            continue

        book = _clean_text(row.get("book"))
        chapter = _int_or(row.get("chapter"), 0)
        verse = _int_or(row.get("verse"), 0)
        ref = f"{book} {chapter}:{verse}".strip()
        yield latin, english, ref


def main() -> None:
    args = parse_args()

    dataset = load_dataset(args.dataset, split=args.split)
    rows: List[Dict[str, Any]] = [dict(record) for record in dataset]
    rows.sort(
        key=lambda row: (
            _int_or(row.get("book_number"), 10**9),
            _int_or(row.get("chapter"), 10**9),
            _int_or(row.get("verse"), 10**9),
        )
    )

    selected = list(
        iter_parallel_rows(
            rows=rows,
            latin_column=args.latin_column,
            english_column=args.english_column,
            testament_filter=args.testament,
        )
    )
    if args.max_verses > 0:
        selected = selected[: args.max_verses]

    if not selected:
        raise ValueError("No usable verse pairs were produced from the selected dataset/columns.")

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    latin_path = data_dir / f"{args.output_prefix}_latin.txt"
    english_path = data_dir / f"{args.output_prefix}_english.txt"
    refs_path = data_dir / f"{args.output_prefix}_refs.txt"
    summary_path = data_dir / f"{args.output_prefix}_summary.json"

    latin_lines = [latin for latin, _, _ in selected]
    english_lines = [english for _, english, _ in selected]
    ref_lines = [ref for _, _, ref in selected]

    latin_path.write_text("\n".join(latin_lines) + "\n", encoding="utf-8")
    english_path.write_text("\n".join(english_lines) + "\n", encoding="utf-8")
    refs_path.write_text("\n".join(ref_lines) + "\n", encoding="utf-8")

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "latin_column": args.latin_column,
        "english_column": args.english_column,
        "testament": args.testament,
        "rows_written": len(selected),
        "outputs": {
            "latin": str(latin_path),
            "english": str(english_path),
            "refs": str(refs_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {len(selected)} aligned verse pairs")
    print(f"Latin:   {latin_path}")
    print(f"English: {english_path}")
    print(f"Refs:    {refs_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
