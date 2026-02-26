#!/usr/bin/env python3
"""Build supervised fine-tuning data from aligned Latin/English text files.

Expected parallel files:
- Generic line-aligned pairs:
  - *_latin.txt
  - *_english.txt
- Or chapter/verse files where lines look like:
  - Chapter 22
  - 1 In principio...

If chapter/verse structure is detected in both files, alignment is done by
(chapter, verse) keys instead of raw line number.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


SYSTEM_PROMPT = (
    "You are a precise translator between Latin and English. "
    "Translate faithfully, preserve meaning, names, numbers, and formatting, "
    "and do not add commentary."
)


VerseKey = Tuple[int, int]


@dataclass
class ParallelPair:
    latin_path: Path
    english_path: Path


@dataclass
class Example:
    direction: str
    source_file: str
    line_number: int
    prompt: str
    response: str

    def to_record(self) -> dict:
        text = (
            f"### System\n{SYSTEM_PROMPT}\n\n"
            f"### Instruction\n{self.prompt}\n\n"
            f"### Response\n{self.response}"
        )
        return {
            "text": text,
            "prompt": self.prompt,
            "response": self.response,
            "direction": self.direction,
            "source_file": self.source_file,
            "line_number": self.line_number,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SFT dataset from parallel Latin/English files")
    parser.add_argument("--data-dir", default="data", help="Directory containing *_latin.txt and *_english.txt")
    parser.add_argument("--latin-file", help="Optional path to a single Latin file")
    parser.add_argument("--english-file", help="Optional path to the matching English file")
    parser.add_argument("--output-dir", default="training/datasets", help="Output directory for JSONL files")
    parser.add_argument("--eval-ratio", type=float, default=0.02, help="Fraction of examples for eval split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--min-chars", type=int, default=3, help="Minimum length for each side after trimming")
    parser.add_argument("--max-examples", type=int, default=0, help="Optional cap on total examples (0 = no cap)")
    parser.add_argument(
        "--bilingual",
        action="store_true",
        help="Include both directions (Latin->English and English->Latin)",
    )
    parser.add_argument(
        "--align-mode",
        choices=["auto", "line", "verse"],
        default="auto",
        help="Alignment mode: auto-detect, force raw line alignment, or force chapter+verse alignment.",
    )
    return parser.parse_args()


def discover_pairs(args: argparse.Namespace) -> List[ParallelPair]:
    if args.latin_file or args.english_file:
        if not args.latin_file or not args.english_file:
            raise ValueError("When using --latin-file/--english-file, provide both.")
        latin = Path(args.latin_file)
        english = Path(args.english_file)
        if not latin.exists() or not english.exists():
            raise FileNotFoundError("Provided latin/english file path does not exist.")
        return [ParallelPair(latin_path=latin, english_path=english)]

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    pairs: List[ParallelPair] = []
    for latin_path in sorted(data_dir.glob("*_latin.txt")):
        english_path = latin_path.with_name(latin_path.name.replace("_latin.txt", "_english.txt"))
        if english_path.exists():
            pairs.append(ParallelPair(latin_path=latin_path, english_path=english_path))

    if not pairs:
        raise FileNotFoundError(
            "No parallel files found. Expected *_latin.txt + *_english.txt in data dir."
        )

    return pairs


def load_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def parse_chapter_verse(lines: List[str]) -> Tuple[Dict[VerseKey, str], List[VerseKey]]:
    """Parse lines like 'Chapter 22' and '1 Verse text' into keyed entries."""

    entries: Dict[VerseKey, str] = {}
    order: List[VerseKey] = []
    current_chapter: Optional[int] = None

    chapter_re = re.compile(r"^Chapter\s+(\d+)\s*$", re.IGNORECASE)
    verse_re = re.compile(r"^(\d+)\s+(.+)$")

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        chapter_match = chapter_re.match(line)
        if chapter_match:
            current_chapter = int(chapter_match.group(1))
            continue

        verse_match = verse_re.match(line)
        if verse_match and current_chapter is not None:
            verse_num = int(verse_match.group(1))
            text = verse_match.group(2).strip()
            if not text:
                continue
            key = (current_chapter, verse_num)
            if key not in entries:
                order.append(key)
            entries[key] = text

    return entries, order


def looks_like_chapter_verse(lines: List[str]) -> bool:
    entries, _ = parse_chapter_verse(lines)
    return len(entries) >= 20


def iter_line_aligned(
    latin_lines: List[str],
    english_lines: List[str],
    min_chars: int,
) -> Iterable[Tuple[int, str, str]]:
    total = min(len(latin_lines), len(english_lines))
    for idx in range(total):
        latin = latin_lines[idx].strip()
        english = english_lines[idx].strip()
        if len(latin) < min_chars or len(english) < min_chars:
            continue
        yield idx + 1, latin, english


def iter_verse_aligned(
    latin_lines: List[str],
    english_lines: List[str],
    min_chars: int,
) -> Iterable[Tuple[int, str, str]]:
    latin_entries, latin_order = parse_chapter_verse(latin_lines)
    english_entries, _ = parse_chapter_verse(english_lines)

    line_number = 0
    for key in latin_order:
        if key not in english_entries:
            continue
        latin = latin_entries[key].strip()
        english = english_entries[key].strip()
        if len(latin) < min_chars or len(english) < min_chars:
            continue
        line_number += 1
        yield line_number, latin, english


def build_examples(
    pair: ParallelPair,
    min_chars: int,
    bilingual: bool,
    align_mode: str,
) -> Iterable[Example]:
    latin_lines = load_lines(pair.latin_path)
    english_lines = load_lines(pair.english_path)

    use_verse_alignment = False
    if align_mode == "verse":
        use_verse_alignment = True
    elif align_mode == "auto":
        use_verse_alignment = looks_like_chapter_verse(latin_lines) and looks_like_chapter_verse(english_lines)

    source_file = f"{pair.latin_path.name}|{pair.english_path.name}"

    if use_verse_alignment:
        aligned_iter = iter_verse_aligned(latin_lines, english_lines, min_chars)
    else:
        aligned_iter = iter_line_aligned(latin_lines, english_lines, min_chars)

    for line_number, latin, english in aligned_iter:
        yield Example(
            direction="lat-en",
            source_file=source_file,
            line_number=line_number,
            prompt=f"Translate this Latin text to English:\n{latin}",
            response=english,
        )

        if bilingual:
            yield Example(
                direction="en-lat",
                source_file=source_file,
                line_number=line_number,
                prompt=f"Translate this English text to Latin:\n{english}",
                response=latin,
            )


def write_jsonl(path: Path, records: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    pairs = discover_pairs(args)

    examples: List[Example] = []
    for pair in pairs:
        examples.extend(
            build_examples(
                pair,
                min_chars=args.min_chars,
                bilingual=args.bilingual,
                align_mode=args.align_mode,
            )
        )

    if not examples:
        raise ValueError("No usable aligned examples were found.")

    random.seed(args.seed)
    random.shuffle(examples)

    if args.max_examples > 0:
        examples = examples[: args.max_examples]

    eval_size = int(len(examples) * args.eval_ratio)
    eval_size = max(1, eval_size) if len(examples) >= 50 else 0

    eval_examples = examples[:eval_size]
    train_examples = examples[eval_size:]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = [ex.to_record() for ex in train_examples]
    eval_records = [ex.to_record() for ex in eval_examples]
    all_records = [ex.to_record() for ex in examples]

    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "eval.jsonl", eval_records)
    write_jsonl(output_dir / "all.jsonl", all_records)

    summary = {
        "num_pairs": len(pairs),
        "num_examples_total": len(all_records),
        "num_examples_train": len(train_records),
        "num_examples_eval": len(eval_records),
        "bilingual": bool(args.bilingual),
        "eval_ratio": args.eval_ratio,
        "align_mode": args.align_mode,
        "files": [
            {
                "latin": str(pair.latin_path),
                "english": str(pair.english_path),
            }
            for pair in pairs
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Dataset build complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
