#!/usr/bin/env python3
"""Utilities to verify QUEST label definitions against training CSV headers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

from src.inference_qwen3_ollama import QUEST_LABELS

BASE_COLUMNS = [
    "qa_id",
    "question_title",
    "question_body",
    "question_user_name",
    "question_user_page",
    "answer",
    "answer_user_name",
    "answer_user_page",
    "url",
    "category",
    "host",
]


def extract_label_columns(csv_path: Path) -> List[str]:
    """Return all label columns found in the given CSV (after metadata columns)."""
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

    return [column for column in header if column not in BASE_COLUMNS]


def main(csv_path: str = "data/train.csv") -> None:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find CSV at {path}")

    csv_labels = extract_label_columns(path)

    print(f"Labels discovered in {path} ({len(csv_labels)} columns):")
    for column in csv_labels:
        print(f"  - {column}")

    print("\nLabels defined in inference_qwen3.QUEST_LABELS:")
    for column in QUEST_LABELS:
        print(f"  - {column}")

    missing_from_csv = [label for label in QUEST_LABELS if label not in csv_labels]
    missing_from_code = [label for label in csv_labels if label not in QUEST_LABELS]

    if not missing_from_csv and not missing_from_code:
        print("\n✅ Label sets match exactly.")
    else:
        print("\n⚠️ Label mismatch detected:")
        if missing_from_csv:
            print("  • Present in code but missing from CSV header:")
            for label in missing_from_csv:
                print(f"    - {label}")
        if missing_from_code:
            print("  • Present in CSV header but missing from code:")
            for label in missing_from_code:
                print(f"    - {label}")


if __name__ == "__main__":
    main()
