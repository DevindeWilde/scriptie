#!/usr/bin/env python3
"""Count objects per class in YOLO label directories."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count class frequencies in YOLO label .txt files.")
    parser.add_argument(
        "--labels",
        type=Path,
        nargs="+",
        required=True,
        help="One or more label directories (e.g., stage1/test/labels).",
    )
    parser.add_argument(
        "--names",
        type=str,
        default="",
        help="Optional comma-separated class names to print alongside counts.",
    )
    return parser.parse_args()


def count_dir(label_dir: Path) -> Counter:
    counter: Counter = Counter()
    txt_files = sorted(label_dir.glob("*.txt"))
    for path in txt_files:
        with path.open("r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls = int(float(parts[0]))
                counter[cls] += 1
    return counter


def main() -> None:
    args = parse_args()
    name_map: Dict[int, str] = {}
    if args.names:
        for idx, name in enumerate(args.names.split(",")):
            name_map[idx] = name.strip()

    for label_dir in args.labels:
        if not label_dir.exists():
            print(f"{label_dir}: not found, skipping")
            continue
        counts = count_dir(label_dir)
        total = sum(counts.values())
        print(f"=== {label_dir} ===")
        print(f"Total objects: {total}")
        for cls in sorted(counts.keys()):
            label = name_map.get(cls, "")
            print(f"  class {cls:>2} {label:>15}: {counts[cls]}")


if __name__ == "__main__":
    main()
