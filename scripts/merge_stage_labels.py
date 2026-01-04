#!/usr/bin/env python3
"""Merge YOLO label files from multiple stages so each file contains all class entries."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge label files from several stage folders.")
    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Destination labels directory (e.g., stage2/test/labels) where merged files will be written.",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        nargs="+",
        required=True,
        help="Source label directories ordered from oldest stage to newest.",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="Remove duplicate label lines while preserving order.",
    )
    return parser.parse_args()


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def merge_files(dest: Path, sources: list[Path], dedup: bool) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    all_files = set()
    for src in sources:
        if not src.exists():
            raise FileNotFoundError(f"Source directory not found: {src}")
        all_files.update(p.name for p in src.glob("*.txt"))

    for name in sorted(all_files):
        merged: list[str] = []
        seen = set()
        for src in sources:
            lines = read_lines(src / name)
            for line in lines:
                if dedup:
                    if line in seen:
                        continue
                    seen.add(line)
                merged.append(line)
        out_path = dest / name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            f.write("\n".join(merged) + ("\n" if merged else ""))
        print(f"Merged {name}: {len(merged)} labels")


def main() -> None:
    args = parse_args()
    merge_files(args.dest, args.sources, args.dedup)


if __name__ == "__main__":
    main()
