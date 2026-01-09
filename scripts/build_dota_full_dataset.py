#!/usr/bin/env python3
"""
Build a unified DOTA YOLO dataset by merging per-stage label files.

The existing project stores the same images in multiple stage folders (stage1/2/3),
with each stage containing only the labels for the classes introduced in that stage.
This script collects all label files for each split (train/val/test), concatenates
their annotations per image, and writes a consolidated label directory under a new
output dataset root. Optionally, it can also copy or symlink the image folders from
one of the stage roots so the resulting dataset is immediately usable.

Usage:
    python scripts/build_dota_full_dataset.py \
        --stage-roots datasets/dota-yolo-stage1 \
                      datasets/dota-yolo-stage2 \
                      datasets/dota-yolo-stage3 \
        --output-root datasets/dota-yolo-full
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List


def merge_labels(stage_roots: List[Path], split: str, out_root: Path) -> int:
    """Merge label files for a given split and return number of files written."""
    combined = {}
    for stage in stage_roots:
        label_dir = stage / split / "labels"
        if not label_dir.exists():
            continue
        for file in sorted(label_dir.glob("*.txt")):
            if not file.is_file():
                continue
            lines = {line.strip() for line in file.read_text().splitlines() if line.strip()}
            if not lines:
                continue
            combined.setdefault(file.name, set()).update(lines)

    out_dir = out_root / split / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, lines in combined.items():
        out_path = out_dir / name
        out_path.write_text("\n".join(sorted(lines)) + "\n")
    return len(combined)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge DOTA stage datasets into a unified YOLO dataset.")
    parser.add_argument(
        "--stage-roots",
        nargs="+",
        required=True,
        help="List of stage root directories (each containing train/val/test subfolders).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Destination root directory for the merged dataset.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to process (default: train val test).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    stage_roots = [Path(p).resolve() for p in args.stage_roots]
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    total_labels = 0
    for split in args.splits:
        written = merge_labels(stage_roots, split, output_root)
        total_labels += written
        print(f"[INFO] Split '{split}': wrote {written} label files.")

    print(f"[DONE] Merged dataset written to {output_root} with {total_labels} label files total.")


if __name__ == "__main__":
    main()
