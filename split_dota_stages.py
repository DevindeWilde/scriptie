#!/usr/bin/env python
"""Split YOLO-format DOTA labels into stage-specific folders with filtered classes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Set

STAGE_CLASSES: Dict[str, Set[int]] = {
    "stage1": {10, 9, 11, 12, 16, 17},  # small-vehicle, large-vehicle, helicopter, roundabout, airport, helipad
    "stage2": {3, 4, 5, 6, 13, 14},  # baseball, tennis, basketball, ground-track, soccer, swimming
    "stage3": {0, 1, 2, 7, 8, 15},  # plane, ship, storage-tank, harbor, bridge, container-crane
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stage-specific label folders for DOTA YOLO data.")
    parser.add_argument("--labels-root", type=Path, required=True, help="Path to YOLO labels root (contains train/val/test).")
    parser.add_argument("--splits", type=str, nargs="+", default=("train", "val", "test"), help="Dataset splits to process.")
    parser.add_argument("--suffix", type=str, default="", help="Optional suffix appended to stage folder names.")
    return parser.parse_args()


def filter_labels(src_file: Path, dst_file: Path, allowed: Set[int]) -> None:
    lines_out: List[str] = []
    if src_file.exists():
        with src_file.open("r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(float(parts[0]))
                if cls_id in allowed:
                    lines_out.append(line.strip())
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    with dst_file.open("w") as out:
        out.write("\n".join(lines_out))


def main():
    args = parse_args()
    for split in args.splits:
        split_dir = args.labels_root / split
        if not split_dir.exists():
            print(f"Skipping missing split directory: {split_dir}")
            continue
        label_files = sorted(split_dir.glob("*.txt"))
        print(f"[{split}] processing {len(label_files)} label files")
        for label_path in label_files:
            for stage, allowed in STAGE_CLASSES.items():
                stage_name = f"{stage}{args.suffix}".strip()
                dst_dir = split_dir.parent / f"{split}_{stage_name}"
                dst_file = dst_dir / label_path.name
                filter_labels(label_path, dst_file, allowed)
    print("Stage split completed.")


if __name__ == "__main__":
    main()
