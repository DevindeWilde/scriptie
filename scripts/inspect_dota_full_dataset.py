#!/usr/bin/env python3
"""Summarize a YOLO dataset (counts per split, per class, and basic statistics)."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def parse_label_file(path: Path):
    """Yield (class_id, bbox) tuples from a YOLO label file."""
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        bbox = tuple(float(x) for x in parts[1:5])
        yield cls, bbox


def inspect_split(root: Path, split: str):
    """Collect counts and basic bbox stats for a dataset split."""
    labels_dir = root / split / "labels"
    images_dir = root / split / "images"
    files = sorted(labels_dir.glob("*.txt"))
    class_counts = Counter()
    bbox_stats = Counter()
    for file in files:
        for cls, bbox in parse_label_file(file):
            class_counts[cls] += 1
            bbox_stats["total"] += 1
            bbox_stats["cx"] += bbox[0]
            bbox_stats["cy"] += bbox[1]
            bbox_stats["w"] += bbox[2]
            bbox_stats["h"] += bbox[3]
    num_images = len(list(images_dir.glob("*"))) if images_dir.exists() else 0
    return {
        "files": len(files),
        "images": num_images,
        "class_counts": class_counts,
        "bbox_stats": bbox_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect YOLO dataset splits and class counts.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to dataset root (with train/val/test).")
    parser.add_argument(
        "--class-names",
        type=Path,
        help="Optional path to a text file listing class names (one per line) for readable output.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Splits to inspect.")
    args = parser.parse_args()

    class_names = None
    if args.class_names and args.class_names.exists():
        class_names = [line.strip() for line in args.class_names.read_text().splitlines() if line.strip()]

    for split in args.splits:
        stats = inspect_split(args.dataset_root, split)
        print(f"=== {split.upper()} ===")
        print(f"Images: {stats['images']}, Label files: {stats['files']}")
        total = sum(stats["class_counts"].values())
        print(f"Total objects: {total}")
        for cls_id, count in stats["class_counts"].most_common():
            name = class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)
            print(f"  {name:>20}: {count}")
        print("")


if __name__ == "__main__":
    main()
