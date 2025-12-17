#!/usr/bin/env python3
"""Build a tiny exemplar dataset by sampling K images per class."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Set

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create exemplar replay dataset for continual learning.")
    parser.add_argument("--images", type=Path, required=True, help="Path to source images directory.")
    parser.add_argument("--labels", type=Path, required=True, help="Path to source labels directory.")
    parser.add_argument("--dest", type=Path, required=True, help="Destination root (will create images/ and labels/).")
    parser.add_argument("--classes", type=int, nargs="+", required=True, help="Class IDs to keep in memory dataset.")
    parser.add_argument("--per-class", type=int, default=25, help="Images per class to sample.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    return parser.parse_args()


def find_image_for_label(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    images_dir = args.images.resolve()
    labels_dir = args.labels.resolve()
    dest_images = (args.dest / "images").resolve()
    dest_labels = (args.dest / "labels").resolve()
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    class_to_files: Dict[int, List[Path]] = {cls: [] for cls in args.classes}
    label_files = sorted(labels_dir.glob("*.txt"))
    for label_path in label_files:
        with label_path.open("r") as f:
            present = {int(float(line.split()[0])) for line in f if line.strip()}
        for cls in args.classes:
            if cls in present:
                class_to_files[cls].append(label_path)

    selected_files: Set[Path] = set()
    for cls, files in class_to_files.items():
        if not files:
            continue
        sample = files if len(files) <= args.per_class else random.sample(files, args.per_class)
        selected_files.update(sample)

    if not selected_files:
        print("No files selected; check class IDs or source directories.")
        return

    print(f"Selected {len(selected_files)} unique images for exemplar memory.")
    kept_classes = set(args.classes)
    for label_path in selected_files:
        stem = label_path.stem
        image_path = find_image_for_label(images_dir, stem)
        if image_path is None:
            print(f"Skipping {stem}: image file not found.")
            continue
        shutil.copy2(image_path, dest_images / image_path.name)
        with label_path.open("r") as src, (dest_labels / label_path.name).open("w") as dst:
            for line in src:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(float(parts[0]))
                if cls_id in kept_classes:
                    dst.write(line)

    print(f"Memory dataset written to {args.dest}")


if __name__ == "__main__":
    main()
