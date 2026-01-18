#!/usr/bin/env python3
"""Filter DOTA samples by class ids and build a 70/15/15 split."""
from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


@dataclass
class Sample:
    img: Path
    label: Path
    classes: set[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to source split (expects images/ and labels/)")
    parser.add_argument("output", type=Path, help="Destination root for filtered dataset")
    parser.add_argument("--classes", type=int, nargs="+", required=True, help="Class ids to keep")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for shuffling")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Val split ratio")
    return parser.parse_args()


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def parse_label_classes(label_path: Path, keep: set[int]) -> set[int]:
    classes: set[int] = set()
    text = label_path.read_text().strip()
    if not text:
        return classes
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            cls_id = int(float(parts[0]))
        except ValueError:
            continue
        if cls_id in keep:
            classes.add(cls_id)
    return classes


def prepare_dirs(root: Path, splits: Sequence[str]):
    for split in splits:
        for sub in ("images", "labels"):
            (root / split / sub).mkdir(parents=True, exist_ok=True)


def copy_sample(sample: Sample, dest_root: Path, split: str):
    shutil.copy2(sample.img, dest_root / split / "images" / sample.img.name)
    shutil.copy2(sample.label, dest_root / split / "labels" / sample.label.name)


def ensure_class_coverage(split_samples: list[Sample], fallback: list[Sample], classes: set[int]):
    """Ensure every class appears at least once in split_samples by borrowing from fallback."""
    present = set().union(*(s.classes for s in split_samples))
    missing = classes - present
    if not missing:
        return
    for cls in list(missing):
        for idx, candidate in enumerate(fallback):
            if cls in candidate.classes:
                split_samples.append(candidate)
                fallback.pop(idx)
                present.update(candidate.classes)
                missing = classes - present
                break
        else:
            raise SystemExit(f"Unable to ensure coverage for class {cls} in validation/test split")


def main():
    args = parse_args()
    src_images = args.source / "images"
    src_labels = args.source / "labels"
    if not src_images.is_dir() or not src_labels.is_dir():
        raise SystemExit("Source must contain images/ and labels/ directories")

    keep = set(args.classes)
    samples: list[Sample] = []
    for label_path in sorted(src_labels.glob("*.txt")):
        classes = parse_label_classes(label_path, keep)
        if not classes:
            continue
        img_path = find_image(src_images, label_path.stem)
        if not img_path:
            print(f"[WARN] Missing image for {label_path.name}, skipping")
            continue
        samples.append(Sample(img=img_path, label=label_path, classes=classes))

    if not samples:
        raise SystemExit("No samples matched the requested classes.")

    random.seed(args.seed)
    random.shuffle(samples)

    prepare_dirs(args.output, ("train", "val", "test"))
    n = len(samples)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    n_test = n - n_train - n_val

    train_samples = samples[:n_train]
    val_samples = samples[n_train : n_train + n_val]
    test_samples = samples[n_train + n_val :]

    ensure_class_coverage(val_samples, train_samples, keep)
    ensure_class_coverage(test_samples, train_samples, keep)

    splits = (("train", train_samples), ("val", val_samples), ("test", test_samples))
    for split_name, split_samples in splits:
        for sample in split_samples:
            copy_sample(sample, args.output, split_name)

    print(
        f"Filtered {n} samples -> train {len(train_samples)}, val {len(val_samples)}, test {len(test_samples)} at {args.output}"
    )


if __name__ == "__main__":
    main()
