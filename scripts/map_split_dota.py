#!/usr/bin/env python3
"""Filter DOTA samples by class ids, remap class indices, and build a 70/15/15 split."""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path containing images/ and labels/")
    parser.add_argument("output", type=Path, help="Destination root for filtered dataset")
    parser.add_argument(
        "--class-map",
        type=int,
        nargs="+",
        required=True,
        help="Pairs of original->new class ids, e.g. --class-map 0 0 1 1 9 2",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for shuffling")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Val split ratio (rest test)")
    return parser.parse_args()


def parse_class_map(raw: Sequence[int]) -> Dict[int, int]:
    if len(raw) % 2:
        raise SystemExit("--class-map must be provided as pairs: original new ...")
    mapping: Dict[int, int] = {}
    for i in range(0, len(raw), 2):
        mapping[int(raw[i])] = int(raw[i + 1])
    return mapping


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def parse_label(label_path: Path, mapping: Dict[int, int]) -> list[str]:
    remapped: list[str] = []
    text = label_path.read_text().strip()
    if not text:
        return remapped
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            cls_id = int(float(parts[0]))
        except ValueError:
            continue
        if cls_id not in mapping:
            continue
        new_cls = mapping[cls_id]
        remapped.append(" ".join([str(new_cls)] + parts[1:]))
    return remapped


def prepare_dirs(root: Path, splits: Sequence[str]):
    for split in splits:
        for sub in ("images", "labels"):
            (root / split / sub).mkdir(parents=True, exist_ok=True)


def copy_and_write(img: Path, remapped_lines: list[str], dest_root: Path, split: str):
    dest_img = dest_root / split / "images" / img.name
    dest_label = dest_root / split / "labels" / (img.stem + ".txt")
    dest_img.parent.mkdir(parents=True, exist_ok=True)
    dest_label.parent.mkdir(parents=True, exist_ok=True)
    dest_img.write_bytes(img.read_bytes())
    dest_label.write_text("\n".join(remapped_lines) + ("\n" if remapped_lines else ""))


def ensure_class_coverage(split_samples: list[tuple[int, Path, list[str]]], fallback: list[tuple[int, Path, list[str]]], classes: set[int]):
    present = set(sample[0] for sample in split_samples)
    missing = classes - present
    if not missing:
        return
    for cls in list(missing):
        for idx, candidate in enumerate(fallback):
            if candidate[0] == cls:
                split_samples.append(candidate)
                fallback.pop(idx)
                present.add(cls)
                missing = classes - present
                break
        else:
            raise SystemExit(f"Unable to ensure coverage for class {cls} in split")


def main():
    args = parse_args()
    src_images = args.source / "images"
    src_labels = args.source / "labels"
    if not src_images.is_dir() or not src_labels.is_dir():
        raise SystemExit("Source must contain images/ and labels/ directories")

    mapping = parse_class_map(args.class_map)
    target_classes = set(mapping.values())

    total_labels = 0
    skipped_labels = 0
    samples: list[tuple[int, Path, list[str], Path]] = []
    for label_path in sorted(src_labels.glob("*.txt")):
        total_labels += 1
        remapped = parse_label(label_path, mapping)
        if not remapped:
            skipped_labels += 1
            continue
        img_path = find_image(src_images, label_path.stem)
        if not img_path:
            print(f"[WARN] Missing image for {label_path.name}, skipping")
            continue
        # Use first line's class as primary key for coverage
        first_cls = int(remapped[0].split()[0])
        samples.append((first_cls, img_path, remapped, label_path))

    if not samples:
        raise SystemExit("No samples matched the requested classes.")

    random.seed(args.seed)
    random.shuffle(samples)

    total = len(samples)
    n_train = int(total * args.train_ratio)
    n_val = int(total * args.val_ratio)
    n_test = total - n_train - n_val

    train_samples = samples[:n_train]
    val_samples = samples[n_train : n_train + n_val]
    test_samples = samples[n_train + n_val :]

    ensure_class_coverage(val_samples, train_samples, target_classes)
    ensure_class_coverage(test_samples, train_samples, target_classes)

    prepare_dirs(args.output, ("train", "val", "test"))

    for split_name, split_samples in (("train", train_samples), ("val", val_samples), ("test", test_samples)):
        for primary_cls, img_path, remapped, _label_path in split_samples:
            copy_and_write(img_path, remapped, args.output, split_name)

    print(
        f"Filtered {total} samples -> train {len(train_samples)}, val {len(val_samples)}, test {len(test_samples)} at {args.output}"
    )
    print(
        f"Processed {total_labels} label files, skipped {skipped_labels} that did not contain mapped classes"
    )


if __name__ == "__main__":
    main()
