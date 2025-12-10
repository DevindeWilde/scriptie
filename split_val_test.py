#!/usr/bin/env python
"""Split the DOTA validation set into separate val/test folders (80/20)."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split validation split into val/test folders with 80/20 ratio.")
    parser.add_argument("--root", type=Path, required=True, help="Dataset root containing images/ and labels/ directories.")
    parser.add_argument("--split", type=str, default="val", help="Name of the source split to divide (default: val).")
    parser.add_argument("--ratio", type=float, default=0.2, help="Fraction of files to move into the new test split.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffling.")
    parser.add_argument("--image-subdir", type=str, default="images", help="Top-level image folder (default: images).")
    parser.add_argument("--label-subdir", type=str, default="labels", help="Top-level label folder (default: labels).")
    parser.add_argument("--ext", type=str, nargs="+", default=[".jpg", ".png", ".jpeg", ".tif"], help="Allowed image extensions.")
    return parser.parse_args()


def resolve_path(root: Path, subdir: str, split: str) -> Path:
    candidates = [root / subdir / split, root / split / subdir]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {subdir}/{split} under {root}")


def main():
    args = parse_args()
    random.seed(args.seed)

    val_img_dir = resolve_path(args.root, args.image_subdir, args.split)
    val_lbl_dir = resolve_path(args.root, args.label_subdir, args.split)

    test_img_dir = val_img_dir.parent / "test"
    test_lbl_dir = val_lbl_dir.parent / "test"
    test_img_dir.mkdir(parents=True, exist_ok=True)
    test_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in val_img_dir.iterdir() if p.suffix.lower() in {ext.lower() for ext in args.ext}])
    random.shuffle(images)
    n_move = max(1, int(len(images) * args.ratio))
    to_move = images[:n_move]

    print(f"Moving {len(to_move)} of {len(images)} files ({args.ratio*100:.1f}%) from {val_img_dir} to {test_img_dir}")
    for img_path in to_move:
        label_path = val_lbl_dir / f"{img_path.stem}.txt"
        shutil.move(str(img_path), test_img_dir / img_path.name)
        if label_path.exists():
            shutil.move(str(label_path), test_lbl_dir / label_path.name)


if __name__ == "__main__":
    main()
