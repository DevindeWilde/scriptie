#!/usr/bin/env python
"""Materialize stage-specific DOTA datasets (images + filtered labels)."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List, Set

STAGE_CLASSES: Dict[str, Set[int]] = {
    "stage1": {10, 9, 11, 12, 16, 17},
    "stage2": {3, 4, 5, 6, 13, 14},
    "stage3": {0, 1, 2, 7, 8, 15},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stage-specific DOTA datasets with filtered labels.")
    parser.add_argument("--root", type=Path, required=True, help="Path to YOLO-format DOTA dataset (images/ + labels/).")
    parser.add_argument("--splits", type=str, nargs="+", default=("train", "val", "test"), help="Splits to process.")
    parser.add_argument(
        "--dst-base",
        type=Path,
        default=None,
        help="Base directory to place stage datasets (defaults to root's parent).",
    )
    parser.add_argument("--copy-images", action="store_true", help="Physically copy image directories instead of symlinks.")
    parser.add_argument("--suffix", type=str, default="", help="Optional suffix appended to stage folder names.")
    return parser.parse_args()


def resolve_split_dir(root: Path, split: str, kind: str) -> Path:
    candidates = [root / kind / split, root / split / kind]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing directory for {kind}/{split} under {root}")


def ensure_images(src: Path, dst: Path, copy_images: bool) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_images:
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        os.symlink(src.resolve(), dst)


def filter_labels(src_file: Path, allowed: Set[int]) -> List[str]:
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
    return lines_out


def write_stage_labels(src_dir: Path, dst_dir: Path, allowed: Set[int]) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    label_files = sorted(src_dir.glob("*.txt"))
    for label_path in label_files:
        filtered = filter_labels(label_path, allowed)
        dst_file = dst_dir / label_path.name
        with dst_file.open("w") as out:
            out.write("\n".join(filtered))


def stage_dataset_root(base_root: Path, stage_name: str, dst_base: Path | None) -> Path:
    base = dst_base if dst_base else base_root.parent
    suffix = f"-{stage_name}" if stage_name else ""
    return base / f"{base_root.name}{suffix}"


def main():
    args = parse_args()
    root = args.root.resolve()
    dst_base = args.dst_base.resolve() if args.dst_base else None

    for stage, allowed in STAGE_CLASSES.items():
        tag = f"{stage}{('-' + args.suffix) if args.suffix else ''}"
        stage_root = stage_dataset_root(root, tag, dst_base)
        print(f"Building {stage_root}")
        for split in args.splits:
            try:
                src_img_dir = resolve_split_dir(root, split, "images")
                src_lbl_dir = resolve_split_dir(root, split, "labels")
            except FileNotFoundError as err:
                print(err)
                continue
            dst_split_dir = stage_root / split
            dst_img_dir = dst_split_dir / "images"
            dst_lbl_dir = dst_split_dir / "labels"
            ensure_images(src_img_dir, dst_img_dir, args.copy_images)
            write_stage_labels(src_lbl_dir, dst_lbl_dir, allowed)
    print("Stage dataset generation complete.")


if __name__ == "__main__":
    main()
