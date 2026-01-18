#!/usr/bin/env python3
"""Build stage-wise datasets with per-stage train filtering and cumulative validation labels."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

DEFAULT_STAGES = {
    "stage1": [0, 1],
    "stage2": [2, 3],
    "stage3": [4],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Root dataset with train/val/test subfolders")
    parser.add_argument("output", type=Path, help="Destination root for stage datasets")
    parser.add_argument(
        "--stage",
        action="append",
        help="Optional stage definition like stage1:0,1, overrides defaults",
    )
    return parser.parse_args()


def load_stages(raw: List[str] | None) -> Dict[str, List[int]]:
    if not raw:
        return DEFAULT_STAGES
    stages: Dict[str, List[int]] = {}
    for entry in raw:
        name, cls_part = entry.split(":", 1)
        class_ids = [int(x) for x in cls_part.split(",") if x != ""]
        stages[name] = class_ids
    return stages


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def filter_label(label_path: Path, allowed: set[int]) -> List[str]:
    kept: List[str] = []
    text = label_path.read_text().strip()
    if not text:
        return kept
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            cls_id = int(float(parts[0]))
        except ValueError:
            continue
        if cls_id in allowed:
            kept.append(" ".join([str(cls_id)] + parts[1:]))
    return kept


def write_label(dest: Path, lines: List[str]):
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + ("\n" if lines else ""))


def copy_image(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def build_stage(stage_name: str, current_classes: List[int], source_root: Path, output_root: Path, cumulative_classes: set[int]):
    stage_root = output_root / stage_name
    if stage_root.exists():
        shutil.rmtree(stage_root)
    train_src_imgs = source_root / "train" / "images"
    train_src_lbls = source_root / "train" / "labels"

    stage_train_img = stage_root / "train" / "images"
    stage_train_lbl = stage_root / "train" / "labels"
    stage_val_img = stage_root / "val" / "images"
    stage_val_lbl = stage_root / "val" / "labels"
    stage_val_seen_img = stage_root / "val_seen" / "images"
    stage_val_seen_lbl = stage_root / "val_seen" / "labels"
    stage_test_img = stage_root / "test" / "images"
    stage_test_lbl = stage_root / "test" / "labels"

    allowed = set(current_classes)
    train_count = 0
    for label_path in sorted(train_src_lbls.glob("*.txt")):
        lines = filter_label(label_path, allowed)
        if not lines:
            continue
        img_path = find_image(train_src_imgs, label_path.stem)
        if not img_path:
            print(f"[WARN] Missing train image for {label_path}")
            continue
        copy_image(img_path, stage_train_img / img_path.name)
        write_label(stage_train_lbl / label_path.name, lines)
        train_count += 1

    # Copy full val/test
    copy_tree(source_root / "val" / "images", stage_val_img)
    copy_tree(source_root / "val" / "labels", stage_val_lbl)
    copy_tree(source_root / "test" / "images", stage_test_img)
    copy_tree(source_root / "test" / "labels", stage_test_lbl)

    # Build cumulative val_seen labels
    val_imgs = source_root / "val" / "images"
    val_lbls = source_root / "val" / "labels"
    for label_path in sorted(val_lbls.glob("*.txt")):
        lines = filter_label(label_path, cumulative_classes)
        img_path = find_image(val_imgs, label_path.stem)
        if not img_path:
            continue
        copy_image(img_path, stage_val_seen_img / img_path.name)
        write_label(stage_val_seen_lbl / label_path.name, lines if lines else [])

    print(
        f"{stage_name}: wrote {train_count} train samples using classes {sorted(allowed)}; "
        f"val/test copied, val_seen labels cover classes {sorted(cumulative_classes)}"
    )


def main():
    args = parse_args()
    stages = load_stages(args.stage)
    cumulative: set[int] = set()
    for stage_name, classes in stages.items():
        cumulative.update(classes)
        build_stage(stage_name, classes, args.source, args.output, cumulative.copy())


if __name__ == "__main__":
    main()
