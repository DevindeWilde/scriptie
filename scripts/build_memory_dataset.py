#!/usr/bin/env python3
"""Build a tiny exemplar dataset by sampling K images per class."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Set

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create cumulative exemplar replay datasets for continual learning.")
    p.add_argument("--root", type=Path, required=True,
                   help="Dataset root containing Stage 1/, Stage 2/, ... (e.g. dota_full).")
    p.add_argument("--dest", type=Path, required=True,
                   help="Destination root for memory/ (will create memory/stageX/images and labels).")
    p.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3],
                   help="Stage numbers to build memory for (default: 1 2 3).")
    p.add_argument("--split", type=str, default="Train",
                   help="Split folder name inside each stage (default: Train).")
    p.add_argument("--classes", type=int, nargs="+", required=True,
                   help="Class IDs to keep in memory labels (filtered).")
    p.add_argument("--per-class", type=int, default=25, dest="per_class",
                   help="Images per class to sample from each stage (default: 25).")
    p.add_argument("--seed", type=int, default=32,
                   help="Random seed for reproducibility.")
    return p.parse_args()


def find_image_for_label(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None

def read_present_classes(label_path: Path) -> Set[int]:
    present: Set[int] = set()
    with label_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cls_id = int(float(line.split()[0]))
            present.add(cls_id)
    return present

def build_class_to_files(labels_dir: Path, classes: Iterable[int]) -> Dict[int, List[Path]]:
    class_to_files: Dict[int, List[Path]] = {c: [] for c in classes}
    for label_path in sorted(labels_dir.glob("*.txt")):
        present = read_present_classes(label_path)
        for c in class_to_files.keys():
            if c in present:
                class_to_files[c].append(label_path)
    return class_to_files

def sample_label_files(class_to_files: Dict[int, List[Path]], per_class: int, rng: random.Random) -> Set[Path]:
    selected: Set[Path] = set()
    for cls, files in class_to_files.items():
        if not files:
            continue
        if len(files) <= per_class:
            chosen = files
        else:
            chosen = rng.sample(files, per_class)
        selected.update(chosen)
    return selected

def copy_previous_memory(prev_stage_dir: Path, cur_stage_dir: Path) -> None:
    """Copy prev memory/stageK/{images,labels} into current memory/stage(K+1)/{images,labels}."""
    prev_images = prev_stage_dir / "images"
    prev_labels = prev_stage_dir / "labels"
    cur_images = cur_stage_dir / "images"
    cur_labels = cur_stage_dir / "labels"

    cur_images.mkdir(parents=True, exist_ok=True)
    cur_labels.mkdir(parents=True, exist_ok=True)

    if prev_images.exists():
        for p in prev_images.iterdir():
            if p.is_file():
                dst = cur_images / p.name
                if not dst.exists():
                    shutil.copy2(p, dst)

    if prev_labels.exists():
        for p in prev_labels.iterdir():
            if p.is_file():
                dst = cur_labels / p.name
                if not dst.exists():
                    shutil.copy2(p, dst)


def write_filtered_label(src_label: Path, dst_label: Path, kept_classes: Set[int]) -> None:
    with src_label.open("r") as src, dst_label.open("w") as dst:
        for line in src:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(float(parts[0]))
            if cls_id in kept_classes:
                dst.write(line)
    
def add_stage_exemplars(
    stage_images: Path,
    stage_labels: Path,
    dest_stage_dir: Path,
    kept_classes: Set[int],
    per_class: int,
    rng: random.Random,
) -> Tuple[int, int]:
    """Sample from stage Train, then add into dest stage memory folder (skip duplicates)."""
    dest_images = dest_stage_dir / "images"
    dest_labels = dest_stage_dir / "labels"
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    class_to_files = build_class_to_files(stage_labels, kept_classes)
    selected_labels = sample_label_files(class_to_files, per_class=per_class, rng=rng)

    copied = 0
    skipped_missing_img = 0

    for label_path in selected_labels:
        stem = label_path.stem
        img_path = find_image_for_label(stage_images, stem)
        if img_path is None:
            skipped_missing_img += 1
            continue

        # Avoid duplicates by filename (stem match) in current memory stage
        dst_img = dest_images / img_path.name
        dst_lbl = dest_labels / label_path.name

        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)

        # Always (re)write filtered label for current stage memory if not exists
        # (If it already existed from previous stage copy, we leave it untouched.)
        if not dst_lbl.exists():
            write_filtered_label(label_path, dst_lbl, kept_classes)

        copied += 1

    return copied, skipped_missing_img


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    kept_classes = set(args.classes)

    root = args.root.resolve()
    memory_root = (args.dest / "memory").resolve()
    memory_root.mkdir(parents=True, exist_ok=True)

    stages = list(args.stages)
    stages.sort()

    prev_memory_stage_dir: Path | None = None

    for stage_num in stages:
        stage_dir = root / f"Stage {stage_num}" / args.split
        stage_images = stage_dir / "images"
        stage_labels = stage_dir / "labels"

        if not stage_images.exists() or not stage_labels.exists():
            raise FileNotFoundError(f"Missing images/labels in: {stage_dir}")

        cur_memory_stage_dir = memory_root / f"stage{stage_num}"

        # Step 1: initialize current stage memory by copying previous stage memory (cumulative)
        if prev_memory_stage_dir is not None:
            copy_previous_memory(prev_memory_stage_dir, cur_memory_stage_dir)
        else:
            (cur_memory_stage_dir / "images").mkdir(parents=True, exist_ok=True)
            (cur_memory_stage_dir / "labels").mkdir(parents=True, exist_ok=True)

        # Step 2: add new exemplars from this stage
        copied, missing = add_stage_exemplars(
            stage_images=stage_images,
            stage_labels=stage_labels,
            dest_stage_dir=cur_memory_stage_dir,
            kept_classes=kept_classes,
            per_class=args.per_class,
            rng=rng,
        )

        n_imgs = len(list((cur_memory_stage_dir / "images").glob("*")))
        n_lbls = len(list((cur_memory_stage_dir / "labels").glob("*.txt")))

        print(
            f"[stage{stage_num}] added_from_stage={copied}, missing_images={missing} | "
            f"memory_now: images={n_imgs}, labels={n_lbls} -> {cur_memory_stage_dir}"
        )

        prev_memory_stage_dir = cur_memory_stage_dir

    print(f"Done. Memory datasets written under: {memory_root}")


if __name__ == "__main__":
    main()
