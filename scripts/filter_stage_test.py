#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_stage_classes(yaml_path: Path) -> set[int]:
    classes: set[int] = set()
    in_names = False
    for raw in yaml_path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("names:"):
            in_names = True
            continue
        if not in_names:
            continue
        if ":" not in line:
            continue
        key, _ = line.split(":", 1)
        try:
            classes.add(int(key.strip()))
        except ValueError:
            continue
    return classes


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def filter_labels(src_labels: Path, dst_labels: Path, allowed: set[int]) -> int:
    dst_labels.parent.mkdir(parents=True, exist_ok=True)
    if not src_labels.exists():
        return 0
    text = src_labels.read_text().strip()
    if not text:
        dst_labels.write_text("")
        return 0
    kept: list[str] = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            cls_id = int(float(parts[0]))
        except ValueError:
            continue
        if cls_id in allowed:
            kept.append(line)
    dst_labels.write_text("\n".join(kept) + ("\n" if kept else ""))
    return len(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter DOTA stage test labels to stage class ids.")
    parser.add_argument("--stage-root", type=Path, required=True, help="Stage root (e.g. datasets/dota_stages/stage1)")
    parser.add_argument("--yaml", type=Path, required=True, help="Stage YAML for class ids")
    parser.add_argument("--src-split", type=str, default="test", help="Source split name (default: test)")
    parser.add_argument("--dst-split", type=str, default="test_seen", help="Destination split name")
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        help="Optional explicit class ids to keep (overrides YAML names)",
    )
    parser.add_argument("--copy-images", action="store_true", help="Copy images into destination split")
    args = parser.parse_args()

    allowed = set(args.classes) if args.classes else load_stage_classes(args.yaml)
    if not allowed:
        raise SystemExit(f"No class ids found in {args.yaml}")

    src = args.stage_root / args.src_split
    dst = args.stage_root / args.dst_split
    src_labels = src / "labels"
    src_images = src / "images"
    dst_labels = dst / "labels"
    dst_images = dst / "images"
    dst_labels.mkdir(parents=True, exist_ok=True)
    if args.copy_images:
        dst_images.mkdir(parents=True, exist_ok=True)

    label_files = sorted(src_labels.glob("*.txt"))
    kept_total = 0
    for label_path in label_files:
        stem = label_path.stem
        kept = filter_labels(label_path, dst_labels / label_path.name, allowed)
        kept_total += kept
        if args.copy_images and src_images.exists():
            img_path = find_image(src_images, stem)
            if img_path is not None:
                (dst_images / img_path.name).write_bytes(img_path.read_bytes())

    print(
        f"Filtered {len(label_files)} label files from {src} -> {dst} "
        f"with classes {sorted(allowed)}. Kept {kept_total} labels."
    )


if __name__ == "__main__":
    main()
