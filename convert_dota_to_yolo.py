#!/usr/bin/env python
"""Convert DOTA oriented bounding-box annotations to YOLO (HBB) format."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image
from tqdm import tqdm


# Official DOTA class order
CLASS_MAPPING: Dict[str, int] = {
    "plane": 0,
    "ship": 1,
    "storage-tank": 2,
    "baseball-diamond": 3,
    "tennis-court": 4,
    "basketball-court": 5,
    "ground-track-field": 6,
    "harbor": 7,
    "bridge": 8,
    "large-vehicle": 9,
    "small-vehicle": 10,
    "helicopter": 11,
    "roundabout": 12,
    "soccer-ball-field": 13,
    "swimming-pool": 14,
    "container-crane": 15,
    "airport": 16,
    "helipad": 17,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DOTA annotations to YOLO (xc, yc, w, h) format.")
    parser.add_argument("--src", type=Path, required=True, help="Root directory of the raw DOTA dataset.")
    parser.add_argument(
        "--dst",
        type=Path,
        required=True,
        help="Output root directory where YOLO-style images/labels will be written.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=("train", "val"),
        help="Dataset splits to convert (default: train val).",
    )
    parser.add_argument(
        "--image-subdir",
        type=str,
        default="images",
        help="Sub-directory that stores images (e.g., 'images' or 'Img').",
    )
    parser.add_argument(
        "--label-subdir",
        type=str,
        default="labelTxt",
        help="Sub-directory that stores oriented annotations.",
    )
    parser.add_argument(
        "--skip-difficult",
        action="store_true",
        help="Drop annotations whose difficulty flag is 1.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy image files instead of creating symlinks in the destination directory.",
    )
    return parser.parse_args()


def resolve_split_dir(root: Path, split: str, subdir: str) -> Path:
    """Return the existing directory that holds split data."""
    candidates = [
        root / subdir / split,
        root / split / subdir,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find directory for split '{split}' under '{root}' (searched {candidates})")


def oriented_to_yolo(box: List[float], width: int, height: int) -> Tuple[float, float, float, float]:
    xs = box[0::2]
    ys = box[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = max(x_max - x_min, 1e-6)
    h = max(y_max - y_min, 1e-6)
    cx = x_min + w / 2
    cy = y_min + h / 2
    return cx / width, cy / height, w / width, h / height


def read_dota_label(path: Path, skip_difficult: bool) -> Iterable[Tuple[int, List[float]]]:
    with path.open("r") as f:
        for raw in f:
            parts = raw.strip().split()
            if len(parts) < 9 or parts[0].startswith("imagesource") or parts[0].startswith("gsd"):
                continue
            cls_name = parts[8]
            if cls_name == "###":
                continue
            difficulty = int(parts[9]) if len(parts) > 9 else 0
            if skip_difficult and difficulty == 1:
                continue
            if cls_name not in CLASS_MAPPING:
                continue
            coords = [float(v) for v in parts[:8]]
            yield CLASS_MAPPING[cls_name], coords


def ensure_image(dest: Path, src: Path, copy: bool) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copyfile(src, dest)
    else:
        dest.symlink_to(src.resolve())


def convert_split(split: str, args: argparse.Namespace) -> None:
    src_img_dir = resolve_split_dir(args.src, split, args.image_subdir)
    src_label_dir = resolve_split_dir(args.src, split, args.label_subdir)

    dst_img_dir = args.dst / "images" / split
    dst_lbl_dir = args.dst / "labels" / split
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted([p for p in src_img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif"}])
    for img_path in tqdm(image_files, desc=f"[{split}] converting"):
        label_path = src_label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        with Image.open(img_path) as im:
            width, height = im.size

        yolo_lines = []
        for cls_id, coords in read_dota_label(label_path, skip_difficult=args.skip_difficult):
            cx, cy, w, h = oriented_to_yolo(coords, width, height)
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            w = min(max(w, 0.0), 1.0)
            h = min(max(h, 0.0), 1.0)
            if w <= 0 or h <= 0:
                continue
            yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if yolo_lines:
            ensure_image(dst_img_dir / img_path.name, img_path, copy=args.copy_images)
            with (dst_lbl_dir / f"{img_path.stem}.txt").open("w") as out:
                out.write("\n".join(yolo_lines))


def main():
    args = parse_args()
    for split in args.splits:
        convert_split(split, args)
    print(f"Conversion finished. YOLO data written to {args.dst}")


if __name__ == "__main__":
    main()
