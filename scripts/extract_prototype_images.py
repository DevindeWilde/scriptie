"""Extract source images for each fine prototype slot and draw bounding boxes.

For each (class, FPN level, slot) triple in a saved replay buffer, this script:
  1. Loads the source image that initialized/reinitialized that prototype slot
  2. Reads the YOLO label file to recover bounding boxes
  3. Matches the specific object using the stored max_edge
  4. Draws the matched bbox in green (other same-class objects in gray)
  5. Saves the annotated image into an organized folder

Usage:
    python scripts/extract_prototype_images.py \
        --buffer runs/dota_saab/stage3-saab/replay/buffer.pt \
        --outdir prototype_images \
        --names small-vehicle,large-vehicle,plane,helicopter,ship
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont


def get_label_path(im_file: str) -> Path:
    """Derive YOLO label path from image path: images/ → labels/, .ext → .txt."""
    p = Path(im_file)
    label_dir = p.parent.parent / "labels" / p.stem
    # Standard YOLO layout: .../images/X.png → .../labels/X.txt
    return Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt")


def parse_yolo_labels(label_path: Path, cls_id: int) -> list[tuple[float, float, float, float]]:
    """Read YOLO label file and return normalized (cx, cy, w, h) for matching class."""
    if not label_path.exists():
        return []
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            if int(parts[0]) == cls_id:
                boxes.append(tuple(float(x) for x in parts[1:5]))
    return boxes


def match_by_max_edge(
    boxes_norm: list[tuple[float, float, float, float]],
    stored_max_edge: float,
    imgsz: int,
) -> int:
    """Find the box whose max(w, h) in pixel space best matches stored_max_edge."""
    if not boxes_norm:
        return -1
    best_idx = 0
    best_diff = float("inf")
    for i, (cx, cy, w, h) in enumerate(boxes_norm):
        me = max(w * imgsz, h * imgsz)
        diff = abs(me - stored_max_edge)
        if diff < best_diff:
            best_diff = diff
            best_idx = i
    return best_idx


def norm_to_pixel(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int):
    """Convert normalized xywh to pixel xyxy."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return x1, y1, x2, y2


def draw_annotated(
    image: Image.Image,
    boxes_norm: list[tuple[float, float, float, float]],
    match_idx: int,
    cls_name: str,
    level: str,
    slot: int,
    count: int,
    epoch: int,
) -> Image.Image:
    """Draw bboxes on image: green for matched object, gray for others."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size

    # Draw non-matched objects first (gray, thin)
    for i, (cx, cy, w, h) in enumerate(boxes_norm):
        if i == match_idx:
            continue
        x1, y1, x2, y2 = norm_to_pixel(cx, cy, w, h, img_w, img_h)
        draw.rectangle([x1, y1, x2, y2], outline=(160, 160, 160), width=1)

    # Draw matched object (green, thick)
    if 0 <= match_idx < len(boxes_norm):
        cx, cy, w, h = boxes_norm[match_idx]
        x1, y1, x2, y2 = norm_to_pixel(cx, cy, w, h, img_w, img_h)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)

        # Label
        label = f"{cls_name} | {level} slot{slot} | count={count} epoch={epoch}"
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except (IOError, OSError):
            font = ImageFont.load_default()
        # Background for text
        bbox = draw.textbbox((x1, max(0, y1 - 20)), label, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0))
        draw.text((x1, max(0, y1 - 20)), label, fill=(0, 255, 0), font=font)

    return img


def main():
    parser = argparse.ArgumentParser(description="Extract prototype source images with bbox overlay")
    parser.add_argument("--buffer", type=str, required=True, help="Path to buffer.pt")
    parser.add_argument("--outdir", type=str, default="prototype_images", help="Output directory")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size (for max_edge matching)")
    parser.add_argument("--names", type=str, default=None,
                        help="Comma-separated class names (e.g. small-vehicle,large-vehicle,...)")
    args = parser.parse_args()

    state = torch.load(args.buffer, map_location="cpu")
    storage = state.get("storage", {})
    if not storage:
        print("ERROR: No storage found in buffer")
        return

    # Build class name map
    name_map: dict[int, str] = {}
    if args.names:
        for i, name in enumerate(args.names.split(",")):
            name_map[i] = name.strip()

    outdir = Path(args.outdir)
    total = 0
    skipped = 0

    print(f"{'Level':<6} {'Cls':<4} {'Slot':<5} {'Count':<6} {'Epoch':<6} {'Source image'}")
    print("-" * 90)

    for level, class_map in sorted(storage.items()):
        for cls_id_str, entry in sorted(class_map.items(), key=lambda x: int(x[0])):
            cls_id = int(cls_id_str)
            cls_name = name_map.get(cls_id, f"class{cls_id}")
            fine_slots = entry.get("fine", [])

            for slot_idx, slot in enumerate(fine_slots):
                source = slot.get("source")
                if source is None:
                    skipped += 1
                    continue

                im_file = source.get("im_file")
                max_edge = source.get("max_edge", 0.0)
                epoch = source.get("epoch", -1)
                count = slot.get("count", 0)

                if not im_file:
                    skipped += 1
                    continue

                im_path = Path(im_file)
                stem = im_path.stem
                print(f"{level:<6} {cls_id:<4} {slot_idx:<5} {count:<6} {epoch:<6} {im_path.name}")

                # Load image
                if not im_path.exists():
                    print(f"  WARNING: image not found: {im_path}")
                    skipped += 1
                    continue

                image = Image.open(im_path).convert("RGB")

                # Try stored bbox first (available after train.py bbox fix)
                bbox_stored = source.get("bbox_xywh_norm")
                if bbox_stored:
                    boxes = [tuple(bbox_stored)]
                    match_idx = 0
                else:
                    # Fallback: label file lookup (won't work for mosaic images)
                    label_path = get_label_path(im_file)
                    boxes = parse_yolo_labels(label_path, cls_id)
                    match_idx = match_by_max_edge(boxes, max_edge, args.imgsz)
                    if not boxes:
                        print(f"  WARNING: no bbox in source and no labels for cls={cls_id}")

                # Draw and save
                annotated = draw_annotated(
                    image, boxes, match_idx, cls_name, level, slot_idx, count, epoch
                )

                save_dir = outdir / cls_name / level
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"slot{slot_idx}_{stem}_epoch{epoch}.png"
                annotated.save(save_path)
                total += 1

    print("-" * 90)
    print(f"Saved {total} annotated images to {outdir}/  (skipped {skipped})")


if __name__ == "__main__":
    main()
