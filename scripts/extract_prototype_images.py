"""Extract source crops for each fine prototype slot and draw bounding boxes.

For each (class, FPN level, slot) triple in a saved replay buffer, this script:
  1. Loads the stored source crop (from the augmented batch image at training time)
  2. Draws the bbox overlay (green) — both crop and bbox are in the same coordinate frame
  3. Saves the annotated crop into an organized folder

Falls back to loading the original image (without bbox) for older buffers that
don't have source_crop stored.

Usage:
    python scripts/extract_prototype_images.py \
        --buffer runs/dota_saab/stage3-saab/replay/buffer.pt \
        --outdir prototype_images \
        --names small-vehicle,large-vehicle,plane,helicopter,ship
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description="Extract prototype source images with bbox overlay")
    parser.add_argument("--buffer", type=str, required=True, help="Path to buffer.pt")
    parser.add_argument("--outdir", type=str, default="prototype_images", help="Output directory")
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
    print(f"{'Level':<6} {'Cls':<4} {'Slot':<5} {'Count':<6} {'Epoch':<6} {'Source'}")
    print("-" * 80)

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

                epoch = source.get("epoch", -1)
                count = slot.get("count", 0)
                im_file = source.get("im_file", "")
                stem = Path(im_file).stem if im_file else "unknown"

                # --- Preferred: use stored crop from augmented batch image ---
                source_crop = source.get("source_crop")
                bbox_in_crop = source.get("bbox_in_crop")

                if source_crop is not None:
                    print(f"{level:<6} {cls_id:<4} {slot_idx:<5} {count:<6} {epoch:<6} crop")
                    # source_crop is (3, H, W) uint8 tensor
                    img = Image.fromarray(source_crop.permute(1, 2, 0).numpy())
                    draw = ImageDraw.Draw(img)
                    if bbox_in_crop:
                        draw.rectangle(bbox_in_crop, outline=(0, 255, 0), width=2)

                # --- Fallback: load original image (no reliable bbox) ---
                elif im_file:
                    im_path = Path(im_file)
                    print(f"{level:<6} {cls_id:<4} {slot_idx:<5} {count:<6} {epoch:<6} {im_path.name}")
                    if not im_path.exists():
                        print(f"  WARNING: image not found: {im_path}")
                        skipped += 1
                        continue
                    img = Image.open(im_path).convert("RGB")
                else:
                    skipped += 1
                    continue

                save_dir = outdir / cls_name / level
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"slot{slot_idx}_{stem}_epoch{epoch}.png"
                img.save(save_path)
                total += 1

    print("-" * 80)
    print(f"Saved {total} annotated images to {outdir}/  (skipped {skipped})")


if __name__ == "__main__":
    main()
