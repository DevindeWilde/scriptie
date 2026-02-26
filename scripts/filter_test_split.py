"""
Filter a YOLO test folder to keep only specified classes.

Copies images whose labels contain at least one kept class,
and strips out label rows for non-kept classes.  Class IDs
are remapped to 0..N-1 in the order given by --classes.

Usage:
    python scripts/filter_test_split.py \
        --src datasets/EO/test \
        --dst datasets/EO/test_seen \
        --classes 0 1 2 3 4 5
"""

import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source test folder (contains images/ and labels/)")
    ap.add_argument("--dst", required=True, help="Destination folder")
    ap.add_argument("--classes", nargs="+", type=int, required=True,
                    help="Class IDs to keep (will be remapped to 0..N-1 in this order)")
    ap.add_argument("--no-remap", action="store_true",
                    help="Keep original class IDs instead of remapping")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    keep = set(args.classes)
    remap = {c: i for i, c in enumerate(args.classes)} if not args.no_remap else {c: c for c in args.classes}

    src_imgs = src / "images"
    src_lbls = src / "labels"
    dst_imgs = dst / "images"
    dst_lbls = dst / "labels"
    dst_imgs.mkdir(parents=True, exist_ok=True)
    dst_lbls.mkdir(parents=True, exist_ok=True)

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    copied, skipped = 0, 0

    for lbl_file in sorted(src_lbls.glob("*.txt")):
        # read and filter label rows
        kept_lines = []
        for line in lbl_file.read_text().strip().splitlines():
            parts = line.split()
            cls_id = int(parts[0])
            if cls_id in keep:
                parts[0] = str(remap[cls_id])
                kept_lines.append(" ".join(parts))

        if not kept_lines:
            skipped += 1
            continue

        # find matching image
        stem = lbl_file.stem
        img_file = None
        for ext in img_exts:
            candidate = src_imgs / (stem + ext)
            if candidate.exists():
                img_file = candidate
                break

        if img_file is None:
            print(f"  WARN: no image for {lbl_file.name}, skipping")
            skipped += 1
            continue

        # write filtered label and copy image
        (dst_lbls / lbl_file.name).write_text("\n".join(kept_lines) + "\n")
        shutil.copy2(img_file, dst_imgs / img_file.name)
        copied += 1

    print(f"Done. Copied {copied} images+labels, skipped {skipped}.")
    print(f"  Classes kept: {args.classes}")
    if not args.no_remap:
        print(f"  Remap: {dict(sorted(remap.items()))}")
    print(f"  Output: {dst}")


if __name__ == "__main__":
    main()
