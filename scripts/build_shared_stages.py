#!/usr/bin/env python3
"""Build shared-image-pool stage datasets for continual object detection.

All stages share the same image pool (symlinked). Labels are filtered per stage:
  train/labels  → new-class annotations only  (model only trained on new classes)
  val/labels    → cumulative-class annotations (forgetting measured on all seen classes)
  test/labels   → cumulative-class annotations

Stages 2+ also get a train_replay/ split where a small set of replay images
has old-class annotations merged back in alongside new-class annotations.

Usage:
    python scripts/build_shared_stages.py
    python scripts/build_shared_stages.py --src /path/to/source --dst /path/to/output
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# ──────────────────────────────── Configuration ───────────────────────────────

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

CLASS_NAMES: Dict[int, str] = {
    0: "small-vehicle",
    1: "large-vehicle",
    2: "plane",
    3: "helicopter",
    4: "ship",
}

# new_classes: classes introduced this stage
# cumulative:  all classes seen up to and including this stage
STAGE_DEFS = [
    {"name": "stage1", "new": [0, 1],  "cumulative": [0, 1]},
    {"name": "stage2", "new": [2, 3],  "cumulative": [0, 1, 2, 3]},
    {"name": "stage3", "new": [4],     "cumulative": [0, 1, 2, 3, 4]},
]

REPLAY_PER_CLASS = 25   # exemplar images per old class

# ──────────────────────────────── Label I/O ───────────────────────────────────

def read_labels(path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Return list of (cls, cx, cy, w, h) from a YOLO label file."""
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows = []
    with path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                rows.append((int(parts[0]),
                              float(parts[1]), float(parts[2]),
                              float(parts[3]), float(parts[4])))
    return rows


def write_labels(path: Path, labels: List[Tuple]) -> None:
    """Write YOLO labels to file. Writes empty file when labels is empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for cls, cx, cy, w, h in labels:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def filter_labels(
    labels: List[Tuple], keep: List[int]
) -> List[Tuple]:
    keep_set = set(keep)
    return [row for row in labels if row[0] in keep_set]


def img_to_label(img_path: Path) -> Path:
    """Derive label path from image path: images/ → labels/, .<ext> → .txt"""
    parts = list(img_path.parts)
    # Replace the 'images' segment with 'labels'
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


# ──────────────────────────────── Copy helper ─────────────────────────────────

def copy_images(src: Path, dst: Path) -> None:
    """Copy all image files from src/ into dst/, skipping existing files."""
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in src.iterdir():
        if f.suffix.lower() in IMG_EXTS:
            dest_file = dst / f.name
            if not dest_file.exists():
                shutil.copy2(f, dest_file)
                copied += 1
    if copied:
        print(f"    Copied {copied} images → {dst}")


# ──────────────────────────────── YAML writer ─────────────────────────────────

def write_yaml(path: Path, dataset_root: Path, train_dir: str,
               cumulative: List[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"path: {dataset_root}",
        f"train: {train_dir}",
        "val: val/images",
        "test: test/images",
        "names:",
    ]
    for cid in cumulative:
        lines.append(f"  {cid}: {CLASS_NAMES[cid]}")
    lines.append("")
    path.write_text("\n".join(lines))


# ──────────────────────────────── Core build ──────────────────────────────────

def collect_images(images_dir: Path) -> List[Path]:
    """Return sorted list of image files under images_dir."""
    return sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)


def build_label_split(
    src_labels_dir: Path,
    dst_labels_dir: Path,
    img_stems: List[str],
    keep_classes: List[int],
    stats: Dict,
    split_key: str,
) -> None:
    """Filter source labels to `keep_classes` and write to dst_labels_dir.

    Creates an empty .txt for every image stem (no missing label files).
    Updates `stats[split_key]` with annotation counts.
    """
    keep_set = set(keep_classes)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    has_annots = 0
    cls_counts: Dict[int, int] = defaultdict(int)

    for stem in img_stems:
        src_lbl = src_labels_dir / (stem + ".txt")
        raw = read_labels(src_lbl)
        filtered = [(c, cx, cy, w, h) for c, cx, cy, w, h in raw if c in keep_set]
        write_labels(dst_labels_dir / (stem + ".txt"), filtered)
        if filtered:
            has_annots += 1
            for c, *_ in filtered:
                cls_counts[c] += 1

    stats[split_key] = {
        "total_files": len(img_stems),
        "with_annotations": has_annots,
        "empty": len(img_stems) - has_annots,
        "class_counts": dict(cls_counts),
    }


def select_replay_images(
    src_train_labels_dir: Path,
    img_stems: List[str],
    old_classes: List[int],
    n_per_class: int,
) -> Dict[int, List[str]]:
    """Select top-n images per old class by annotation count.

    Returns {class_id: [stem, ...]} manifest.
    """
    # Count annotations per class per image
    cls_img_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for stem in img_stems:
        lbl = src_train_labels_dir / (stem + ".txt")
        for cls, *_ in read_labels(lbl):
            if cls in old_classes:
                cls_img_counts[cls][stem] += 1

    manifest: Dict[int, List[str]] = {}
    for cls in old_classes:
        counts = cls_img_counts[cls]
        ranked = sorted(counts.keys(), key=lambda s: -counts[s])
        manifest[cls] = ranked[:n_per_class]
    return manifest


def build_replay_labels(
    src_train_labels_dir: Path,
    stage_train_labels_dir: Path,
    dst_labels_dir: Path,
    img_stems: List[str],
    new_classes: List[int],
    old_classes: List[int],
    manifest: Dict[int, List[str]],
    stats: Dict,
) -> None:
    """Build train_replay/labels.

    For most images: same as stage train/labels (new classes only).
    For replay images: merge new-class annotations + old-class annotations
    from source.
    """
    # Build set of all replay stems
    replay_stems: Dict[str, List[int]] = defaultdict(list)
    for cls, stems in manifest.items():
        for s in stems:
            replay_stems[s].append(cls)

    dst_labels_dir.mkdir(parents=True, exist_ok=True)
    new_set = set(new_classes)
    old_set = set(old_classes)

    merged_total = 0
    old_cls_restored: Dict[int, int] = defaultdict(int)
    replay_img_set = set(replay_stems.keys())

    for stem in img_stems:
        new_annots = read_labels(stage_train_labels_dir / (stem + ".txt"))
        new_annots = [(c, cx, cy, w, h) for c, cx, cy, w, h in new_annots if c in new_set]

        if stem in replay_img_set:
            src_raw = read_labels(src_train_labels_dir / (stem + ".txt"))
            old_annots = [(c, cx, cy, w, h) for c, cx, cy, w, h in src_raw if c in old_set]
            merged = new_annots + old_annots
            for c, *_ in old_annots:
                old_cls_restored[c] += 1
            merged_total += len(old_annots)
        else:
            merged = new_annots

        write_labels(dst_labels_dir / (stem + ".txt"), merged)

    stats["replay_train"] = {
        "replay_images": len(replay_img_set),
        "old_class_annotations_restored": dict(old_cls_restored),
        "total_old_annotations": merged_total,
    }


# ──────────────────────────────── Main ───────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src",
        type=Path,
        default=Path("/home/ddwilde/scriptie/datasets/dota-filtered-full"),
        help="Source dataset root (default: %(default)s)",
    )
    p.add_argument(
        "--dst",
        type=Path,
        default=Path("/home/ddwilde/scriptie/datasets/dota_stages_shared"),
        help="Output root (default: %(default)s)",
    )
    p.add_argument(
        "--replay-per-class",
        type=int,
        default=REPLAY_PER_CLASS,
        help="Exemplar images per old class for replay split (default: %(default)s)",
    )
    return p.parse_args()


def main() -> None:
    opt = parse_args()
    src: Path = opt.src
    dst: Path = opt.dst
    n_replay = opt.replay_per_class

    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")

    dst.mkdir(parents=True, exist_ok=True)
    all_stats: Dict[str, Dict] = {}

    # ── Collect image stems per split ────────────────────────────────────────
    split_stems: Dict[str, List[str]] = {}
    for split in ("train", "val", "test"):
        imgs = collect_images(src / split / "images")
        split_stems[split] = [p.stem for p in imgs]
        print(f"  {split}: {len(imgs)} images")

    # ── Build per-stage datasets ──────────────────────────────────────────────
    for stage_idx, sdef in enumerate(STAGE_DEFS):
        stage_name = sdef["name"]
        new_cls   = sdef["new"]
        cum_cls   = sdef["cumulative"]
        old_cls   = [c for c in cum_cls if c not in new_cls]  # empty for stage1

        stage_dir = dst / stage_name
        stats: Dict = {}
        print(f"\n{'─'*60}")
        print(f"  {stage_name}  new={new_cls}  cumulative={cum_cls}")
        print(f"{'─'*60}")

        # ── Copy image directories ────────────────────────────────────────────
        for split in ("train", "val", "test"):
            copy_images(src / split / "images",
                        stage_dir / split / "images")

        # ── train/labels → new classes only ──────────────────────────────────
        print(f"  Building train/labels (new classes {new_cls}) ...")
        build_label_split(
            src_labels_dir=src / "train" / "labels",
            dst_labels_dir=stage_dir / "train" / "labels",
            img_stems=split_stems["train"],
            keep_classes=new_cls,
            stats=stats,
            split_key="train",
        )

        # ── val/labels and test/labels → cumulative classes ───────────────────
        for split in ("val", "test"):
            print(f"  Building {split}/labels (cumulative classes {cum_cls}) ...")
            build_label_split(
                src_labels_dir=src / split / "labels",
                dst_labels_dir=stage_dir / split / "labels",
                img_stems=split_stems[split],
                keep_classes=cum_cls,
                stats=stats,
                split_key=split,
            )

        # ── YAML: stage{i}.yaml ───────────────────────────────────────────────
        yaml_path = dst / f"{stage_name}.yaml"
        write_yaml(yaml_path, stage_dir, "train/images", cum_cls)
        print(f"  Wrote {yaml_path.name}")

        # ── Replay split (stages 2+) ──────────────────────────────────────────
        if old_cls:
            print(f"  Selecting replay images ({n_replay} per class, old={old_cls}) ...")
            manifest = select_replay_images(
                src_train_labels_dir=src / "train" / "labels",
                img_stems=split_stems["train"],
                old_classes=old_cls,
                n_per_class=n_replay,
            )

            # Save manifest
            manifest_path = stage_dir / "replay_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with manifest_path.open("w") as f:
                json.dump({str(k): v for k, v in manifest.items()}, f, indent=2)
            print(f"  Saved {manifest_path.name}")

            # train_replay/images is the same pool — hardlink to avoid re-copying
            # (falls back to copy if hardlinks unsupported)
            replay_img_dst = stage_dir / "train_replay" / "images"
            replay_img_dst.mkdir(parents=True, exist_ok=True)
            train_img_dst = stage_dir / "train" / "images"
            linked = 0
            for f in train_img_dst.iterdir():
                dest = replay_img_dst / f.name
                if not dest.exists():
                    try:
                        os.link(f, dest)
                    except OSError:
                        shutil.copy2(f, dest)
                    linked += 1
            if linked:
                print(f"    Linked/copied {linked} images → {replay_img_dst}")

            print(f"  Building train_replay/labels ...")
            build_replay_labels(
                src_train_labels_dir=src / "train" / "labels",
                stage_train_labels_dir=stage_dir / "train" / "labels",
                dst_labels_dir=stage_dir / "train_replay" / "labels",
                img_stems=split_stems["train"],
                new_classes=new_cls,
                old_classes=old_cls,
                manifest=manifest,
                stats=stats,
            )

            # YAML: stage{i}_replay.yaml
            replay_yaml_path = dst / f"{stage_name}_replay.yaml"
            write_yaml(replay_yaml_path, stage_dir, "train_replay/images", cum_cls)
            print(f"  Wrote {replay_yaml_path.name}")

        all_stats[stage_name] = stats

    # ── Print statistics ──────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  STATISTICS")
    print(f"{'═'*60}")

    for stage_name, stats in all_stats.items():
        print(f"\n  {stage_name}")
        for split in ("train", "val", "test"):
            if split not in stats:
                continue
            s = stats[split]
            print(f"    {split:12s}  total={s['total_files']:5d}  "
                  f"with_annots={s['with_annotations']:5d}  "
                  f"empty={s['empty']:5d}")
            for cid in sorted(s["class_counts"].keys()):
                print(f"      class {cid} ({CLASS_NAMES.get(cid, '?'):15s}): "
                      f"{s['class_counts'][cid]:6d} annotations")

        if "replay_train" in stats:
            r = stats["replay_train"]
            print(f"    replay        replay_images={r['replay_images']:4d}  "
                  f"total_old_annots={r['total_old_annotations']:6d}")
            for cid in sorted(r["old_class_annotations_restored"].keys()):
                print(f"      class {cid} ({CLASS_NAMES.get(cid, '?'):15s}) restored: "
                      f"{r['old_class_annotations_restored'][cid]:6d}")

    # Save stats JSON
    stats_path = dst / "build_stats.json"
    with stats_path.open("w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n  Stats saved to {stats_path}")
    print(f"\n  Done. Output: {dst}")


if __name__ == "__main__":
    main()
