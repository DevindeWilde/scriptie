#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def count_images(img_dir: Path) -> int:
    if not img_dir.exists():
        return 0
    return sum(1 for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)


def read_label_classes(label_path: Path) -> Iterable[int]:
    text = label_path.read_text().strip()
    if not text:
        return []
    classes = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            cls_id = int(float(parts[0]))
        except ValueError:
            continue
        classes.append(cls_id)
    return classes


def count_labels_and_classes(lbl_dir: Path) -> tuple[int, Counter]:
    if not lbl_dir.exists():
        return 0, Counter()
    label_files = [p for p in lbl_dir.iterdir() if p.suffix.lower() == ".txt"]
    class_counts = Counter()
    for p in label_files:
        class_counts.update(read_label_classes(p))
    return len(label_files), class_counts


def report_split(root: Path, split: str) -> tuple[int, int, Counter]:
    img_dir = root / split / "images"
    lbl_dir = root / split / "labels"
    return count_images(img_dir), *count_labels_and_classes(lbl_dir)


def format_counter(counter: Counter, name_map: dict[int, str] | None = None) -> str:
    if not counter:
        return "(none)"
    items = []
    for cls_id, count in sorted(counter.items()):
        label = name_map.get(cls_id, str(cls_id)) if name_map else str(cls_id)
        items.append(f"{label}:{count}")
    return ", ".join(items)


def load_names(yaml_path: Path) -> dict[int, str]:
    if not yaml_path.exists():
        return {}
    names: dict[int, str] = {}
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
        key, value = line.split(":", 1)
        try:
            idx = int(key.strip())
        except ValueError:
            continue
        names[idx] = value.strip()
    return names


def report_dataset(root: Path, splits: list[str], name_map: dict[int, str] | None = None) -> str:
    lines = []
    total_imgs = 0
    total_lbls = 0
    total_classes = Counter()
    for split in splits:
        img_count, lbl_count, class_counts = report_split(root, split)
        total_imgs += img_count
        total_lbls += lbl_count
        total_classes.update(class_counts)
        lines.append(
            f"  {split}: images={img_count} labels={lbl_count} classes=[{format_counter(class_counts, name_map)}]"
        )
    lines.append(
        f"  total: images={total_imgs} labels={total_lbls} classes=[{format_counter(total_classes, name_map)}]"
    )
    return "\n".join(lines)


def main():
    base = Path("datasets")
    out_lines = []

    # dota-filtered-full
    dota_filtered = base / "dota-filtered-full"
    names_filtered = load_names(Path("yaml/DOTA-filtered.yaml"))
    out_lines.append("dota-filtered-full")
    out_lines.append(report_dataset(dota_filtered, ["train", "val", "test"], names_filtered))

    # dota_stages
    stages_root = base / "dota_stages"
    stage_yamls = {
        "stage1": Path("yaml/DOTA-stage1.yaml"),
        "stage2": Path("yaml/DOTA-stage2.yaml"),
        "stage3": Path("yaml/DOTA-stage3.yaml"),
    }
    for stage in ["stage1", "stage2", "stage3"]:
        names_stage = load_names(stage_yamls[stage])
        out_lines.append("")
        out_lines.append(f"dota_stages/{stage}")
        out_lines.append(report_dataset(stages_root / stage, ["train", "val_seen", "val", "test"], names_stage))

    report = "\n".join(out_lines) + "\n"
    report_path = Path("results/dota_dataset_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(report)
    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()
