#!/usr/bin/env python3
"""Generate cumulative validation label sets for each DOTA stage."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List


STAGE_ORDER = ("stage1", "stage2", "stage3")
DATASET_ROOT = Path("datasets")


def read_label_dir(label_dir: Path) -> Dict[str, List[str]]:
    records: Dict[str, List[str]] = {}
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing labels directory: {label_dir}")
    for txt_path in sorted(label_dir.glob("*.txt")):
        lines = [ln.strip() for ln in txt_path.read_text().splitlines() if ln.strip()]
        records[txt_path.name] = lines
    return records


def write_label_dir(target_dir: Path, labels: Dict[str, List[str]]) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for name, lines in labels.items():
        out_path = target_dir / name
        out_path.write_text("\n".join(lines))

def main() -> None:
    cumulative: Dict[str, List[str]] = {}
    for stage in STAGE_ORDER:
        stage_dir = DATASET_ROOT / f"dota-yolo-{stage}"
        val_labels = stage_dir / "val" / "labels"
        stage_records = read_label_dir(val_labels)
        if not cumulative:
            cumulative = {name: lines[:] for name, lines in stage_records.items()}
        else:
            for name, lines in stage_records.items():
                cumulative.setdefault(name, []).extend(lines)
        val_all_dir = stage_dir / "val_all" / "labels"
        write_label_dir(val_all_dir, cumulative)
        print(f"Wrote {len(cumulative)} label files to {val_all_dir}")


if __name__ == "__main__":
    main()
