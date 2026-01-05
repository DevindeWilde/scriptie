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

def ensure_stage_specific_copy(stage_dir: Path) -> Path:
    """Persist the original stage-specific val labels under val_sep/labels."""
    sep_dir = stage_dir / "val_sep" / "labels"
    if sep_dir.exists():
        return sep_dir
    val_dir = stage_dir / "val" / "labels"
    sep_dir.mkdir(parents=True, exist_ok=True)
    for txt in sorted(val_dir.glob("*.txt")):
        sep_dir.joinpath(txt.name).write_text(txt.read_text())
    return sep_dir


def main() -> None:
    cumulative: Dict[str, List[str]] | None = None
    for stage in STAGE_ORDER:
        stage_dir = DATASET_ROOT / f"dota-yolo-{stage}"
        base_dir = ensure_stage_specific_copy(stage_dir)
        stage_records = read_label_dir(base_dir)
        if cumulative is None:
            cumulative = {name: lines[:] for name, lines in stage_records.items()}
        else:
            for name, lines in stage_records.items():
                cumulative.setdefault(name, []).extend(lines)
        target_val = stage_dir / "val" / "labels"
        write_label_dir(target_val, cumulative)
        print(f"Stage {stage} cumulative val labels written ({len(cumulative)} files).")


if __name__ == "__main__":
    main()
