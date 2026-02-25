"""Generate LaTeX tables from DOTA validation results.

Reads per_class_metrics.json files from runs/val/{method}/{stage}/val/
and outputs two LaTeX tables:
  1. Summary table: mAP@0.5, mAP@[.5:.95], RSD, RPD, Ω_all
  2. Per-stage table: Old/New mAP@[0.5:0.95] at each stage

Usage:
    python scripts/generate_latex_tables.py --val-root runs/val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# ── DOTA configuration ────────────────────────────────────────────────────────

CLASS_NAMES = ["small-vehicle", "large-vehicle", "plane", "helicopter", "ship"]

# Classes introduced at each stage
STAGE_NEW = {
    1: ["small-vehicle", "large-vehicle"],
    2: ["plane", "helicopter"],
    3: ["ship"],
}

NUM_STAGES = 3

# Methods: (display_name, folder_name)
METHODS = [
    ("Naive fine-tuning",       "lower"),
    ("Joint training (upper)",  "upper"),
    ("Exemplar Replay",         "replay"),
    ("Pseudo-labeling",         "pseudo"),
    ("Replay + KD",             "kd"),
    ("SA-AB",                   "saab"),
    ("KD + Pseudo",             "pseudo_kd"),
    ("SA-AB + Pseudo",          "pseudo_saab"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def classes_up_to(stage: int) -> list[str]:
    """Return all class names seen up to and including *stage*."""
    out = []
    for t in range(1, stage + 1):
        out.extend(STAGE_NEW[t])
    return out


def old_classes_at(stage: int) -> list[str]:
    """Return classes introduced before *stage*."""
    return classes_up_to(stage - 1) if stage > 1 else []


def load_per_class(json_path: Path) -> dict[str, dict] | None:
    """Load per_class_metrics.json → {class_name: {map50, map, ...}} or None."""
    if not json_path.exists():
        return None
    with open(json_path) as f:
        data = json.load(f)
    if not data:
        return None
    return data[0].get("metrics", {})


def avg_metric(per_class: dict, class_list: list[str], key: str = "map") -> float | None:
    """Average *key* over *class_list*.  Returns None if no data."""
    vals = []
    for cls in class_list:
        entry = per_class.get(cls)
        if entry is not None:
            vals.append(entry[key])
    return sum(vals) / len(vals) if vals else None


def compute_rsd(inc_pc: dict, joint_pc: dict, old_classes: list[str]) -> float | None:
    """Rate of Stability Deficit (%) — degradation on old classes vs joint."""
    if not old_classes:
        return None
    terms = []
    for cls in old_classes:
        j = joint_pc.get(cls, {}).get("map", 0.0)
        i = inc_pc.get(cls, {}).get("map", 0.0)
        if j > 0:
            terms.append((j - i) / j)
    return (sum(terms) / len(terms) * 100) if terms else None


def compute_rpd(inc_pc: dict, joint_pc: dict, new_classes: list[str]) -> float | None:
    """Rate of Plasticity Deficit (%) — degradation on new classes vs joint."""
    if not new_classes:
        return None
    terms = []
    for cls in new_classes:
        j = joint_pc.get(cls, {}).get("map", 0.0)
        i = inc_pc.get(cls, {}).get("map", 0.0)
        if j > 0:
            terms.append((j - i) / j)
    return (sum(terms) / len(terms) * 100) if terms else None


def compute_omega(val_root: Path, folder: str, joint_pc: dict) -> float | None:
    """Ω_all = (1/T) Σ_t (α_all,t / α_offline,t)."""
    ratios = []
    for t in range(1, NUM_STAGES + 1):
        seen = classes_up_to(t)
        # CL model's mAP on all classes seen up to stage t
        stage_dir = "final" if folder == "upper" else f"stage{t}"
        pc = load_per_class(val_root / folder / stage_dir / "val" / "per_class_metrics.json")
        if pc is None:
            return None
        alpha_cl = avg_metric(pc, seen)
        alpha_joint = avg_metric(joint_pc, seen)
        if alpha_cl is None or alpha_joint is None or alpha_joint == 0:
            return None
        ratios.append(alpha_cl / alpha_joint)
    return sum(ratios) / len(ratios) * NUM_STAGES  # scale: perfect = T


def fmt(val: float | None, decimals: int = 2) -> str:
    """Format a float for LaTeX, or '--' if None."""
    if val is None:
        return "--"
    return f"{val:.{decimals}f}"


# ── Table generators ──────────────────────────────────────────────────────────

def generate_summary_table(val_root: Path, joint_pc: dict) -> str:
    """Table 1: final-stage summary with mAP, RSD, RPD, Ω_all."""
    rows = []
    final_old = old_classes_at(NUM_STAGES)
    final_new = STAGE_NEW[NUM_STAGES]

    for display, folder in METHODS:
        if folder == "upper":
            # Upper bound: use joint per-class directly
            map50 = avg_metric(joint_pc, CLASS_NAMES, "map50")
            map5095 = avg_metric(joint_pc, CLASS_NAMES, "map")
            rsd = 0.0
            rpd = 0.0
            omega = float(NUM_STAGES)
        else:
            pc = load_per_class(
                val_root / folder / f"stage{NUM_STAGES}" / "val" / "per_class_metrics.json"
            )
            if pc is None:
                rows.append(f"  {display:<26s} & -- & -- & -- & -- & -- \\\\")
                continue
            map50 = avg_metric(pc, CLASS_NAMES, "map50")
            map5095 = avg_metric(pc, CLASS_NAMES, "map")
            rsd = compute_rsd(pc, joint_pc, final_old)
            rpd = compute_rpd(pc, joint_pc, final_new)
            omega = compute_omega(val_root, folder, joint_pc)

        rows.append(
            f"  {display:<26s} & {fmt(map50)} & {fmt(map5095)} "
            f"& {fmt(rsd)} & {fmt(rpd)} & {fmt(omega)} \\\\"
        )

    # Insert midrule after joint training
    if len(rows) >= 2:
        rows.insert(2, "  \\midrule")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{l c c c c c}",
        r"\toprule",
        r"\textbf{Method}",
        r"  & \textbf{mAP@0.5} $\uparrow$",
        r"  & \textbf{mAP@[.5:.95]} $\uparrow$",
        r"  & \textbf{RSD (\%)} $\downarrow$",
        r"  & \textbf{RPD (\%)} $\downarrow$",
        r"  & $\boldsymbol{\Omega_{\textbf{all}}}$ $\uparrow$ \\",
        r"\midrule",
    ]
    lines.extend(rows)
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Final detection accuracy and continual learning performance on the",
        r"DOTA dataset after completing all incremental stages.}",
        r"\label{tab:dota-final}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def generate_perstage_table(val_root: Path, joint_pc: dict) -> str:
    """Table 2: per-stage Old/New mAP@[0.5:0.95]."""
    rows = []

    for display, folder in METHODS:
        cells = []
        for t in range(1, NUM_STAGES + 1):
            old_cls = old_classes_at(t)
            new_cls = STAGE_NEW[t]

            if folder == "upper":
                pc = joint_pc
            else:
                stage_dir = f"stage{t}"
                pc = load_per_class(
                    val_root / folder / stage_dir / "val" / "per_class_metrics.json"
                )

            if pc is None:
                cells.append("-- & --")
                continue

            old_val = avg_metric(pc, old_cls) if old_cls else None
            new_val = avg_metric(pc, new_cls)
            cells.append(f"{fmt(old_val)} & {fmt(new_val)}")

        row = f"  {display:<26s} & " + " & ".join(cells) + r" \\"
        rows.append(row)

    # Insert midrule after joint training
    if len(rows) >= 2:
        rows.insert(2, "  \\midrule")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{l cc cc cc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{\textbf{Stage 1}}",
        r"& \multicolumn{2}{c}{\textbf{Stage 2}}",
        r"& \multicolumn{2}{c}{\textbf{Stage 3}} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"\textbf{Method} & Old & New & Old & New & Old & New \\",
        r"\midrule",
    ]
    lines.extend(rows)
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Per-stage detection performance (mAP@[0.5:0.95]) on the DOTA",
        r"dataset, separating performance on previously learned (\textit{old}) and newly",
        r"introduced (\textit{new}) classes at each incremental stage.}",
        r"\label{tab:dota-perstage}",
        r"\end{table}",
    ])
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from DOTA val results")
    parser.add_argument("--val-root", type=str, default="runs/val",
                        help="Root directory containing method/stage validation folders")
    args = parser.parse_args()

    val_root = Path(args.val_root)
    joint_pc = load_per_class(val_root / "upper" / "final" / "val" / "per_class_metrics.json")
    if joint_pc is None:
        print("ERROR: Could not load upper bound (joint training) results from")
        print(f"  {val_root / 'upper' / 'final' / 'val' / 'per_class_metrics.json'}")
        return

    print("% ═══════════════════════════════════════════════════════════════")
    print("% Table 1: Summary (final stage)")
    print("% ═══════════════════════════════════════════════════════════════")
    print()
    print(generate_summary_table(val_root, joint_pc))
    print()
    print()
    print("% ═══════════════════════════════════════════════════════════════")
    print("% Table 2: Per-stage Old / New")
    print("% ═══════════════════════════════════════════════════════════════")
    print()
    print(generate_perstage_table(val_root, joint_pc))


if __name__ == "__main__":
    main()
