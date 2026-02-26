"""Generate LaTeX Table 5.3: Size-stratified mAP and RSD for DOTA.

Reads JSON outputs from eval_by_size.py and computes:
  - mAP@0.5 per size bucket from each method's Stage 3 JSON (all classes)
  - RSD per size bucket using Eq 4.1: average per-class relative deficit on
    OLD classes compared to joint training, within each size category

Usage:
    python scripts/make_size_table.py          # prints + saves table_5_3.tex
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ===========================================================================
# CONFIG
# ===========================================================================

# Old classes = everything introduced before the final stage
OLD_CLASSES = ["small-vehicle", "large-vehicle", "plane", "helicopter"]

# Joint training reference (single file, used for all RSD computations)
JOINT_JSON = "runs/val_by_size/upper/final.json"

METHODS = [
    ("Naive fine-tuning", {
        "stage3": "runs/val_by_size/lower/stage3.json",
    }),
    ("Joint training (upper)", {
        "stage3": "runs/val_by_size/upper/final.json",
    }),
    ("Exemplar Replay", {
        "stage3": "runs/val_by_size/replay/stage3.json",
    }),
    ("Pseudo-labeling", {
        "stage3": "runs/val_by_size/pseudo/stage3.json",
    }),
    ("Replay + KD", {
        "stage3": "runs/val_by_size/kd/stage3.json",
    }),
    ("SA-AB", {
        "stage3": "runs/val_by_size/saab/stage3.json",
    }),
    ("KD + Pseudo", {
        "stage3": "runs/val_by_size/kd-pseudo/stage3.json",
    }),
    ("SA-AB + Pseudo", {
        "stage3": "runs/val_by_size/saab-pseudo/stage3.json",
    }),
]

SIZE_BUCKETS = ["tiny", "small", "medium+"]

# Row index of joint training — RSD forced to 0, excluded from bolding
UPPER_IDX = 1

# Insert \midrule AFTER these row indices (0-based)
GROUP_BREAKS = [1, 5]

OUT_TEX = "table_5_3.tex"

# ===========================================================================
# Helpers
# ===========================================================================

def load_json(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        print(f"  WARNING: missing {p}", file=sys.stderr)
        return None
    with open(p) as f:
        return json.load(f)


def get_map50(data: dict | None, bucket: str) -> float | None:
    """Extract mAP50 for a size bucket from eval_by_size JSON."""
    if data is None:
        return None
    entry = data.get(bucket)
    if entry is None:
        return None
    return entry.get("mAP50")


def compute_rsd_eq41(
    joint_data: dict | None,
    method_data: dict | None,
    bucket: str,
    old_classes: list[str],
) -> float | None:
    """Eq 4.1 applied within a size bucket.

    RSD = (1/N_old) * sum_i [(AP_joint,i - AP_method,i) / AP_joint,i] * 100

    Uses per-class AP50 from the per_class field within the given size bucket.
    """
    if joint_data is None or method_data is None:
        return None
    j_bucket = joint_data.get(bucket)
    m_bucket = method_data.get(bucket)
    if j_bucket is None or m_bucket is None:
        return None
    j_pc = j_bucket.get("per_class", {})
    m_pc = m_bucket.get("per_class", {})

    terms = []
    for cls in old_classes:
        j_ap = j_pc.get(cls, {}).get("AP50", 0.0)
        m_ap = m_pc.get(cls, {}).get("AP50", 0.0)
        if j_ap > 0:
            terms.append((j_ap - m_ap) / j_ap)
        # j_ap == 0: joint training has no detections for this class at this
        # size — skip (no meaningful reference)
    if not terms:
        return None
    return (sum(terms) / len(terms)) * 100.0


def fmt_map(val: float | None, bold: bool = False) -> str:
    if val is None:
        return "---"
    s = f"{val:.2f}"
    return f"\\textbf{{{s}}}" if bold else s


def fmt_rsd(val: float | None, bold: bool = False) -> str:
    if val is None:
        return "---"
    s = f"{val:.1f}"
    return f"\\textbf{{{s}}}" if bold else s


# ===========================================================================
# Main
# ===========================================================================

def main():
    # ── Load joint training reference ──────────────────────────────────────
    joint_data = load_json(JOINT_JSON)
    if joint_data is None:
        print("ERROR: joint training JSON not found, cannot compute RSD",
              file=sys.stderr)

    # ── Load all method JSONs and compute metrics ──────────────────────────
    rows: list[dict] = []
    negative_warnings: list[str] = []

    for name, paths in METHODS:
        s3 = load_json(paths["stage3"])
        row = {"name": name}
        for b in SIZE_BUCKETS:
            row[f"map_{b}"] = get_map50(s3, b)
            row[f"rsd_{b}"] = compute_rsd_eq41(joint_data, s3, b, OLD_CLASSES)
        rows.append(row)

    # Force joint training RSD to 0.0 (it's the reference for itself)
    for b in SIZE_BUCKETS:
        rows[UPPER_IDX][f"rsd_{b}"] = 0.0

    # ── Check for negative RSD (unusual — CL method beats joint) ──────────
    for i, row in enumerate(rows):
        if i == UPPER_IDX:
            continue
        for b in SIZE_BUCKETS:
            rsd = row[f"rsd_{b}"]
            if rsd is not None and rsd < 0:
                msg = (f"  WARNING: negative RSD for '{row['name']}' "
                       f"at {b}: {rsd:.1f}% (CL beats joint training)")
                negative_warnings.append(msg)
                print(msg, file=sys.stderr)

    # ── Determine best per column (excluding upper) ────────────────────────
    cl_indices = [i for i in range(len(rows)) if i != UPPER_IDX]

    best_map = {}   # bucket → row index with highest mAP among CL methods
    best_rsd = {}   # bucket → row index with lowest RSD among CL methods

    for b in SIZE_BUCKETS:
        maps = [(i, rows[i][f"map_{b}"]) for i in cl_indices
                if rows[i][f"map_{b}"] is not None]
        rsds = [(i, rows[i][f"rsd_{b}"]) for i in cl_indices
                if rows[i][f"rsd_{b}"] is not None]
        best_map[b] = max(maps, key=lambda x: x[1])[0] if maps else -1
        best_rsd[b] = min(rsds, key=lambda x: x[1])[0] if rsds else -1

    # ── Plain-text table ───────────────────────────────────────────────────
    hdr = f"{'Method':<25}"
    for b in SIZE_BUCKETS:
        hdr += f"  {'mAP':>6} {'RSD':>6}"
    print(hdr)
    print("-" * len(hdr))

    for i, row in enumerate(rows):
        line = f"{row['name']:<25}"
        for b in SIZE_BUCKETS:
            m = row[f"map_{b}"]
            r = row[f"rsd_{b}"]
            ms = f"{m:.2f}" if m is not None else "---"
            rs = f"{r:.1f}" if r is not None else "---"
            line += f"  {ms:>6} {rs:>6}"
        print(line)
        if i in GROUP_BREAKS:
            print("-" * len(hdr))

    # ── Build LaTeX ────────────────────────────────────────────────────────
    tex_lines = []
    tex_lines.append(r"\begin{table}[t]")
    tex_lines.append(r"\centering")
    tex_lines.append(
        r"\caption{Size-stratified detection performance and forgetting on the "
        r"DOTA dataset (shared datalake). mAP@0.5 reports final performance "
        r"on all classes after Stage~3. RSD~(\%) measures the average relative "
        r"deficit on old classes (small-vehicle, large-vehicle, plane, "
        r"helicopter) compared to joint training per size category, following "
        r"Eq.~4.1. Size categories: "
        r"Tiny ($< 16 \times 16$\,px), Small ($16$--$32$\,px), "
        r"Medium+ ($> 32 \times 32$\,px).}"
    )
    tex_lines.append(r"\label{tab:dota-size}")
    tex_lines.append(r"\begin{tabular}{l cc cc cc}")
    tex_lines.append(r"\toprule")
    tex_lines.append(
        r"           & \multicolumn{2}{c}{Tiny}"
        r" & \multicolumn{2}{c}{Small}"
        r" & \multicolumn{2}{c}{Medium+} \\"
    )
    tex_lines.append(
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}"
    )
    tex_lines.append(
        r"Method     & mAP & RSD & mAP & RSD & mAP & RSD \\"
    )
    tex_lines.append(r"\midrule")

    for i, row in enumerate(rows):
        cells = [f"{row['name']:<25}"]
        for b in SIZE_BUCKETS:
            m_val = row[f"map_{b}"]
            r_val = row[f"rsd_{b}"]
            cells.append(fmt_map(m_val, bold=(i == best_map[b])))
            cells.append(fmt_rsd(r_val, bold=(i == best_rsd[b])))
        tex_lines.append(" & ".join(cells) + r" \\")
        if i in GROUP_BREAKS:
            tex_lines.append(r"\midrule")

    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabular}")
    tex_lines.append(r"\end{table}")

    latex = "\n".join(tex_lines)
    print(f"\n{'='*70}")
    print(latex)
    print(f"{'='*70}")

    Path(OUT_TEX).write_text(latex + "\n")
    print(f"\nSaved to {OUT_TEX}")

    # ── Summary: mean RSD per size across CL methods ───────────────────────
    # CL methods = exclude naive (idx 0) and joint (idx 1)
    cl_only = [i for i in range(len(rows)) if i not in (0, UPPER_IDX)]
    print(f"\nMean RSD across CL methods ({[rows[i]['name'] for i in cl_only]}):")
    for b in SIZE_BUCKETS:
        vals = [rows[i][f"rsd_{b}"] for i in cl_only
                if rows[i][f"rsd_{b}"] is not None]
        if vals:
            mean_rsd = sum(vals) / len(vals)
            print(f"  {b:<10}: {mean_rsd:.1f}%  (from {len(vals)} methods)")
        else:
            print(f"  {b:<10}: no data")

    if negative_warnings:
        print(f"\n{len(negative_warnings)} negative-RSD warning(s) — "
              "see stderr for details")


if __name__ == "__main__":
    main()
