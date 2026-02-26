"""Generate LaTeX tables for the Military-UAV dataset.

Reads per_class_metrics.json files and eval_by_size JSONs, then outputs
four tables (no absolute mAP — dataset is restricted):
  5.4 — Overall relative metrics (RSD, RPD, Omega_all)
  5.5 — Per-stage Old/New mAP normalized to joint training
  5.6 — Per-size RSD (Eq 4.1 applied within each size bucket)
  5.7 — Cross-dataset comparison (DOTA vs Military-UAV)

Usage:
    python scripts/generate_military_tables.py
    python scripts/generate_military_tables.py \
        --val-root runs/eo_val_zip \
        --size-root runs/eo_val_by_size
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# Military-UAV configuration
# ═══════════════════════════════════════════════════════════════════════════════

CLASS_NAMES = [
    "person", "howitzer", "boxer",      # Stage 1
    "fennek", "truck", "other",         # Stage 2
    "buffel", "cv90", "leopard",        # Stage 3
]

STAGE_NEW = {
    1: ["person", "howitzer", "boxer"],
    2: ["fennek", "truck", "other"],
    3: ["buffel", "cv90", "leopard"],
}

NUM_STAGES = 3

# (display_name, folder_name) — folder names match eo_val.sh output
METHODS = [
    ("Naive fine-tuning",       "finetune"),
    ("Joint training (upper)",  "upper"),
    ("Exemplar Replay",         "replay"),
    ("Pseudo-labeling",         "pseudo"),
    ("Replay + KD",             "kd"),
    ("SA-AB",                   "saab"),
    ("KD + Pseudo",             "kd_pseudo"),
    ("SA-AB + Pseudo",          "saab_pseudo"),
]

UPPER_IDX = 1                # row index of joint training
GROUP_BREAKS = [1, 5]        # insert \midrule AFTER these row indices

SIZE_BINS = ["tiny", "small", "medium+"]

# Hardcoded DOTA results for cross-dataset comparison (from Table 5.1)
DOTA_RESULTS = {
    "Naive fine-tuning":      {"RSD": 100.00, "omega": 1.39},
    "Joint training (upper)": {"RSD": 0.00,   "omega": 3.00},
    "Exemplar Replay":        {"RSD": 65.37,  "omega": 1.94},
    "Pseudo-labeling":        {"RSD": 27.58,  "omega": 2.77},
    "Replay + KD":            {"RSD": 60.03,  "omega": 2.00},
    "SA-AB":                  {"RSD": 68.30,  "omega": 1.91},
    "KD + Pseudo":            {"RSD": 28.60,  "omega": 2.74},
    "SA-AB + Pseudo":         {"RSD": 28.97,  "omega": 2.76},
}

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def classes_up_to(stage: int) -> list[str]:
    """All class names seen up to and including *stage*."""
    out = []
    for t in range(1, stage + 1):
        out.extend(STAGE_NEW[t])
    return out


def old_classes_at(stage: int) -> list[str]:
    """Classes introduced before *stage*."""
    return classes_up_to(stage - 1) if stage > 1 else []


def load_per_class(json_path: Path) -> dict[str, dict] | None:
    """Load per_class_metrics.json → {class_name: {map50, map, ...}} or None.

    If the exact path doesn't exist, searches sibling val*/ directories
    (handles val2/, val3/ created by repeated val.py runs).
    """
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        if data:
            return data[0].get("metrics", {})

    # Fallback: search val*/ directories in the parent stage folder
    stage_dir = json_path.parent.parent
    if stage_dir.is_dir():
        for vdir in sorted(stage_dir.glob("val*"), reverse=True):
            candidate = vdir / "per_class_metrics.json"
            if candidate.exists():
                with open(candidate) as f:
                    data = json.load(f)
                if data:
                    return data[0].get("metrics", {})
    return None


def load_size_json(path: Path) -> dict | None:
    """Load an eval_by_size JSON file."""
    if not path.exists():
        print(f"  WARNING: missing {path}")
        return None
    with open(path) as f:
        return json.load(f)


def avg_metric(per_class: dict, class_list: list[str],
               key: str = "map") -> float | None:
    """Average *key* over *class_list*.  Returns None if no data."""
    vals = []
    for cls in class_list:
        entry = per_class.get(cls)
        if entry is not None:
            vals.append(entry[key])
    return sum(vals) / len(vals) if vals else None


def compute_rsd(inc_pc: dict, joint_pc: dict,
                old_classes: list[str]) -> float | None:
    """Rate of Stability Deficit (%) — Eq 4.1."""
    if not old_classes:
        return None
    terms = []
    for cls in old_classes:
        j = joint_pc.get(cls, {}).get("map", 0.0)
        i = inc_pc.get(cls, {}).get("map", 0.0)
        if j > 0:
            terms.append((j - i) / j)
    return (sum(terms) / len(terms) * 100) if terms else None


def compute_rpd(inc_pc: dict, joint_pc: dict,
                new_classes: list[str]) -> float | None:
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
    """Omega_all = (1/T) * sum_t (alpha_cl,t / alpha_joint,t), scaled so perfect = T."""
    ratios = []
    for t in range(1, NUM_STAGES + 1):
        seen = classes_up_to(t)
        stage_dir = "final" if folder == "upper" else f"stage{t}"
        pc = load_per_class(
            val_root / folder / stage_dir / "val" / "per_class_metrics.json"
        )
        if pc is None:
            return None
        alpha_cl = avg_metric(pc, seen)
        alpha_joint = avg_metric(joint_pc, seen)
        if alpha_cl is None or alpha_joint is None or alpha_joint == 0:
            return None
        ratios.append(alpha_cl / alpha_joint)
    return sum(ratios) / len(ratios) * NUM_STAGES


def compute_size_rsd_eq41(
    joint_data: dict | None,
    method_data: dict | None,
    bucket: str,
    old_classes: list[str],
) -> float | None:
    """Eq 4.1 applied within a size bucket using per-class AP50."""
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
    if not terms:
        return None
    return (sum(terms) / len(terms)) * 100.0


def fmt(val: float | None, decimals: int = 2, bold: bool = False) -> str:
    """Format a float for LaTeX, or '--' if None."""
    if val is None:
        return "--"
    s = f"{val:.{decimals}f}"
    return f"\\textbf{{{s}}}" if bold else s


def best_idx(data: list, col: int, minimize: bool,
             exclude: set[int] | None = None) -> int:
    """Find row index with best value in a column, ignoring None and excluded rows."""
    if exclude is None:
        exclude = set()
    candidates = []
    for i, row in enumerate(data):
        v = row[col]
        if i not in exclude and v is not None:
            candidates.append((i, v))
    if not candidates:
        return -1
    if minimize:
        return min(candidates, key=lambda x: x[1])[0]
    return max(candidates, key=lambda x: x[1])[0]


# ═══════════════════════════════════════════════════════════════════════════════
# Table 5.4 — Overall relative metrics
# ═══════════════════════════════════════════════════════════════════════════════

def generate_overall_table(method_data: list[dict]) -> str:
    """Table 5.4: RSD (%), RPD (%), Omega_all — no absolute mAP."""
    vals = [(d["rsd"], d["rpd"], d["omega"]) for d in method_data]
    excl = {UPPER_IDX}

    b_rsd   = best_idx(vals, 0, minimize=True,  exclude=excl)
    b_rpd   = best_idx(vals, 1, minimize=True,  exclude=excl)
    b_omega = best_idx(vals, 2, minimize=False, exclude=excl)

    rows = []
    for i, d in enumerate(method_data):
        cells = [
            f"  {d['name']:<26s}",
            fmt(d["rsd"],   1, bold=(i == b_rsd)),
            fmt(d["rpd"],   1, bold=(i == b_rpd)),
            fmt(d["omega"], 2, bold=(i == b_omega)),
        ]
        rows.append(" & ".join(cells) + r" \\")
        if i in GROUP_BREAKS:
            rows.append(r"  \midrule")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Continual learning performance on the Military-UAV dataset",
        r"(individual datastreams). Only relative metrics are shown due to dataset",
        r"restrictions. RSD and RPD follow Eq.~4.1; $\Omega_{\text{all}}$ aggregates",
        r"all stages (perfect $= " + str(NUM_STAGES) + r"$).}",
        r"\label{tab:military-overall}",
        r"\begin{tabular}{l ccc}",
        r"\toprule",
        r"Method & RSD (\%) $\downarrow$ & RPD (\%) $\downarrow$"
        r" & $\Omega_{\text{all}}$ $\uparrow$ \\",
        r"\midrule",
    ]
    lines.extend(rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Table 5.5 — Normalized per-stage Old/New
# ═══════════════════════════════════════════════════════════════════════════════

def generate_normalized_perstage_table(val_root: Path, joint_pc: dict) -> str:
    """Table 5.5: Old/New mAP normalized to joint training at each stage."""
    n_cols = NUM_STAGES * 2
    grid: list[list[float | None]] = []
    names: list[str] = []

    for display, folder in METHODS:
        names.append(display)
        row_vals: list[float | None] = []

        for t in range(1, NUM_STAGES + 1):
            old_cls = old_classes_at(t)
            new_cls = STAGE_NEW[t]

            if folder == "upper":
                pc = joint_pc
            else:
                pc = load_per_class(
                    val_root / folder / f"stage{t}" / "val"
                    / "per_class_metrics.json"
                )

            # Old classes
            if not old_cls:
                row_vals.append(None)
            elif pc is None:
                row_vals.append(None)
            else:
                m_val = avg_metric(pc, old_cls)
                j_val = avg_metric(joint_pc, old_cls)
                if m_val is not None and j_val is not None and j_val > 0:
                    row_vals.append(m_val / j_val)
                else:
                    row_vals.append(None)

            # New classes
            if pc is None:
                row_vals.append(None)
            else:
                m_val = avg_metric(pc, new_cls)
                j_val = avg_metric(joint_pc, new_cls)
                if m_val is not None and j_val is not None and j_val > 0:
                    row_vals.append(m_val / j_val)
                else:
                    row_vals.append(None)

        grid.append(row_vals)

    excl = {UPPER_IDX}
    best_col = [best_idx(grid, c, minimize=False, exclude=excl)
                for c in range(n_cols)]

    rows = []
    for i in range(len(METHODS)):
        cells = [f"  {names[i]:<26s}"]
        for c in range(n_cols):
            cells.append(fmt(grid[i][c], 2, bold=(i == best_col[c])))
        rows.append(" & ".join(cells) + r" \\")
        if i in GROUP_BREAKS:
            rows.append(r"  \midrule")

    col_spec = "l " + "cc " * NUM_STAGES
    stage_hdrs = " ".join(
        rf"& \multicolumn{{2}}{{c}}{{Stage {t}}}" for t in range(1, NUM_STAGES + 1)
    )
    cmidrules = " ".join(
        rf"\cmidrule(lr){{{2 + (t-1)*2}-{3 + (t-1)*2}}}" for t in range(1, NUM_STAGES + 1)
    )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Per-stage normalized detection performance on the Military-UAV",
        r"dataset. Values represent the ratio of each method's mAP@[0.5:0.95] to that",
        r"of joint training at the corresponding stage. A value of 1.00 indicates",
        r"parity with the upper bound.}",
        r"\label{tab:military-oldnew}",
        rf"\begin{{tabular}}{{{col_spec.strip()}}}",
        r"\toprule",
        stage_hdrs + r" \\",
        cmidrules,
        r"Method" + r" & Old & New" * NUM_STAGES + r" \\",
        r"\midrule",
    ]
    lines.extend(rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Table 5.6 — Per-size RSD (Eq 4.1 within each size bucket)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_size_rsd_table(size_root: Path) -> str:
    """Table 5.6: RSD per size bucket, using per-class AP50 deficit vs joint."""
    old_cls = old_classes_at(NUM_STAGES)

    # Load joint training size JSON as reference
    joint_size = load_size_json(size_root / "upper" / "stage3.json")

    # Compute RSD per method per size bucket
    vals: list[list[float | None]] = []  # [method_idx][size_idx]
    for display, folder in METHODS:
        if folder == "upper":
            row = [0.0] * len(SIZE_BINS)
        else:
            m_size = load_size_json(size_root / folder / "stage3.json")
            row = [
                compute_size_rsd_eq41(joint_size, m_size, b, old_cls)
                for b in SIZE_BINS
            ]
        vals.append(row)

    excl = {UPPER_IDX}
    best_per_size = [best_idx(vals, c, minimize=True, exclude=excl)
                     for c in range(len(SIZE_BINS))]

    rows = []
    for i, (display, _) in enumerate(METHODS):
        cells = [f"  {display:<26s}"]
        for c in range(len(SIZE_BINS)):
            cells.append(fmt(vals[i][c], 1, bold=(i == best_per_size[c])))
        rows.append(" & ".join(cells) + r" \\")
        if i in GROUP_BREAKS:
            rows.append(r"  \midrule")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Size-stratified forgetting (RSD) on the Military-UAV dataset,",
        r"following Eq.~4.1 applied per size category. RSD is computed over old",
        r"classes only, relative to joint training. Size categories:",
        r"Tiny ($< 16 \times 16$\,px), Small ($16$--$32$\,px),",
        r"Medium+ ($> 32 \times 32$\,px).}",
        r"\label{tab:military-size}",
        r"\begin{tabular}{l ccc}",
        r"\toprule",
        r"Method & Tiny (\%) $\downarrow$ & Small (\%) $\downarrow$"
        r" & Medium+ (\%) $\downarrow$ \\",
        r"\midrule",
    ]
    lines.extend(rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Table 5.7 — Cross-dataset comparison
# ═══════════════════════════════════════════════════════════════════════════════

def generate_crossdataset_table(method_data: list[dict]) -> str:
    """Table 5.7: DOTA vs Military-UAV using RSD and Omega_all."""
    vals: list[tuple[float | None, ...]] = []
    for d in method_data:
        dota = DOTA_RESULTS.get(d["name"], {})
        vals.append((
            dota.get("RSD"),
            dota.get("omega"),
            d["rsd"],
            d["omega"],
        ))

    excl = {UPPER_IDX}
    b = [
        best_idx(vals, 0, minimize=True,  exclude=excl),
        best_idx(vals, 1, minimize=False, exclude=excl),
        best_idx(vals, 2, minimize=True,  exclude=excl),
        best_idx(vals, 3, minimize=False, exclude=excl),
    ]

    rows = []
    for i, d in enumerate(method_data):
        v = vals[i]
        cells = [
            f"  {d['name']:<26s}",
            fmt(v[0], 1, bold=(i == b[0])),
            fmt(v[1], 2, bold=(i == b[1])),
            fmt(v[2], 1, bold=(i == b[2])),
            fmt(v[3], 2, bold=(i == b[3])),
        ]
        rows.append(" & ".join(cells) + r" \\")
        if i in GROUP_BREAKS:
            rows.append(r"  \midrule")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Cross-dataset comparison of continual learning methods. DOTA",
        r"uses a shared datalake where old-class objects appear unannotated in new",
        r"stages. Military-UAV uses individual datastreams where each stage contains",
        r"distinct images.}",
        r"\label{tab:cross-dataset}",
        r"\begin{tabular}{l cc cc}",
        r"\toprule",
        r"            & \multicolumn{2}{c}{DOTA (shared)}"
        r" & \multicolumn{2}{c}{Military-UAV (individual)} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"Method      & RSD $\downarrow$"
        r" & $\Omega_{\text{all}}$ $\uparrow$"
        r" & RSD $\downarrow$"
        r" & $\Omega_{\text{all}}$ $\uparrow$ \\",
        r"\midrule",
    ]
    lines.extend(rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate Military-UAV LaTeX tables"
    )
    parser.add_argument("--val-root", type=str, default="runs/eo_val_zip",
                        help="Root dir with method/stage validation folders")
    parser.add_argument("--size-root", type=str, default="runs/eo_val_by_size",
                        help="Root dir with eval_by_size JSON outputs")
    args = parser.parse_args()

    val_root = Path(args.val_root)
    size_root = Path(args.size_root)

    # Load joint training reference
    joint_path = val_root / "upper" / "final" / "val" / "per_class_metrics.json"
    joint_pc = load_per_class(joint_path)
    if joint_pc is None:
        print(f"ERROR: Could not load joint training results from\n  {joint_path}")
        return

    # ── Compute shared method data (Tables 5.4 and 5.7) ───────────────────
    final_old = old_classes_at(NUM_STAGES)
    final_new = STAGE_NEW[NUM_STAGES]

    method_data: list[dict] = []
    for display, folder in METHODS:
        d: dict = {"name": display, "folder": folder}
        if folder == "upper":
            d["rsd"] = 0.0
            d["rpd"] = 0.0
            d["omega"] = float(NUM_STAGES)
        else:
            pc = load_per_class(
                val_root / folder / f"stage{NUM_STAGES}" / "val"
                / "per_class_metrics.json"
            )
            if pc is None:
                d["rsd"] = d["rpd"] = d["omega"] = None
            else:
                d["rsd"] = compute_rsd(pc, joint_pc, final_old)
                d["rpd"] = compute_rpd(pc, joint_pc, final_new)
                d["omega"] = compute_omega(val_root, folder, joint_pc)
        method_data.append(d)

    # ── Generate all tables ────────────────────────────────────────────────
    tables = [
        ("Table 5.4: Overall relative metrics",   "table_5_4.tex",
         generate_overall_table(method_data)),
        ("Table 5.5: Normalized per-stage",        "table_5_5.tex",
         generate_normalized_perstage_table(val_root, joint_pc)),
        ("Table 5.6: Per-size RSD",                "table_5_6.tex",
         generate_size_rsd_table(size_root)),
        ("Table 5.7: Cross-dataset comparison",    "table_5_7.tex",
         generate_crossdataset_table(method_data)),
    ]

    for title, filename, content in tables:
        Path(filename).write_text(content + "\n")
        print(f"\n% {'=' * 68}")
        print(f"% {title}")
        print(f"% {'=' * 68}\n")
        print(content)

    print(f"\nSaved: {', '.join(t[1] for t in tables)}")

    # ── Plain-text summary ─────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("Summary of computed metrics:")
    print(f"{'─' * 60}")
    print(f"  {'Method':<26s} {'RSD':>6} {'RPD':>6} {'Omega':>6}")
    for d in method_data:
        r = f"{d['rsd']:.1f}" if d["rsd"] is not None else "--"
        p = f"{d['rpd']:.1f}" if d["rpd"] is not None else "--"
        o = f"{d['omega']:.2f}" if d["omega"] is not None else "--"
        print(f"  {d['name']:<26s} {r:>6} {p:>6} {o:>6}")


if __name__ == "__main__":
    main()
