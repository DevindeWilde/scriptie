#!/usr/bin/env python3
"""Generate LaTeX tables for Military-UAV continual learning experiments.

Outputs four tables:
  5.4  Overall relative metrics (RSD, RPD, Ω_all)
  5.5  Per-stage normalized Old/New mAP (ratio to joint training)
  5.6  Per-size RSD (tiny / small / medium+)
  5.7  Cross-dataset comparison (DOTA vs Military-UAV)

No absolute mAP values are shown — dataset is restricted.

Usage:
    python scripts/make_military_tables.py
    python scripts/make_military_tables.py --outdir results/military_tables
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — fill in actual paths and values
# ═══════════════════════════════════════════════════════════════════════════════

METHODS = [
    "Naive fine-tuning",
    "Joint training (upper)",
    "Exemplar Replay",
    "Pseudo-labeling",
    "Replay + KD",
    "SA-AB",
    "KD + Pseudo",
    "SA-AB + Pseudo",
]

# Method groups for \midrule placement (indices into METHODS)
GROUP_BREAKS = [2, 6]  # midrule after index 1 (joint) and index 5 (SA-AB)

CONFIG = {
    # --- Per-size JSON files (from eval_by_size.py) ---
    # Each JSON: {"all": {"mAP50": ...}, "tiny": {"mAP50": ...}, ...}
    # Paths match eo_eval_by_size.sh output: $RUNS/eo_val_by_size/<method>/stage{1,3}.json
    "size_jsons": {
        "Naive fine-tuning": {
            "stage1": "runs/eo_val_by_size/finetune/stage1.json",
            "stage3": "runs/eo_val_by_size/finetune/stage3.json",
        },
        "Joint training (upper)": {
            "stage1": "runs/eo_val_by_size/upper/stage1.json",
            "stage3": "runs/eo_val_by_size/upper/stage3.json",
        },
        "Exemplar Replay": {
            "stage1": "runs/eo_val_by_size/replay/stage1.json",
            "stage3": "runs/eo_val_by_size/replay/stage3.json",
        },
        "Pseudo-labeling": {
            "stage1": "runs/eo_val_by_size/pseudo/stage1.json",
            "stage3": "runs/eo_val_by_size/pseudo/stage3.json",
        },
        "Replay + KD": {
            "stage1": "runs/eo_val_by_size/kd/stage1.json",
            "stage3": "runs/eo_val_by_size/kd/stage3.json",
        },
        "SA-AB": {
            "stage1": "runs/eo_val_by_size/saab/stage1.json",
            "stage3": "runs/eo_val_by_size/saab/stage3.json",
        },
        "KD + Pseudo": {
            "stage1": "runs/eo_val_by_size/kd_pseudo/stage1.json",
            "stage3": "runs/eo_val_by_size/kd_pseudo/stage3.json",
        },
        "SA-AB + Pseudo": {
            "stage1": "runs/eo_val_by_size/saab_pseudo/stage1.json",
            "stage3": "runs/eo_val_by_size/saab_pseudo/stage3.json",
        },
    },

    # --- Per-stage Old/New mAP values (manual input from eval logs) ---
    # Stage 1 has no "old" classes → old = None
    "per_stage": {
        "Naive fine-tuning": {
            "stage1": {"old": None, "new": 0.00},
            "stage2": {"old": 0.00, "new": 0.00},
            "stage3": {"old": 0.00, "new": 0.00},
        },
        "Joint training (upper)": {
            "stage1": {"old": None, "new": 0.00},
            "stage2": {"old": 0.00, "new": 0.00},
            "stage3": {"old": 0.00, "new": 0.00},
        },
        "Exemplar Replay": {
            "stage1": {"old": None, "new": 0.00},
            "stage2": {"old": 0.00, "new": 0.00},
            "stage3": {"old": 0.00, "new": 0.00},
        },
        "Pseudo-labeling": {
            "stage1": {"old": None, "new": 0.00},
            "stage2": {"old": 0.00, "new": 0.00},
            "stage3": {"old": 0.00, "new": 0.00},
        },
        "Replay + KD": {
            "stage1": {"old": None, "new": 0.00},
            "stage2": {"old": 0.00, "new": 0.00},
            "stage3": {"old": 0.00, "new": 0.00},
        },
        "SA-AB": {
            "stage1": {"old": None, "new": 0.00},
            "stage2": {"old": 0.00, "new": 0.00},
            "stage3": {"old": 0.00, "new": 0.00},
        },
        "KD + Pseudo": {
            "stage1": {"old": None, "new": 0.00},
            "stage2": {"old": 0.00, "new": 0.00},
            "stage3": {"old": 0.00, "new": 0.00},
        },
        "SA-AB + Pseudo": {
            "stage1": {"old": None, "new": 0.00},
            "stage2": {"old": 0.00, "new": 0.00},
            "stage3": {"old": 0.00, "new": 0.00},
        },
    },

    # --- Overall relative metrics (manual input) ---
    "overall": {
        "Naive fine-tuning":      {"RSD": 0.0, "RPD": 0.0, "omega": 0.0},
        "Joint training (upper)": {"RSD": 0.0, "RPD": 0.0, "omega": 3.0},
        "Exemplar Replay":        {"RSD": 0.0, "RPD": 0.0, "omega": 0.0},
        "Pseudo-labeling":        {"RSD": 0.0, "RPD": 0.0, "omega": 0.0},
        "Replay + KD":            {"RSD": 0.0, "RPD": 0.0, "omega": 0.0},
        "SA-AB":                  {"RSD": 0.0, "RPD": 0.0, "omega": 0.0},
        "KD + Pseudo":            {"RSD": 0.0, "RPD": 0.0, "omega": 0.0},
        "SA-AB + Pseudo":         {"RSD": 0.0, "RPD": 0.0, "omega": 0.0},
    },

    # --- DOTA results for cross-dataset comparison (from Table 5.1) ---
    "dota_overall": {
        "Naive fine-tuning":      {"RSD": 100.00, "omega": 1.39},
        "Joint training (upper)": {"RSD": 0.00,   "omega": 3.00},
        "Exemplar Replay":        {"RSD": 65.37,  "omega": 1.94},
        "Pseudo-labeling":        {"RSD": 27.58,  "omega": 2.77},
        "Replay + KD":            {"RSD": 60.03,  "omega": 2.00},
        "SA-AB":                  {"RSD": 68.30,  "omega": 1.91},
        "KD + Pseudo":            {"RSD": 28.60,  "omega": 2.74},
        "SA-AB + Pseudo":         {"RSD": 28.97,  "omega": 2.76},
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

JOINT = "Joint training (upper)"
SIZE_BINS = ["tiny", "small", "medium+"]


def _tex(s: str) -> str:
    """Escape underscores for LaTeX."""
    return s.replace("_", r"\_").replace("&", r"\&")


def _short_name(method: str) -> str:
    """Shorter display name for tables (drop '(upper)' tag)."""
    return method.replace(" (upper)", "")


def _load_json(path: str) -> dict | None:
    """Load a JSON file, return None with a warning on failure."""
    p = Path(path)
    if not p.exists():
        print(f"  WARNING: missing {p}")
        return None
    with open(p) as f:
        return json.load(f)


def _bold(val: str) -> str:
    return rf"\textbf{{{val}}}"


def _find_best(values: dict[str, float | None], minimize: bool) -> str | None:
    """Find method name with best value (excluding joint training).
    Returns None if no valid candidates."""
    candidates = {
        m: v for m, v in values.items()
        if v is not None and m != JOINT
    }
    if not candidates:
        return None
    if minimize:
        return min(candidates, key=candidates.get)
    return max(candidates, key=candidates.get)


def _fmt_1(v: float | None, best: bool = False) -> str:
    if v is None:
        return "--"
    s = f"{v:.1f}"
    return _bold(s) if best else s


def _fmt_2(v: float | None, best: bool = False) -> str:
    if v is None:
        return "--"
    s = f"{v:.2f}"
    return _bold(s) if best else s


def _write_and_print(path: Path, content: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")
    print(f"\n{'=' * 72}")
    print(f"  {title}  ->  {path}")
    print(f"{'=' * 72}")
    print(content)


# ═══════════════════════════════════════════════════════════════════════════════
# Table 5.4 — Overall Relative Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def make_table_5_4(outdir: Path) -> str:
    data = CONFIG["overall"]

    # Find bests (excluding joint)
    rsd_vals = {m: data[m]["RSD"] for m in METHODS if m in data}
    rpd_vals = {m: data[m]["RPD"] for m in METHODS if m in data}
    omg_vals = {m: data[m]["omega"] for m in METHODS if m in data}
    best_rsd = _find_best(rsd_vals, minimize=True)
    best_rpd = _find_best(rpd_vals, minimize=True)
    best_omg = _find_best(omg_vals, minimize=False)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Continual learning performance on the Military-UAV dataset (individual datastreams).",
        r"Only relative metrics are shown due to dataset restrictions.}",
        r"\label{tab:military-overall}",
        r"\begin{tabular}{l ccc}",
        r"\toprule",
        r"Method & RSD (\%) $\downarrow$ & RPD (\%) $\downarrow$"
        r" & $\Omega_{\text{all}}$ $\uparrow$ \\",
        r"\midrule",
    ]

    for i, method in enumerate(METHODS):
        if i in GROUP_BREAKS:
            lines.append(r"\midrule")
        d = data.get(method, {})
        rsd = d.get("RSD")
        rpd = d.get("RPD")
        omg = d.get("omega")

        rsd_s = _fmt_1(rsd, best=(method == best_rsd))
        rpd_s = _fmt_1(rpd, best=(method == best_rpd))
        omg_s = _fmt_2(omg, best=(method == best_omg))

        lines.append(f"{_short_name(method)} & {rsd_s} & {rpd_s} & {omg_s} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    content = "\n".join(lines)

    # Plain-text summary
    print("\n  Table 5.4 — Military-UAV Overall Relative Metrics")
    print(f"  {'Method':<25} {'RSD%':>7} {'RPD%':>7} {'Omega':>7}")
    print("  " + "-" * 50)
    for method in METHODS:
        d = data.get(method, {})
        r = d.get("RSD", 0)
        p = d.get("RPD", 0)
        o = d.get("omega", 0)
        tag = " *" if method in (best_rsd, best_rpd, best_omg) else ""
        print(f"  {_short_name(method):<25} {r:>7.1f} {p:>7.1f} {o:>7.2f}{tag}")

    _write_and_print(outdir / "table_5_4.tex", content, "Table 5.4")
    return content


# ═══════════════════════════════════════════════════════════════════════════════
# Table 5.5 — Per-Stage Normalized Old/New
# ═══════════════════════════════════════════════════════════════════════════════

def make_table_5_5(outdir: Path) -> str:
    data = CONFIG["per_stage"]
    joint = data.get(JOINT, {})

    # Compute normalized values: method_mAP / joint_mAP
    stages = ["stage1", "stage2", "stage3"]
    splits = ["old", "new"]

    # Pre-compute all normalized values
    norm: dict[str, dict[str, dict[str, float | None]]] = {}
    for method in METHODS:
        norm[method] = {}
        md = data.get(method, {})
        for stage in stages:
            norm[method][stage] = {}
            for split in splits:
                if stage == "stage1" and split == "old":
                    norm[method][stage][split] = None  # no old classes
                    continue
                m_val = md.get(stage, {}).get(split)
                j_val = joint.get(stage, {}).get(split)
                if m_val is None or j_val is None or j_val == 0:
                    norm[method][stage][split] = None
                else:
                    norm[method][stage][split] = m_val / j_val

    # Find best per column (excluding joint)
    col_keys = []
    for stage in stages:
        for split in splits:
            if stage == "stage1" and split == "old":
                continue
            col_keys.append((stage, split))

    best_per_col: dict[tuple, str | None] = {}
    for key in col_keys:
        stage, split = key
        vals = {
            m: norm[m][stage][split]
            for m in METHODS if norm[m][stage][split] is not None
        }
        best_per_col[key] = _find_best(vals, minimize=False)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Per-stage normalized detection performance on the Military-UAV dataset.",
        r"Values represent the ratio of each method's mAP@0.5 to that of joint training",
        r"at the corresponding stage. A value of 1.00 indicates parity with the upper bound.}",
        r"\label{tab:military-oldnew}",
        r"\begin{tabular}{l cc cc cc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Stage 1} & \multicolumn{2}{c}{Stage 2}"
        r" & \multicolumn{2}{c}{Stage 3} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"Method & Old & New & Old & New & Old & New \\",
        r"\midrule",
    ]

    for i, method in enumerate(METHODS):
        if i in GROUP_BREAKS:
            lines.append(r"\midrule")
        cells = [_short_name(method)]
        for stage in stages:
            for split in splits:
                if stage == "stage1" and split == "old":
                    cells.append("--")
                    continue
                v = norm[method][stage][split]
                is_best = (best_per_col.get((stage, split)) == method)
                cells.append(_fmt_2(v, best=is_best))
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    content = "\n".join(lines)

    # Plain-text summary
    print("\n  Table 5.5 — Military-UAV Normalized Old/New Per Stage")
    hdr = f"  {'Method':<25}"
    for stage in stages:
        for split in splits:
            if stage == "stage1" and split == "old":
                continue
            hdr += f" {stage[5]}_{split[0]:>5}"
    print(hdr)
    print("  " + "-" * 65)
    for method in METHODS:
        row = f"  {_short_name(method):<25}"
        for stage in stages:
            for split in splits:
                if stage == "stage1" and split == "old":
                    continue
                v = norm[method][stage][split]
                row += f" {v:>6.2f}" if v is not None else f" {'--':>6}"
        print(row)

    _write_and_print(outdir / "table_5_5.tex", content, "Table 5.5")
    return content


# ═══════════════════════════════════════════════════════════════════════════════
# Table 5.6 — Per-Size RSD
# ═══════════════════════════════════════════════════════════════════════════════

def make_table_5_6(outdir: Path) -> str:
    size_cfgs = CONFIG["size_jsons"]

    # Load JSONs and compute RSD per size bin
    rsd_by_method: dict[str, dict[str, float | None]] = {}

    for method in METHODS:
        rsd_by_method[method] = {}
        paths = size_cfgs.get(method, {})
        j1 = _load_json(paths.get("stage1", ""))
        j3 = _load_json(paths.get("stage3", ""))

        for size_bin in SIZE_BINS:
            if j1 is None or j3 is None:
                rsd_by_method[method][size_bin] = None
                continue
            s1_data = j1.get(size_bin, {})
            s3_data = j3.get(size_bin, {})
            map1 = s1_data.get("mAP50")
            map3 = s3_data.get("mAP50")
            if map1 is None or map3 is None or map1 == 0:
                rsd_by_method[method][size_bin] = None
            else:
                rsd_by_method[method][size_bin] = (map1 - map3) / map1 * 100

    # Find best per column (lowest RSD, excluding joint)
    best_per_size: dict[str, str | None] = {}
    for size_bin in SIZE_BINS:
        vals = {m: rsd_by_method[m][size_bin] for m in METHODS
                if rsd_by_method[m][size_bin] is not None}
        best_per_size[size_bin] = _find_best(vals, minimize=True)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Size-stratified forgetting (RSD) on the Military-UAV dataset.",
        r"Higher RSD indicates greater performance degradation on previously learned classes.}",
        r"\label{tab:military-size}",
        r"\begin{tabular}{l ccc}",
        r"\toprule",
        r"& Tiny ($<$16$\times$16) & Small (16--32) & Medium+ ($>$32$\times$32) \\",
        r"Method & RSD (\%) & RSD (\%) & RSD (\%) \\",
        r"\midrule",
    ]

    for i, method in enumerate(METHODS):
        if i in GROUP_BREAKS:
            lines.append(r"\midrule")
        cells = [_short_name(method)]
        for size_bin in SIZE_BINS:
            v = rsd_by_method[method][size_bin]
            is_best = (best_per_size.get(size_bin) == method)
            cells.append(_fmt_1(v, best=is_best))
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    content = "\n".join(lines)

    # Plain-text summary
    print("\n  Table 5.6 — Military-UAV Per-Size RSD")
    print(f"  {'Method':<25} {'Tiny':>8} {'Small':>8} {'Med+':>8}")
    print("  " + "-" * 50)
    for method in METHODS:
        row = f"  {_short_name(method):<25}"
        for size_bin in SIZE_BINS:
            v = rsd_by_method[method][size_bin]
            row += f" {v:>8.1f}" if v is not None else f" {'--':>8}"
        print(row)

    _write_and_print(outdir / "table_5_6.tex", content, "Table 5.6")
    return content


# ═══════════════════════════════════════════════════════════════════════════════
# Table 5.7 — Cross-Dataset Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def make_table_5_7(outdir: Path) -> str:
    dota = CONFIG["dota_overall"]
    mil = CONFIG["overall"]

    # Find bests per column (excluding joint)
    dota_rsd_vals = {m: dota[m]["RSD"] for m in METHODS if m in dota}
    dota_omg_vals = {m: dota[m]["omega"] for m in METHODS if m in dota}
    mil_rsd_vals = {m: mil[m]["RSD"] for m in METHODS if m in mil}
    mil_omg_vals = {m: mil[m]["omega"] for m in METHODS if m in mil}

    best_dota_rsd = _find_best(dota_rsd_vals, minimize=True)
    best_dota_omg = _find_best(dota_omg_vals, minimize=False)
    best_mil_rsd = _find_best(mil_rsd_vals, minimize=True)
    best_mil_omg = _find_best(mil_omg_vals, minimize=False)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Cross-dataset comparison of continual learning methods.",
        r"DOTA uses a shared datalake where old-class objects appear unannotated in new stages.",
        r"Military-UAV uses individual datastreams where each stage contains distinct images.}",
        r"\label{tab:cross-dataset}",
        r"\begin{tabular}{l cc cc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{DOTA (shared)}"
        r" & \multicolumn{2}{c}{Military-UAV (individual)} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"Method & RSD (\%) $\downarrow$ & $\Omega_{\text{all}}$ $\uparrow$"
        r" & RSD (\%) $\downarrow$ & $\Omega_{\text{all}}$ $\uparrow$ \\",
        r"\midrule",
    ]

    for i, method in enumerate(METHODS):
        if i in GROUP_BREAKS:
            lines.append(r"\midrule")
        d = dota.get(method, {})
        m = mil.get(method, {})

        dr = _fmt_1(d.get("RSD"), best=(method == best_dota_rsd))
        do = _fmt_2(d.get("omega"), best=(method == best_dota_omg))
        mr = _fmt_1(m.get("RSD"), best=(method == best_mil_rsd))
        mo = _fmt_2(m.get("omega"), best=(method == best_mil_omg))

        lines.append(f"{_short_name(method)} & {dr} & {do} & {mr} & {mo} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    content = "\n".join(lines)

    # Plain-text summary
    print("\n  Table 5.7 — Cross-Dataset Comparison")
    print(f"  {'Method':<25} {'D-RSD':>7} {'D-Omg':>7} {'M-RSD':>7} {'M-Omg':>7}")
    print("  " + "-" * 55)
    for method in METHODS:
        d = dota.get(method, {})
        m = mil.get(method, {})
        print(
            f"  {_short_name(method):<25}"
            f" {d.get('RSD', 0):>7.1f} {d.get('omega', 0):>7.2f}"
            f" {m.get('RSD', 0):>7.1f} {m.get('omega', 0):>7.2f}"
        )

    _write_and_print(outdir / "table_5_7.tex", content, "Table 5.7")
    return content


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Military-UAV LaTeX tables (relative metrics only).",
    )
    parser.add_argument(
        "--outdir", default="results/military_tables",
        help="Directory for .tex output files.",
    )
    args = parser.parse_args()
    outdir = Path(args.outdir)

    make_table_5_4(outdir)
    make_table_5_5(outdir)
    make_table_5_6(outdir)
    make_table_5_7(outdir)

    print(f"\n{'━' * 72}")
    print(f"  All tables saved to: {outdir}/")
    print(f"{'━' * 72}\n")


if __name__ == "__main__":
    main()
