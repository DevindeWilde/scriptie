#!/usr/bin/env python3
"""
dataset_stats.py — YOLO-format dataset statistics for tiny-object detection.

Computes per-class and per-split statistics:
  - Image counts, instance counts
  - Object area: mean, median, min, max (in pixels²)
  - Size-category breakdown: tiny (<32×32 px), small (32–96 px), medium+ (>96×96 px)

Outputs:
  - Console table
  - dataset_statistics.csv
  - dataset_statistics.tex  (two LaTeX tables: counts + size stats)
  - bar_chart.png, size_histogram.png, size_boxplot.png  (if matplotlib available)

All outputs saved to:  <repo_root>/dataset_stats/<dataset_name>/

Usage:
    python scripts/dataset_stats.py --data_path datasets/dota-filtered-full
    python scripts/dataset_stats.py --data_path datasets/dota-filtered-full \\
        --data_yaml yaml/DOTA-filtered.yaml --imgsz 640
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Optional: PyYAML
try:
    import yaml as _yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Optional: matplotlib (may fail with NumPy version mismatch)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:  # noqa: BLE001
    HAS_MPL = False

# ── Size thresholds ────────────────────────────────────────────────────────────
TINY_MAX_SIDE  = 16   # px  — <16×16 counts as "tiny"
SMALL_MAX_SIDE = 32   # px  — 16–32 px counts as "small"
TINY_MAX_AREA  = TINY_MAX_SIDE  ** 2   #   256
SMALL_MAX_AREA = SMALL_MAX_SIDE ** 2   # 1 024


# ── Data loading ───────────────────────────────────────────────────────────────

def load_class_names(data_yaml: Optional[Path]) -> Dict[int, str]:
    """Read class names from a YOLO data YAML.  Returns {} on failure."""
    if data_yaml is None or not data_yaml.exists():
        return {}
    if not HAS_YAML:
        print("  ⚠  PyYAML not installed — class names will be inferred from label IDs.")
        return {}
    with open(data_yaml, encoding="utf-8") as fh:
        cfg = _yaml.safe_load(fh)
    names = cfg.get("names", {})
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def parse_label_file(
    path: Path,
    imgsz: int,
) -> List[Tuple[int, float, float, float]]:
    """
    Parse one YOLO label file.

    Returns list of (class_id, w_px, h_px, area_px2).
    Skips malformed lines silently.
    """
    rows = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls   = int(parts[0])
                w_px  = float(parts[3]) * imgsz
                h_px  = float(parts[4]) * imgsz
                area  = w_px * h_px
                rows.append((cls, w_px, h_px, area))
    except (OSError, ValueError):
        pass
    return rows


def collect_split_stats(split_dir: Path, imgsz: int) -> Optional[Dict]:
    """
    Walk `split_dir/labels/` and accumulate per-class statistics.

    Returns None if the labels directory does not exist or is empty.
    """
    labels_dir = split_dir / "labels"
    if not labels_dir.exists():
        return None

    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        return None

    per_class_areas:  Dict[int, List[float]]               = defaultdict(list)
    per_class_widths: Dict[int, List[float]]               = defaultdict(list)
    per_class_heights: Dict[int, List[float]]              = defaultdict(list)
    per_class_images: Dict[int, set]                       = defaultdict(set)
    total_instances = 0

    n = len(label_files)
    print(f"    Parsing {n:,} label files", end="", flush=True)
    for i, lf in enumerate(label_files):
        if i % 500 == 0 and i > 0:
            print(f" {i:,}", end="", flush=True)
        stem = lf.stem
        for cls_id, w_px, h_px, area in parse_label_file(lf, imgsz):
            per_class_areas[cls_id].append(area)
            per_class_widths[cls_id].append(w_px)
            per_class_heights[cls_id].append(h_px)
            per_class_images[cls_id].add(stem)
            total_instances += 1
    print(" — done.")

    return {
        "n_images":          len(label_files),
        "total_instances":   total_instances,
        "per_class_areas":   dict(per_class_areas),
        "per_class_widths":  dict(per_class_widths),
        "per_class_heights": dict(per_class_heights),
        "per_class_images":  {k: len(v) for k, v in per_class_images.items()},
    }


# ── Statistics computation ─────────────────────────────────────────────────────

def class_stats(areas: List[float]) -> Dict:
    """Descriptive statistics for a list of pixel areas."""
    if not areas:
        return {}
    a = np.asarray(areas, dtype=np.float64)
    return {
        "n":           int(len(a)),
        "mean_area":   float(np.mean(a)),
        "median_area": float(np.median(a)),
        "min_area":    float(np.min(a)),
        "max_area":    float(np.max(a)),
        "std_area":    float(np.std(a)),
        "pct_tiny":    float(np.mean(a <  TINY_MAX_AREA)                         * 100),
        "pct_small":   float(np.mean((a >= TINY_MAX_AREA) & (a < SMALL_MAX_AREA)) * 100),
        "pct_medium":  float(np.mean(a >= SMALL_MAX_AREA)                        * 100),
    }


def _side(area: float) -> str:
    """Format an area as approx equivalent square side, e.g. '24 px'."""
    s = area ** 0.5
    return f"{s:.1f}"


# ── Console report ─────────────────────────────────────────────────────────────

def _cn(cid: int, class_names: Dict[int, str]) -> str:
    return class_names.get(cid, f"class_{cid}")


def print_console_report(
    splits: Dict[str, Optional[Dict]],
    class_names: Dict[int, str],
    all_class_ids: List[int],
) -> None:
    active = {k: v for k, v in splits.items() if v is not None}

    for split_name, data in active.items():
        print(f"\n{'═' * 80}")
        print(
            f"  Split : {split_name.upper():<8}"
            f"  Images: {data['n_images']:,}"
            f"  Instances: {data['total_instances']:,}"
            f"  Avg inst/image: {data['total_instances'] / max(data['n_images'], 1):.1f}"
        )
        print(f"{'═' * 80}")

        hdr = (
            f"  {'Class':<18} {'Imgs':>6} {'Inst':>8}  "
            f"{'Mean':>9} {'Median':>9} {'Min':>8} {'Max':>9}  "
            f"{'Tiny%':>6} {'Sml%':>6} {'Med+%':>6}"
        )
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))

        for cid in all_class_ids:
            areas  = data["per_class_areas"].get(cid, [])
            n_img  = data["per_class_images"].get(cid, 0)
            st     = class_stats(areas)
            if not st:
                print(f"  {_cn(cid, class_names):<18} {'—':>6} {'—':>8}")
                continue
            print(
                f"  {_cn(cid, class_names):<18} {n_img:>6,} {st['n']:>8,}  "
                f"{st['mean_area']:>9,.0f} {st['median_area']:>9,.0f} "
                f"{st['min_area']:>8,.0f} {st['max_area']:>9,.0f}  "
                f"{st['pct_tiny']:>5.1f}% {st['pct_small']:>5.1f}% {st['pct_medium']:>5.1f}%"
            )

        # Totals row
        all_areas = [a for cid in all_class_ids for a in data["per_class_areas"].get(cid, [])]
        if all_areas:
            a = np.asarray(all_areas)
            print("  " + "─" * (len(hdr) - 2))
            print(
                f"  {'TOTAL':<18} {data['n_images']:>6,} {data['total_instances']:>8,}  "
                f"{np.mean(a):>9,.0f} {np.median(a):>9,.0f} "
                f"{np.min(a):>8,.0f} {np.max(a):>9,.0f}  "
                f"{np.mean(a < TINY_MAX_AREA)*100:>5.1f}% "
                f"{np.mean((a>=TINY_MAX_AREA)&(a<SMALL_MAX_AREA))*100:>5.1f}% "
                f"{np.mean(a >= SMALL_MAX_AREA)*100:>5.1f}%"
            )

    # Cross-split summary
    print(f"\n{'─' * 80}")
    print("  CROSS-SPLIT SUMMARY")
    print(f"{'─' * 80}")
    col_w = 12
    header = f"  {'Split':<10}" + "".join(
        f"  {cn:>{col_w}}"
        for cn in [_cn(cid, class_names) for cid in all_class_ids] + ["TOTAL"]
    )
    print(header + "  (instances)")
    for split_name, data in active.items():
        row = f"  {split_name:<10}"
        total = 0
        for cid in all_class_ids:
            n = len(data["per_class_areas"].get(cid, []))
            row += f"  {n:>{col_w},}"
            total += n
        row += f"  {total:>{col_w},}"
        print(row)


# ── CSV output ─────────────────────────────────────────────────────────────────

def write_csv(
    out_path: Path,
    splits: Dict[str, Optional[Dict]],
    class_names: Dict[int, str],
    all_class_ids: List[int],
) -> None:
    fields = [
        "split", "class_id", "class_name",
        "n_images", "n_instances", "avg_inst_per_image",
        "mean_area_px2", "median_area_px2", "min_area_px2", "max_area_px2", "std_area_px2",
        "mean_side_px", "median_side_px",
        "pct_tiny", "pct_small", "pct_medium",
    ]
    rows = []
    for split_name, data in splits.items():
        if data is None:
            continue
        for cid in all_class_ids:
            areas = data["per_class_areas"].get(cid, [])
            n_img = data["per_class_images"].get(cid, 0)
            st    = class_stats(areas)
            rows.append({
                "split":               split_name,
                "class_id":            cid,
                "class_name":          _cn(cid, class_names),
                "n_images":            n_img,
                "n_instances":         st.get("n", 0),
                "avg_inst_per_image":  round(st.get("n", 0) / max(n_img, 1), 3),
                "mean_area_px2":       round(st.get("mean_area",   0.0), 2),
                "median_area_px2":     round(st.get("median_area", 0.0), 2),
                "min_area_px2":        round(st.get("min_area",    0.0), 2),
                "max_area_px2":        round(st.get("max_area",    0.0), 2),
                "std_area_px2":        round(st.get("std_area",    0.0), 2),
                "mean_side_px":        round(st.get("mean_area",   0.0) ** 0.5, 2),
                "median_side_px":      round(st.get("median_area", 0.0) ** 0.5, 2),
                "pct_tiny":            round(st.get("pct_tiny",    0.0), 2),
                "pct_small":           round(st.get("pct_small",   0.0), 2),
                "pct_medium":          round(st.get("pct_medium",  0.0), 2),
            })
        # Total row for this split
        all_areas = [a for cid in all_class_ids for a in data["per_class_areas"].get(cid, [])]
        if all_areas:
            a  = np.asarray(all_areas)
            ni = data["n_images"]
            ti = data["total_instances"]
            rows.append({
                "split":               split_name,
                "class_id":            -1,
                "class_name":          "TOTAL",
                "n_images":            ni,
                "n_instances":         ti,
                "avg_inst_per_image":  round(ti / max(ni, 1), 3),
                "mean_area_px2":       round(float(np.mean(a)),   2),
                "median_area_px2":     round(float(np.median(a)), 2),
                "min_area_px2":        round(float(np.min(a)),    2),
                "max_area_px2":        round(float(np.max(a)),    2),
                "std_area_px2":        round(float(np.std(a)),    2),
                "mean_side_px":        round(float(np.mean(a)**0.5),   2),
                "median_side_px":      round(float(np.median(a)**0.5), 2),
                "pct_tiny":            round(float(np.mean(a < TINY_MAX_AREA)  * 100), 2),
                "pct_small":           round(float(np.mean((a>=TINY_MAX_AREA)&(a<SMALL_MAX_AREA))*100), 2),
                "pct_medium":          round(float(np.mean(a >= SMALL_MAX_AREA) * 100), 2),
            })
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  ✓ CSV        → {out_path}")


# ── LaTeX output ───────────────────────────────────────────────────────────────

def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&")


def write_latex(
    out_path: Path,
    splits: Dict[str, Optional[Dict]],
    class_names: Dict[int, str],
    all_class_ids: List[int],
    dataset_name: str,
    imgsz: int,
) -> None:
    """
    Write two complementary LaTeX tables:
      Table A — per-class instance/image counts per split
      Table B — per-class size statistics (train split)
    """
    active_splits = [s for s, d in splits.items() if d is not None]
    label_base = dataset_name.replace("-", "_").replace(" ", "_")

    def cn(cid: int) -> str:
        return _tex_escape(_cn(cid, class_names))

    lines: List[str] = []

    # ── Table A: counts ──────────────────────────────────────────────────────
    n_sp = len(active_splits)
    col_spec_a = "l" + "rr" * n_sp
    lines += [
        r"% ─────────────────────────────────────────────────────────────────",
        r"% Table A: Instance and image counts per class and split",
        r"% ─────────────────────────────────────────────────────────────────",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Instance and image counts per class in the "
        + _tex_escape(dataset_name)
        + r" dataset.}",
        f"\\label{{tab:{label_base}_counts}}",
        r"\begin{tabular}{" + col_spec_a + "}",
        r"\toprule",
    ]

    # Header row 1
    split_headers = " & ".join(
        f"\\multicolumn{{2}}{{c}}{{{s.capitalize()}}}" for s in active_splits
    )
    lines.append(f"\\textbf{{Class}} & {split_headers} \\\\")

    # Cmidrules
    cmidrule_parts = []
    col = 2
    for _ in active_splits:
        cmidrule_parts.append(f"\\cmidrule(lr){{{col}-{col+1}}}")
        col += 2
    lines.append(" ".join(cmidrule_parts))

    # Header row 2
    sub_hdrs = " & ".join(
        "\\textbf{Images} & \\textbf{Instances}" for _ in active_splits
    )
    lines.append(f" & {sub_hdrs} \\\\")
    lines.append(r"\midrule")

    # Data rows
    for cid in all_class_ids:
        row = [cn(cid)]
        for s in active_splits:
            data = splits[s]
            if data is None:
                row += ["—", "—"]
            else:
                n_img  = data["per_class_images"].get(cid, 0)
                n_inst = len(data["per_class_areas"].get(cid, []))
                row += [f"{n_img:,}", f"{n_inst:,}"]
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\midrule")

    # Total row
    total_row = [r"\textbf{Total}"]
    for s in active_splits:
        data = splits[s]
        if data is None:
            total_row += ["—", "—"]
        else:
            total_row += [
                f"\\textbf{{{data['n_images']:,}}}",
                f"\\textbf{{{data['total_instances']:,}}}",
            ]
    lines.append(" & ".join(total_row) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]

    # ── Table B: size statistics ─────────────────────────────────────────────
    lines += [
        r"% ─────────────────────────────────────────────────────────────────",
        r"% Table B: Object size statistics per class (training split)",
        r"% ─────────────────────────────────────────────────────────────────",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Object size statistics per class (training split, "
        + f"{imgsz}$\\times${imgsz}"
        + r"~px images). "
        + r"Tiny: $<$32\,$\times$\,32\,px; Small: 32--96\,px; Medium+: $>$96\,$\times$\,96\,px.}",
        f"\\label{{tab:{label_base}_sizes}}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"\textbf{Class} & \textbf{N} & \textbf{Mean (px\textsuperscript{2})} "
        r"& \textbf{Median} & \textbf{Min} & \textbf{Max} "
        r"& \textbf{Tiny (\%)} & \textbf{Small (\%)} & \textbf{Med+ (\%)} \\",
        r"\midrule",
    ]

    train_data = splits.get("train")
    for cid in all_class_ids:
        areas = train_data["per_class_areas"].get(cid, []) if train_data else []
        st    = class_stats(areas)
        if not st:
            lines.append(cn(cid) + " & — & — & — & — & — & — & — & — \\\\")
            continue
        lines.append(
            f"{cn(cid)} & {st['n']:,} "
            f"& {st['mean_area']:,.0f} "
            f"& {st['median_area']:,.0f} "
            f"& {st['min_area']:,.0f} "
            f"& {st['max_area']:,.0f} "
            f"& {st['pct_tiny']:.1f} "
            f"& {st['pct_small']:.1f} "
            f"& {st['pct_medium']:.1f} \\\\"
        )

    lines.append(r"\midrule")

    # Total row for Table B
    if train_data:
        all_areas = [a for cid in all_class_ids for a in train_data["per_class_areas"].get(cid, [])]
        if all_areas:
            a = np.asarray(all_areas)
            lines.append(
                f"\\textbf{{Total}} & \\textbf{{{len(a):,}}} "
                f"& \\textbf{{{np.mean(a):,.0f}}} "
                f"& \\textbf{{{np.median(a):,.0f}}} "
                f"& \\textbf{{{np.min(a):,.0f}}} "
                f"& \\textbf{{{np.max(a):,.0f}}} "
                f"& \\textbf{{{np.mean(a < TINY_MAX_AREA)*100:.1f}}} "
                f"& \\textbf{{{np.mean((a>=TINY_MAX_AREA)&(a<SMALL_MAX_AREA))*100:.1f}}} "
                f"& \\textbf{{{np.mean(a >= SMALL_MAX_AREA)*100:.1f}}} \\\\"
            )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  ✓ LaTeX      → {out_path}")


# ── Visualizations ─────────────────────────────────────────────────────────────

def make_bar_chart(
    splits: Dict[str, Optional[Dict]],
    class_names: Dict[int, str],
    all_class_ids: List[int],
    out_path: Path,
) -> None:
    """Grouped bar chart: instance count per class, one bar per split."""
    active = {s: d for s, d in splits.items() if d is not None}
    x = np.arange(len(all_class_ids))
    n_sp  = len(active)
    width = 0.75 / max(n_sp, 1)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (split_name, data) in enumerate(active.items()):
        counts = [len(data["per_class_areas"].get(cid, [])) for cid in all_class_ids]
        offset = (i - n_sp / 2 + 0.5) * width
        bars = ax.bar(x + offset, counts, width, label=split_name.capitalize(),
                      color=colors[i % len(colors)], edgecolor="white", linewidth=0.4)
        max_c = max(counts) if counts else 1
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max_c * 0.012,
                    f"{count:,}", ha="center", va="bottom", fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([_cn(cid, class_names) for cid in all_class_ids],
                       rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Instance count")
    ax.set_title("Instance counts per class and split")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Bar chart  → {out_path}")


def make_size_histogram(
    splits: Dict[str, Optional[Dict]],
    class_names: Dict[int, str],
    all_class_ids: List[int],
    out_path: Path,
) -> None:
    """Histogram of object pixel areas with size-threshold lines (train split)."""
    data = splits.get("train")
    if data is None:
        data = next((d for d in splits.values() if d is not None), None)
    if data is None:
        return

    all_areas = [a for cid in all_class_ids for a in data["per_class_areas"].get(cid, [])]
    if not all_areas:
        return

    a = np.asarray(all_areas)
    clip = float(np.percentile(a, 99))
    a_clip = np.clip(a, 0, clip)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(a_clip, bins=100, color="#4C72B0", edgecolor="white", linewidth=0.2, alpha=0.85)

    ymax = ax.get_ylim()[1]
    for thresh, label, color in [
        (TINY_MAX_AREA,  f"Tiny/Small\n({TINY_MAX_SIDE}×{TINY_MAX_SIDE} px²)",  "#E05C5C"),
        (SMALL_MAX_AREA, f"Small/Med+\n({SMALL_MAX_SIDE}×{SMALL_MAX_SIDE} px²)", "#E09020"),
    ]:
        if thresh <= clip:
            ax.axvline(thresh, color=color, linestyle="--", linewidth=1.5)
            ax.text(thresh, ymax * 0.96, label, color=color,
                    ha="center", va="top", fontsize=8,
                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))

    pct_tiny  = np.mean(a < TINY_MAX_AREA) * 100
    pct_small = np.mean((a >= TINY_MAX_AREA) & (a < SMALL_MAX_AREA)) * 100
    pct_med   = np.mean(a >= SMALL_MAX_AREA) * 100
    ax.text(
        0.99, 0.97,
        f"Tiny: {pct_tiny:.1f}%   Small: {pct_small:.1f}%   Med+: {pct_med:.1f}%",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(fc="white", ec="grey", alpha=0.8, pad=3),
    )
    ax.set_xlabel(f"Object area (px²)  [clipped at 99th pct = {clip:,.0f}]")
    ax.set_ylabel("Count")
    ax.set_title("Object size distribution (train split)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Histogram  → {out_path}")


def make_size_histogram_per_class(
    splits: Dict[str, Optional[Dict]],
    class_names: Dict[int, str],
    all_class_ids: List[int],
    out_path: Path,
) -> None:
    """One-row subplot: histogram per class (train split)."""
    data = splits.get("train")
    if data is None:
        return
    n_cls = len(all_class_ids)
    if n_cls == 0:
        return

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
              "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
    fig, axes = plt.subplots(1, n_cls, figsize=(3 * n_cls, 3.5), sharey=False)
    if n_cls == 1:
        axes = [axes]

    for ax, cid, color in zip(axes, all_class_ids, colors):
        areas = np.asarray(data["per_class_areas"].get(cid, [0]))
        clip  = float(np.percentile(areas, 99)) if len(areas) > 1 else float(np.max(areas))
        a_c   = np.clip(areas, 0, clip)
        ax.hist(a_c, bins=40, color=color, edgecolor="white", linewidth=0.2, alpha=0.85)
        for thresh, tc in [(TINY_MAX_AREA, "#E05C5C"), (SMALL_MAX_AREA, "#E09020")]:
            if thresh <= clip:
                ax.axvline(thresh, color=tc, linestyle="--", linewidth=1.2)
        ax.set_title(_cn(cid, class_names), fontsize=9)
        ax.set_xlabel("area (px²)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Count")
    fig.suptitle("Size distribution per class (train)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Per-class hist → {out_path}")


def make_boxplot(
    splits: Dict[str, Optional[Dict]],
    class_names: Dict[int, str],
    all_class_ids: List[int],
    out_path: Path,
) -> None:
    """Box plot of size distribution per class (train split, outliers hidden)."""
    data = splits.get("train")
    if data is None:
        return

    data_per_class = [np.asarray(data["per_class_areas"].get(cid, [0.0]))
                      for cid in all_class_ids]
    labels = [_cn(cid, class_names) for cid in all_class_ids]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
              "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]

    fig, ax = plt.subplots(figsize=(max(7, len(all_class_ids) * 1.4), 5))
    bp = ax.boxplot(
        data_per_class, labels=labels, patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
        showfliers=False,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for thresh, label, color in [
        (TINY_MAX_AREA,  f"Tiny/Small ({TINY_MAX_SIDE}×{TINY_MAX_SIDE})", "#E05C5C"),
        (SMALL_MAX_AREA, f"Small/Med+ ({SMALL_MAX_SIDE}×{SMALL_MAX_SIDE})", "#E09020"),
    ]:
        ax.axhline(thresh, color=color, linestyle="--", linewidth=1.2, label=label)

    ax.set_ylabel("Object area (px²)  [outliers hidden]")
    ax.set_title("Size distribution per class (train split)")
    ax.legend(fontsize=8, loc="upper right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.tick_params(axis="x", rotation=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Box plot   → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute YOLO-format dataset statistics for object detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_path", required=True,
        help="Root directory of the dataset (must contain train/, val/, and/or test/ subdirs).",
    )
    parser.add_argument(
        "--data_yaml", default=None,
        help="YOLO data YAML file to read class names from (optional).",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Image size in pixels (assumed square).",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        help="Splits to process.",
    )
    parser.add_argument(
        "--out_root", default=None,
        help="Root output directory. Defaults to <repo>/dataset_stats/<dataset_name>/.",
    )
    parser.add_argument(
        "--no_plots", action="store_true",
        help="Skip generating PNG visualizations.",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path).resolve()
    if not data_path.exists():
        print(f"✗  Data path not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    dataset_name = data_path.name

    # Resolve output directory
    if args.out_root:
        out_dir = Path(args.out_root).resolve() / dataset_name
    else:
        repo_root = Path(__file__).resolve().parent.parent
        out_dir   = repo_root / "dataset_stats" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Class names
    yaml_path   = Path(args.data_yaml).resolve() if args.data_yaml else None
    class_names = load_class_names(yaml_path)

    print(f"\n{'━' * 64}")
    print(f"  Dataset  : {dataset_name}")
    print(f"  Path     : {data_path}")
    print(f"  Image sz : {args.imgsz}×{args.imgsz} px")
    print(f"  Splits   : {args.splits}")
    print(f"  Output   : {out_dir}")
    if class_names:
        print(f"  Classes  : {dict(sorted(class_names.items()))}")
    print(f"{'━' * 64}")

    # ── Collect stats ──────────────────────────────────────────────────────────
    splits: Dict[str, Optional[Dict]] = {}
    for split_name in args.splits:
        split_dir = data_path / split_name
        print(f"\n[{split_name.upper()}]  {split_dir}")
        if not split_dir.exists():
            print(f"  ⚠  Directory not found — skipping.")
            splits[split_name] = None
            continue
        result = collect_split_stats(split_dir, args.imgsz)
        if result is None:
            print(f"  ⚠  No .txt label files in {split_dir / 'labels'} — skipping.")
        splits[split_name] = result

    # All class IDs seen in any split
    all_class_ids = sorted({
        cid
        for data in splits.values()
        if data is not None
        for cid in data["per_class_areas"]
    })

    if not all_class_ids:
        print("\n✗  No labels found across any split. Check --data_path.", file=sys.stderr)
        sys.exit(1)

    # Fill in any missing class names
    for cid in all_class_ids:
        class_names.setdefault(cid, f"class_{cid}")

    # ── Outputs ────────────────────────────────────────────────────────────────
    print()
    print_console_report(splits, class_names, all_class_ids)

    print(f"\n{'━' * 64}")
    print("  Writing output files …")
    write_csv(out_dir / "dataset_statistics.csv", splits, class_names, all_class_ids)
    write_latex(out_dir / "dataset_statistics.tex", splits, class_names, all_class_ids,
                dataset_name, args.imgsz)

    if not args.no_plots:
        if not HAS_MPL:
            print(
                "  ⚠  matplotlib unavailable (NumPy version mismatch?) — skipping plots.\n"
                "     Run: pip install --upgrade matplotlib  or use a virtual environment."
            )
        else:
            print("  Generating visualisations …")
            make_bar_chart(splits, class_names, all_class_ids,
                           out_dir / "bar_chart.png")
            make_size_histogram(splits, class_names, all_class_ids,
                                out_dir / "size_histogram.png")
            make_size_histogram_per_class(splits, class_names, all_class_ids,
                                          out_dir / "size_histogram_per_class.png")
            make_boxplot(splits, class_names, all_class_ids,
                         out_dir / "size_boxplot.png")

    print(f"\n{'━' * 64}")
    print(f"  ✅  Done.  Results in: {out_dir}")
    print(f"{'━' * 64}\n")


if __name__ == "__main__":
    main()
