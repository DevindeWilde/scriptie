"""Assemble prototype crop grids into a thesis-ready stacked PDF figure.

Reads the cropped patches saved by visualize_prototypes_val.py and arranges
them into grids: rows = FPN levels, columns = prototype slots.
All classes are stacked vertically into a single PDF with sub-titles
like "(a) small-vehicle", "(b) plane".

Usage:
    python scripts/assemble_prototype_grid.py \
        --outdir prototype_val_matches \
        --classes small-vehicle,plane

    # With LaTeX fonts (requires LaTeX installation):
    python scripts/assemble_prototype_grid.py \
        --outdir prototype_val_matches \
        --classes small-vehicle,plane \
        --usetex
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread


def parse_sim_from_filename(name: str) -> float:
    """Extract cosine similarity from filename like slot0_P0001_sim0.873.png."""
    m = re.search(r"sim(\d+\.\d+)", name)
    return float(m.group(1)) if m else 0.0


def detect_levels(cls_dir: Path) -> list[str]:
    """Auto-detect FPN level subdirectories, sorted by numeric suffix."""
    return sorted(
        [d.name for d in cls_dir.iterdir() if d.is_dir()],
        key=lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 0,
    )


def main():
    ap = argparse.ArgumentParser(description="Assemble prototype crops into a stacked grid PDF")
    ap.add_argument("--outdir", required=True, help="Directory with crops from visualize_prototypes_val.py")
    ap.add_argument("--classes", required=True,
                    help="Comma-separated class names to generate grids for")
    ap.add_argument("--levels", default=None,
                    help="Comma-separated FPN levels (default: auto-detect from dirs)")
    ap.add_argument("--slots", type=int, default=4, help="Number of prototype slots")
    ap.add_argument("--usetex", action="store_true", help="Use LaTeX for text rendering")
    ap.add_argument("--output", default=None, help="Output PDF path (default: <outdir>/prototype_grid.pdf)")
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": "serif",
        "text.usetex": args.usetex,
        "font.size": 9,
    })

    outdir = Path(args.outdir)
    classes = [c.strip() for c in args.classes.split(",")]

    # Gather per-class grid info
    class_grids = []  # list of (cls_name, label, levels, cls_dir)
    for i, cls_name in enumerate(classes):
        cls_dir = outdir / cls_name
        if not cls_dir.exists():
            print(f"  WARN: {cls_dir} not found, skipping {cls_name}")
            continue

        if args.levels:
            levels = [l.strip() for l in args.levels.split(",")]
        else:
            levels = detect_levels(cls_dir)

        if not levels:
            print(f"  WARN: no level subdirectories in {cls_dir}, skipping")
            continue

        label = f"({chr(ord('a') + i)}) {cls_name}"
        class_grids.append((cls_name, label, levels, cls_dir))
        print(f"Grid for '{cls_name}': levels={levels}, slots={args.slots}")

    if not class_grids:
        print("No valid classes found. Exiting.")
        return

    # Compute figure dimensions
    n_cols = args.slots
    fig_w = 6.5
    cell_w = fig_w / n_cols

    # Total rows = sum of level rows across all classes
    total_level_rows = sum(len(levels) for _, _, levels, _ in class_grids)
    n_classes = len(class_grids)
    # Height: cell_w per level row + spacing for class titles
    fig_h = cell_w * total_level_rows + 0.45 * n_classes + 0.15

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Use gridspec for precise layout
    # Each class gets: 1 row for title + n_level rows for images
    total_gs_rows = total_level_rows + n_classes  # title rows + image rows
    gs = fig.add_gridspec(
        total_gs_rows, n_cols,
        hspace=0.35, wspace=0.08,
        left=0.06, right=0.98, top=0.98, bottom=0.02,
    )

    # Height ratios: title rows are thin, image rows are equal
    height_ratios = []
    for _, _, levels, _ in class_grids:
        height_ratios.append(0.15)  # title row
        height_ratios.extend([1.0] * len(levels))
    gs.set_height_ratios(height_ratios)

    gs_row = 0
    for cls_idx, (cls_name, label, levels, cls_dir) in enumerate(class_grids):
        # --- Class title row ---
        title_ax = fig.add_subplot(gs[gs_row, :])
        title_ax.set_axis_off()
        title_ax.text(0.5, 0.2, label, ha="center", va="center",
                      fontsize=10, fontweight="bold",
                      transform=title_ax.transAxes)
        gs_row += 1

        # --- Image grid rows ---
        for row_i, level in enumerate(levels):
            level_dir = cls_dir / level
            for col in range(n_cols):
                ax = fig.add_subplot(gs[gs_row, col])
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(0.4)
                    spine.set_color("gray")

                # Find crop for this slot
                crops = sorted(level_dir.glob(f"slot{col}_*.png")) if level_dir.exists() else []
                if crops:
                    crop_path = crops[0]
                    img = imread(str(crop_path))
                    ax.imshow(img)
                    sim = parse_sim_from_filename(crop_path.name)
                    ax.set_xlabel(
                        f"$\\cos={sim:.2f}$" if args.usetex else f"cos={sim:.2f}",
                        fontsize=7, labelpad=2,
                    )
                else:
                    ax.text(0.5, 0.5, "---", ha="center", va="center",
                            transform=ax.transAxes, fontsize=8, color="gray")

                # Column headers (first level row of first class only)
                if cls_idx == 0 and row_i == 0:
                    ax.set_title(f"Slot {col}", fontsize=9, pad=4)

                # Row labels (first column only)
                if col == 0:
                    ax.set_ylabel(level, fontsize=8, rotation=0, labelpad=18, va="center")

            gs_row += 1

    save_path = Path(args.output) if args.output else outdir / "prototype_grid.pdf"
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
