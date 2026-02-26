"""Assemble prototype crop grids into thesis-ready PDF figures.

Reads the cropped patches saved by visualize_prototypes_val.py and arranges
them into a grid: rows = FPN levels, columns = prototype slots.
One PDF per class.

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
    m = re.search(r"sim([\d.]+)", name)
    return float(m.group(1)) if m else 0.0


def build_grid(cls_dir: Path, cls_name: str, levels: list[str],
               n_slots: int, save_path: Path, usetex: bool):
    """Build and save one grid figure for a single class."""
    plt.rcParams.update({
        "font.family": "serif",
        "text.usetex": usetex,
        "font.size": 9,
    })

    n_rows = len(levels)
    n_cols = n_slots
    cell_w = 6.5 / n_cols       # inches per cell
    fig_w = 6.5
    fig_h = cell_w * n_rows + 0.7  # extra for labels

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    if n_rows == 1:
        axes = axes[None, :]  # ensure 2D

    for row, level in enumerate(levels):
        level_dir = cls_dir / level
        for col in range(n_cols):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("gray")

            # Find crop for this slot
            crops = sorted(level_dir.glob(f"slot{col}_*.png")) if level_dir.exists() else []
            if crops:
                crop_path = crops[0]
                img = imread(str(crop_path))
                ax.imshow(img)
                sim = parse_sim_from_filename(crop_path.name)
                ax.set_xlabel(f"$\\cos={sim:.2f}$" if usetex else f"cos={sim:.2f}",
                              fontsize=7, labelpad=2)
            else:
                ax.text(0.5, 0.5, "---", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="gray")

            # Column labels (top row only)
            if row == 0:
                ax.set_title(f"Slot {col}", fontsize=9, pad=4)

            # Row labels (first column only)
            if col == 0:
                ax.set_ylabel(level, fontsize=9, rotation=0, labelpad=20, va="center")

    title = cls_name.replace("-", " ").replace("_", " ")
    fig.suptitle(f"Prototype matches: {title}", fontsize=11, y=1.01)
    fig.tight_layout(rect=[0.04, 0, 1, 0.98])
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {save_path}")


def main():
    ap = argparse.ArgumentParser(description="Assemble prototype crops into grid PDFs")
    ap.add_argument("--outdir", required=True, help="Directory with crops from visualize_prototypes_val.py")
    ap.add_argument("--classes", required=True,
                    help="Comma-separated class names to generate grids for")
    ap.add_argument("--levels", default=None,
                    help="Comma-separated FPN levels (default: auto-detect from dirs)")
    ap.add_argument("--slots", type=int, default=4, help="Number of prototype slots")
    ap.add_argument("--usetex", action="store_true", help="Use LaTeX for text rendering")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    classes = [c.strip() for c in args.classes.split(",")]

    for cls_name in classes:
        cls_dir = outdir / cls_name
        if not cls_dir.exists():
            print(f"  WARN: {cls_dir} not found, skipping {cls_name}")
            continue

        # Detect or use specified levels
        if args.levels:
            levels = [l.strip() for l in args.levels.split(",")]
        else:
            levels = sorted(
                [d.name for d in cls_dir.iterdir() if d.is_dir()],
                key=lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 0,
            )

        if not levels:
            print(f"  WARN: no level subdirectories in {cls_dir}, skipping")
            continue

        print(f"Grid for '{cls_name}': levels={levels}, slots={args.slots}")
        save_path = outdir / f"grid_{cls_name}.pdf"
        build_grid(cls_dir, cls_name, levels, args.slots, save_path, args.usetex)

    print("Done.")


if __name__ == "__main__":
    main()
