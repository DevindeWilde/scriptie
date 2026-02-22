"""Generate pseudo-labels for continual object detection.

Run this script once before training stage t (t >= 1) to produce YOLO-format
pseudo-annotation files for old classes using the frozen stage t-1 model.

Usage:
    python scripts/generate_pseudo_labels.py \
        --weights runs/stage0/weights/best.pt \
        --data yaml/visdrone-mini.yaml \
        --prev-classes 0 1 2 \
        --outdir runs/stage1_pseudo/train \
        --conf 0.5 --conflict-iou 0.5

Output:
    One .txt file per training image in --outdir.
    Format per line: "cls cx cy w h" (YOLO normalized, float).
    Empty file = no surviving pseudo-detections for that image.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ednet import EDNet
from ednet.data.utils import img2label_paths


# ---------------------------------------------------------------------------
# IoU helpers (xyxy format)
# ---------------------------------------------------------------------------

def _box_area(boxes: np.ndarray) -> np.ndarray:
    """Area of boxes (N, 4) in xyxy format."""
    return (boxes[:, 2] - boxes[:, 0]).clip(0) * (boxes[:, 3] - boxes[:, 1]).clip(0)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute IoU between every pair of boxes in a (M,4) and b (N,4), xyxy format.

    Returns (M, N) IoU matrix.
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (inter_x2 - inter_x1).clip(0) * (inter_y2 - inter_y1).clip(0)
    area_a = _box_area(a)
    area_b = _box_area(b)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Label I/O
# ---------------------------------------------------------------------------

def _load_gt_xyxy(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """Load GT label file and return boxes as (N, 4) xyxy pixel coords.

    Returns empty array if file is missing or empty.
    """
    if not label_path.exists() or label_path.stat().st_size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    lb = np.loadtxt(str(label_path), dtype=np.float32).reshape(-1, 5)
    if len(lb) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    # Convert cx cy w h (normalized) → x1 y1 x2 y2 (pixel)
    cx, cy, w, h = lb[:, 1], lb[:, 2], lb[:, 3], lb[:, 4]
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate pseudo-labels for continual OD stages.")
    p.add_argument("--weights", required=True, help="Path to stage t-1 best.pt checkpoint.")
    p.add_argument("--data", required=True, help="Dataset YAML (e.g. yaml/visdrone-mini.yaml).")
    p.add_argument("--prev-classes", nargs="+", type=int, required=True,
                   help="Class IDs from the previous stage (e.g. 0 1 2).")
    p.add_argument("--outdir", required=True, help="Output directory for pseudo-label .txt files.")
    p.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default 0.5).")
    p.add_argument("--conflict-iou", type=float, default=0.5,
                   help="IoU threshold for conflict removal against GT (default 0.5).")
    p.add_argument("--split", default="train", choices=["train", "val", "test"],
                   help="Dataset split to process (default train).")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size (default 640).")
    p.add_argument("--batch", type=int, default=16, help="Inference batch size (default 16).")
    p.add_argument("--device", default="", help="Device: '' (auto), 'cpu', '0', '0,1', etc.")
    return p.parse_args()


def main():
    opt = parse_args()

    # Resolve paths
    weights = Path(opt.weights)
    data_yaml = Path(opt.data)
    outdir = Path(opt.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load dataset YAML to get image directory
    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)
    dataset_root = Path(data_cfg.get("path", "."))
    split_rel = data_cfg.get(opt.split, f"images/{opt.split}")
    img_dir = dataset_root / split_rel
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    # Collect image files
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    img_files = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in img_extensions)
    if not img_files:
        raise RuntimeError(f"No images found in {img_dir}")
    print(f"Found {len(img_files)} images in {img_dir}")

    # Derive GT label paths using the same logic as the training pipeline
    label_paths = img2label_paths([str(p) for p in img_files])

    # Load model
    print(f"Loading model from {weights} ...")
    model = EDNet(str(weights))

    prev_classes = sorted(set(opt.prev_classes))
    print(f"Generating pseudo-labels | prev_classes={prev_classes} | conf={opt.conf} "
          f"| conflict_iou={opt.conflict_iou}")

    kept_total = 0
    dropped_conf = 0
    dropped_conflict = 0

    for img_path, lbl_path in tqdm(zip(img_files, label_paths), total=len(img_files),
                                   desc="Pseudo-labeling"):
        img_path = Path(img_path)
        lbl_path = Path(lbl_path)
        out_file = outdir / (img_path.stem + ".txt")

        # Run inference
        results = model.predict(
            str(img_path),
            conf=opt.conf,
            classes=prev_classes,
            imgsz=opt.imgsz,
            verbose=False,
            device=opt.device,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            out_file.write_text("")
            continue

        boxes_data = results[0].boxes.data.cpu().numpy()  # (N, 6): x1 y1 x2 y2 conf cls
        img_h, img_w = results[0].orig_shape

        # Filter to prev_classes and confidence threshold (model.predict already applies conf
        # filter, but classes filter may not always be applied depending on version)
        mask = np.isin(boxes_data[:, 5].astype(int), prev_classes) & (boxes_data[:, 4] >= opt.conf)
        boxes_data = boxes_data[mask]
        dropped_conf += (~mask).sum()

        if len(boxes_data) == 0:
            out_file.write_text("")
            continue

        # Conflict removal: drop pseudo-detections with high IoU overlap with any GT box
        gt_xyxy = _load_gt_xyxy(lbl_path, img_w, img_h)
        if len(gt_xyxy) > 0:
            iou_mat = _iou_matrix(boxes_data[:, :4], gt_xyxy)  # (N_pseudo, N_gt)
            max_iou = iou_mat.max(axis=1)                      # (N_pseudo,)
            keep_mask = max_iou < opt.conflict_iou
            dropped_conflict += (~keep_mask).sum()
            boxes_data = boxes_data[keep_mask]

        if len(boxes_data) == 0:
            out_file.write_text("")
            continue

        # Convert pixel xyxy → normalized cx cy w h
        x1, y1, x2, y2 = boxes_data[:, 0], boxes_data[:, 1], boxes_data[:, 2], boxes_data[:, 3]
        cls = boxes_data[:, 5].astype(int)
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w  = (x2 - x1) / img_w
        h  = (y2 - y1) / img_h

        # Write YOLO-format label file
        lines = [f"{c} {cx_:.6f} {cy_:.6f} {w_:.6f} {h_:.6f}"
                 for c, cx_, cy_, w_, h_ in zip(cls, cx, cy, w, h)]
        out_file.write_text("\n".join(lines) + "\n")
        kept_total += len(lines)

    print(f"\nDone.")
    print(f"  Kept pseudo-detections : {kept_total}")
    print(f"  Dropped (low conf)     : {dropped_conf}")
    print(f"  Dropped (GT conflict)  : {dropped_conflict}")
    print(f"  Output directory       : {outdir}")


if __name__ == "__main__":
    main()
