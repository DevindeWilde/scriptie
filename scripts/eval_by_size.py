"""Size-stratified mAP evaluation for continual object detection.

Evaluates a trained model and reports mAP broken down by object size:
  tiny   : area <  256 px²   (roughly < 16×16)
  small  : 256 ≤ area < 1024 px²  (roughly 16×16 – 32×32)
  medium+: area ≥ 1024 px²   (roughly > 32×32)

All areas are in absolute pixels at inference resolution.

Usage:
    python scripts/eval_by_size.py \\
        --weights runs/dota_saab/stage3-saab/weights/best.pt \\
        --data yaml/DOTA-stage3.yaml \\
        --split test \\
        --imgsz 640 \\
        --out results/stage3_saab_by_size.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ednet import EDNet
from ednet.data.utils import img2label_paths
from ednet.utils.metrics import ap_per_class

# ---------------------------------------------------------------------------
# Size bucket definitions (pixel area at inference resolution)
# ---------------------------------------------------------------------------
SIZE_BINS = [
    ("tiny",    0,      256),          # < 16×16
    ("small",   256,    1024),         # 16×16 – 32×32
    ("medium+", 1024,   float("inf")), # > 32×32
]
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_gt_xyxy(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """Load YOLO label file → (N, 5) float32: [cls, x1, y1, x2, y2] pixels."""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    lb = np.loadtxt(str(label_path), dtype=np.float32).reshape(-1, 5)
    if len(lb) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    cx, cy, w, h = lb[:, 1], lb[:, 2], lb[:, 3], lb[:, 4]
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    return np.concatenate([lb[:, 0:1], boxes], axis=1).astype(np.float32)


def _box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(M,4) × (N,4) xyxy → (M,N) IoU matrix."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (ix2 - ix1).clip(0) * (iy2 - iy1).clip(0)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def match_to_gt(
    pred_boxes: np.ndarray,
    pred_cls: np.ndarray,
    pred_conf: np.ndarray,
    gt_boxes: np.ndarray,
    gt_cls: np.ndarray,
) -> np.ndarray:
    """Greedy IoU matching at 10 thresholds (0.5–0.95).

    Returns tp: (N_pred, 10) bool — True if prediction correctly matched a GT
    at the corresponding IoU threshold.
    """
    n_pred = len(pred_boxes)
    tp = np.zeros((n_pred, 10), dtype=bool)
    if n_pred == 0 or len(gt_boxes) == 0:
        return tp

    iou = _box_iou(pred_boxes, gt_boxes)  # (N_pred, N_gt)
    sort_idx = np.argsort(-pred_conf)

    for ti, thr in enumerate(IOU_THRESHOLDS):
        matched_gt = set()
        for pi in sort_idx:
            pc = int(pred_cls[pi])
            best_iou, best_gi = thr - 1e-9, -1
            for gi in range(len(gt_boxes)):
                if gi in matched_gt or int(gt_cls[gi]) != pc:
                    continue
                if iou[pi, gi] > best_iou:
                    best_iou, best_gi = iou[pi, gi], gi
            if best_gi >= 0:
                matched_gt.add(best_gi)
                tp[pi, ti] = True
    return tp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Size-stratified mAP evaluation.")
    p.add_argument("--weights", required=True, help="Path to model checkpoint.")
    p.add_argument("--data", required=True, help="Dataset YAML file.")
    p.add_argument("--split", default="test", choices=["train", "val", "test"],
                   help="Dataset split to evaluate (default: test).")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001,
                   help="Detection confidence threshold (default 0.001 for AP eval).")
    p.add_argument("--iou", type=float, default=0.6,
                   help="NMS IoU threshold (default 0.6).")
    p.add_argument("--device", default="", help="Device: '', 'cpu', '0', etc.")
    p.add_argument("--out", default=None, help="Optional path to save JSON results.")
    return p.parse_args()


def main():
    opt = parse_args()

    # Load dataset config
    with open(opt.data) as f:
        data_cfg = yaml.safe_load(f)
    dataset_root = Path(data_cfg.get("path", "."))
    split_rel = data_cfg.get(opt.split, f"images/{opt.split}")
    img_dir = dataset_root / split_rel
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    class_names = data_cfg.get("names", {})  # {int: str}

    ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    img_files = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in ext)
    label_paths = [Path(lp) for lp in img2label_paths([str(p) for p in img_files])]
    print(f"Dataset : {img_dir}  ({len(img_files)} images, {len(class_names)} classes)")

    # Load model
    model = EDNet(str(opt.weights))

    # Accumulators: bucket → lists of per-image arrays
    bucket_names = ["all"] + [name for name, _, _ in SIZE_BINS]
    acc = {
        name: {"tp": [], "conf": [], "pred_cls": [], "gt_cls": []}
        for name in bucket_names
    }

    for img_path, lbl_path in tqdm(zip(img_files, label_paths), total=len(img_files),
                                   desc="Evaluating"):
        results = model.predict(
            str(img_path), conf=opt.conf, iou=opt.iou,
            imgsz=opt.imgsz, verbose=False, device=opt.device,
        )
        if not results or results[0].boxes is None:
            continue

        img_h, img_w = results[0].orig_shape
        preds = results[0].boxes.data.cpu().numpy()  # (N, 6): x1 y1 x2 y2 conf cls

        pred_boxes = preds[:, :4]
        pred_conf  = preds[:, 4]
        pred_cls   = preds[:, 5].astype(int)

        gt = load_gt_xyxy(lbl_path, img_w, img_h)  # (M, 5): cls x1 y1 x2 y2
        gt_cls_all   = gt[:, 0].astype(int)   if len(gt) else np.array([], dtype=int)
        gt_boxes_all = gt[:, 1:]              if len(gt) else np.zeros((0, 4), dtype=np.float32)
        gt_areas     = ((gt_boxes_all[:, 2] - gt_boxes_all[:, 0]) *
                        (gt_boxes_all[:, 3] - gt_boxes_all[:, 1])) if len(gt) else np.array([])

        # ── "all" bucket (no size filter) ────────────────────────────────────
        tp = match_to_gt(pred_boxes, pred_cls, pred_conf, gt_boxes_all, gt_cls_all)
        acc["all"]["tp"].append(tp)
        acc["all"]["conf"].append(pred_conf)
        acc["all"]["pred_cls"].append(pred_cls)
        acc["all"]["gt_cls"].append(gt_cls_all)

        # ── Size-stratified buckets ───────────────────────────────────────────
        for name, lo, hi in SIZE_BINS:
            if len(gt_areas) > 0:
                mask = (gt_areas >= lo) & (gt_areas < hi)
                gt_b = gt_boxes_all[mask]
                gc_b = gt_cls_all[mask]
            else:
                gt_b = np.zeros((0, 4), dtype=np.float32)
                gc_b = np.array([], dtype=int)

            # All predictions are candidates; TP only if matched to a bucket GT
            tp_b = match_to_gt(pred_boxes, pred_cls, pred_conf, gt_b, gc_b)
            acc[name]["tp"].append(tp_b)
            acc[name]["conf"].append(pred_conf)
            acc[name]["pred_cls"].append(pred_cls)
            acc[name]["gt_cls"].append(gc_b)

    # ---------------------------------------------------------------------------
    # Compute AP per bucket
    # ---------------------------------------------------------------------------
    output = {}

    header = f"\n{'Bucket':<12} {'Class':<20} {'AP50':>8} {'AP50-95':>10} {'N_GT':>8}"
    print(header)
    print("-" * len(header))

    for bucket_name in bucket_names:
        a = acc[bucket_name]
        if not a["tp"]:
            continue

        tp_arr   = np.concatenate(a["tp"],       axis=0)  # (N_total, 10)
        conf_arr = np.concatenate(a["conf"],      axis=0)  # (N_total,)
        cls_arr  = np.concatenate(a["pred_cls"],  axis=0)  # (N_total,)
        gt_arr   = np.concatenate(a["gt_cls"],    axis=0)  # (M_total,)

        if len(gt_arr) == 0:
            continue

        _, _, _, _, _, ap, unique_cls, *_ = ap_per_class(
            tp_arr, conf_arr, cls_arr, gt_arr
        )
        # ap: (nc, 10)
        ap50   = ap[:, 0]
        ap5095 = ap.mean(axis=1)

        n_gt_per_cls = {int(c): int((gt_arr == c).sum()) for c in unique_cls}
        total_gt = len(gt_arr)

        output[bucket_name] = {
            "mAP50":    float(ap50.mean()),
            "mAP50-95": float(ap5095.mean()),
            "n_gt":     total_gt,
            "per_class": {},
        }

        for i, cid in enumerate(unique_cls):
            cname = class_names.get(int(cid), str(cid))
            n = n_gt_per_cls.get(int(cid), 0)
            output[bucket_name]["per_class"][cname] = {
                "AP50":    float(ap50[i]),
                "AP50-95": float(ap5095[i]),
                "n_gt":    n,
            }
            print(f"{bucket_name:<12} {cname:<20} {ap50[i]:>8.3f} {ap5095[i]:>10.3f} {n:>8}")

        map50   = ap50.mean()
        map5095 = ap5095.mean()
        print(f"{bucket_name:<12} {'mAP (mean)':<20} {map50:>8.3f} {map5095:>10.3f} {total_gt:>8}")
        print()

    if opt.out:
        out_path = Path(opt.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
