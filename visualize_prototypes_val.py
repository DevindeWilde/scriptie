"""Visualize prototype slots by finding their nearest val-set detections.

For each (class, FPN level, slot) prototype in a saved buffer, this script:
  1. Runs inference on the validation set with feature tapping
  2. Extracts pre-logit embeddings at each detection's grid cell
  3. Finds the val detection with highest cosine similarity to each prototype
  4. Saves a cropped+resized patch around the matched bbox (thesis-ready)

No augmentation issues — val uses only letterbox/resize, and bboxes are
mapped back to original image space with ops.scale_boxes().

Usage:
    python scripts/visualize_prototypes_val.py \
        --weights runs/dota_saab/stage3-saab/weights/best.pt \
        --buffer  runs/dota_saab/stage3-saab/replay/buffer.pt \
        --data    yaml/DOTA-shared-stage3.yaml \
        --outdir  prototype_val_matches \
        --names   small-vehicle,large-vehicle,plane,helicopter,ship
"""

from __future__ import annotations

import argparse
import heapq
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from ednet import EDNet
from ednet.engine.replay import DetectionPreLogitTapper
from ednet.data.build import build_dataloader, build_yolo_dataset
from ednet.utils import ops, yaml_load
from ednet.utils.torch_utils import de_parallel


def load_prototypes(buffer_path: str) -> dict[tuple[str, int, int], torch.Tensor]:
    """Load prototype vectors from buffer.pt.

    Returns dict mapping (level, cls_id, slot_idx) → normalized vector.
    """
    state = torch.load(buffer_path, map_location="cpu")
    storage = state.get("storage", {})
    protos = {}
    for level, class_map in storage.items():
        for cls_id_str, entry in class_map.items():
            cls_id = int(cls_id_str)
            for slot_idx, slot in enumerate(entry.get("fine", [])):
                vec = slot.get("prototype")
                if vec is None:
                    continue
                vec = vec.float()
                vec = F.normalize(vec, dim=0, eps=1e-6)
                protos[(level, cls_id, slot_idx)] = vec
    return protos


def resolve_level_indices(detect_module, level_strides: dict[str, int]) -> dict[str, int]:
    """Map level names to cv3 branch indices via stride matching."""
    stride_tensor = getattr(detect_module, "stride", torch.tensor([]))
    stride_list = [int(s) for s in stride_tensor.tolist()]
    mapping = {}
    for level, stride in level_strides.items():
        try:
            mapping[level] = stride_list.index(stride)
        except ValueError:
            continue
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Visualize prototypes via val-set matching")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights (.pt)")
    parser.add_argument("--buffer", type=str, required=True, help="Path to buffer.pt")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset YAML")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--outdir", type=str, default="prototype_val_matches", help="Output directory")
    parser.add_argument("--names", type=str, default=None,
                        help="Comma-separated class names (e.g. small-vehicle,large-vehicle,...)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for NMS")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="", help="Device (e.g. 'cuda:0' or 'cpu')")
    parser.add_argument("--crop-size", type=int, default=224, help="Output crop resolution (square)")
    args = parser.parse_args()

    # --- Device ---
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # --- Load model ---
    print(f"Loading model from {args.weights}")
    yolo = EDNet(args.weights)
    model = yolo.model.to(device).eval()

    # --- Load prototypes ---
    print(f"Loading prototypes from {args.buffer}")
    protos = load_prototypes(args.buffer)
    if not protos:
        print("ERROR: No prototypes found in buffer")
        return
    # Collect which levels and classes are present
    levels_in_buffer = sorted(set(k[0] for k in protos))
    classes_in_buffer = sorted(set(k[1] for k in protos))
    print(f"  Levels: {levels_in_buffer}")
    print(f"  Classes: {classes_in_buffer}")
    print(f"  Total prototype slots: {len(protos)}")

    # --- Class name map ---
    name_map: dict[int, str] = {}
    if args.names:
        for i, name in enumerate(args.names.split(",")):
            name_map[i] = name.strip()

    # --- Set up feature tapper ---
    detect_module = de_parallel(model).model[-1]
    level_strides = {"P2": 4, "P3": 8, "P4": 16, "P5": 32}
    level_indices = resolve_level_indices(detect_module, level_strides)
    # Only tap levels that exist in the buffer
    level_indices = {k: v for k, v in level_indices.items() if k in levels_in_buffer}
    print(f"  Tapping levels: {level_indices}")

    tapper = DetectionPreLogitTapper(
        detect_module,
        level_to_indices=level_indices,
        detach=True,
        auto_activate=True,
        branch_attr="cv3",
    )

    # --- Build val dataloader ---
    data_cfg = yaml_load(args.data)
    dataset_root = Path(data_cfg.get("path", "."))
    val_rel = data_cfg.get(args.split, data_cfg.get("val", data_cfg.get("test", "")))
    val_path = str(dataset_root / val_rel)
    print(f"Loading validation data from: {val_path}")
    stride = max(int(s) for s in detect_module.stride.tolist())
    cfg = SimpleNamespace(
        imgsz=args.imgsz, rect=True, cache=False, single_cls=False,
        task="detect", classes=None, fraction=1.0,
        mask_ratio=4, overlap_mask=True, bgr=0.0,
    )
    dataset = build_yolo_dataset(cfg, img_path=val_path, batch=args.batch,
                                 data=data_cfg, mode="val", stride=stride)
    dataloader = build_dataloader(dataset, batch=args.batch, workers=4, shuffle=False)

    # --- Candidate matches per prototype slot (top-K for uniqueness dedup) ---
    MAX_CANDS = 8  # keep top-K per slot so greedy dedup has fallbacks
    candidates: dict[tuple, list] = {key: [] for key in protos}
    _counter = 0   # tiebreaker for heap ordering

    # Move prototype vectors to device
    proto_vecs = {k: v.to(device) for k, v in protos.items()}

    # --- Process val set ---
    print(f"\nProcessing {len(dataloader)} batches...")
    for batch_i, batch_data in enumerate(dataloader):
        imgs = batch_data["img"].to(device).float() / 255.0
        im_files = batch_data.get("im_file", [])
        ori_shapes = batch_data.get("ori_shape", [])
        ratio_pads = batch_data.get("ratio_pad", [])

        # Forward pass (features captured by tapper)
        with torch.no_grad():
            preds = model(imgs)

        # Get tapped features
        features = tapper.pop()

        # Get detections via NMS
        if isinstance(preds, (tuple, list)):
            raw_preds = preds[0]  # (B, num_dets, 4+nc) post-processed
        else:
            raw_preds = preds

        # raw_preds from the model in eval mode is already post-NMS
        # Each element: (num_dets, 6) = [x1, y1, x2, y2, conf, cls]
        # For batch processing, it's a list of tensors per image
        if isinstance(raw_preds, torch.Tensor):
            # Apply NMS if needed
            nms_preds = ops.non_max_suppression(
                raw_preds, conf_thres=args.conf, iou_thres=args.iou,
            )
        elif isinstance(raw_preds, list):
            nms_preds = raw_preds
        else:
            nms_preds = [raw_preds]

        imgsz_h, imgsz_w = imgs.shape[2], imgs.shape[3]

        for b in range(imgs.shape[0]):
            dets = nms_preds[b] if b < len(nms_preds) else torch.empty(0, 6)
            if dets is None or len(dets) == 0:
                continue

            im_file = im_files[b] if b < len(im_files) else ""
            ori_shape = ori_shapes[b] if b < len(ori_shapes) else (imgsz_h, imgsz_w)
            rp = ratio_pads[b] if b < len(ratio_pads) else None

            # Scale detections to original image space
            dets_orig = dets.clone()
            dets_orig[:, :4] = ops.scale_boxes(
                (imgsz_h, imgsz_w), dets_orig[:, :4].clone(), ori_shape, ratio_pad=rp
            )

            for det_idx in range(len(dets)):
                det = dets[det_idx]  # [x1, y1, x2, y2, conf, cls] in augmented space
                cls_id = int(det[5].item())
                conf = float(det[4].item())

                if cls_id not in classes_in_buffer:
                    continue

                # Detection center in augmented pixel space
                cx = (det[0] + det[2]) / 2
                cy = (det[1] + det[3]) / 2

                # Extract embedding at each tapped level
                for level_name, feat in features.items():
                    if feat.shape[0] <= b:
                        continue
                    stride = level_strides.get(level_name, 8)
                    feat_h, feat_w = feat.shape[2], feat.shape[3]
                    gx = min(int(cx / stride), feat_w - 1)
                    gy = min(int(cy / stride), feat_h - 1)
                    gx = max(0, gx)
                    gy = max(0, gy)

                    embedding = feat[b, :, gy, gx]
                    embedding = F.normalize(embedding, dim=0, eps=1e-6)

                    # Check against all matching prototype slots
                    for slot_idx in range(4):
                        key = (level_name, cls_id, slot_idx)
                        if key not in proto_vecs:
                            continue
                        sim = float(torch.dot(embedding, proto_vecs[key]).item())
                        cands = candidates[key]
                        if len(cands) < MAX_CANDS or sim > cands[0][0]:
                            det_orig = dets_orig[det_idx]
                            entry = (sim, _counter, {
                                "im_file": im_file,
                                "det_idx": det_idx,
                                "bbox_xyxy_orig": det_orig[:4].cpu().tolist(),
                                "ori_shape": tuple(int(x) for x in ori_shape),
                                "conf": conf,
                            })
                            _counter += 1
                            if len(cands) < MAX_CANDS:
                                heapq.heappush(cands, entry)
                            else:
                                heapq.heapreplace(cands, entry)

        if (batch_i + 1) % 10 == 0:
            print(f"  Processed {batch_i + 1}/{len(dataloader)} batches")

    # --- Greedy dedup: each detection assigned to at most one slot ---
    all_cands = []
    for key, heap in candidates.items():
        for sim, _, match in heap:
            all_cands.append((sim, key, match))
    all_cands.sort(key=lambda x: x[0], reverse=True)  # highest sim first

    best: dict[tuple, dict] = {}
    used_dets: set[tuple] = set()
    for sim, key, match in all_cands:
        if key in best:
            continue  # slot already assigned
        det_id = (match["im_file"], match["det_idx"])
        if det_id in used_dets:
            continue  # detection already claimed by another slot
        used_dets.add(det_id)
        best[key] = {**match, "sim": sim}

    # Fill unmatched slots
    for key in protos:
        if key not in best:
            best[key] = {"sim": -1.0}

    n_unique = len(used_dets)
    n_slots = sum(1 for v in best.values() if v["sim"] >= 0)
    print(f"\nAssigned {n_slots} slots to {n_unique} unique detections")

    # --- Save results ---
    outdir = Path(args.outdir)
    total = 0
    print(f"\n{'Level':<6} {'Cls':<4} {'Slot':<5} {'CosSim':<8} {'Conf':<6} {'Image'}")
    print("-" * 80)

    for (level, cls_id, slot_idx), match in sorted(best.items()):
        cls_name = name_map.get(cls_id, f"class{cls_id}")
        if match["sim"] < 0:
            print(f"{level:<6} {cls_id:<4} {slot_idx:<5} {'--':<8} {'--':<6} no match")
            continue

        im_file = match["im_file"]
        bbox = match["bbox_xyxy_orig"]
        sim = match["sim"]
        conf = match["conf"]
        im_path = Path(im_file)

        print(f"{level:<6} {cls_id:<4} {slot_idx:<5} {sim:<8.3f} {conf:<6.2f} {im_path.name}")

        if not im_path.exists():
            print(f"  WARNING: image not found: {im_path}")
            continue

        # Load original image and crop around matched bbox
        img = Image.open(im_path).convert("RGB")
        img_w, img_h = img.size
        x1, y1, x2, y2 = bbox
        bbox_w, bbox_h = x2 - x1, y2 - y1
        cx_box, cy_box = (x1 + x2) / 2, (y1 + y2) / 2

        # Context padding: 2× bbox size on each side, minimum 64px
        pad = max(2 * max(bbox_w, bbox_h), 64)
        crop_x1 = max(0, int(cx_box - pad))
        crop_y1 = max(0, int(cy_box - pad))
        crop_x2 = min(img_w, int(cx_box + pad))
        crop_y2 = min(img_h, int(cy_box + pad))
        crop = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        # Resize to consistent output resolution
        cw, ch = crop.size
        out_sz = args.crop_size
        crop = crop.resize((out_sz, out_sz), Image.LANCZOS)
        sx, sy = out_sz / cw, out_sz / ch

        # Draw bbox in resized crop space (green outline, no text)
        bx1 = (x1 - crop_x1) * sx
        by1 = (y1 - crop_y1) * sy
        bx2 = (x2 - crop_x1) * sx
        by2 = (y2 - crop_y1) * sy
        draw = ImageDraw.Draw(crop)
        draw.rectangle([bx1, by1, bx2, by2], outline=(0, 255, 0), width=3)

        # Save
        save_dir = outdir / cls_name / level
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"slot{slot_idx}_{im_path.stem}_sim{sim:.3f}.png"
        crop.save(save_path)
        total += 1

    print("-" * 80)
    print(f"Saved {total} prototype match images to {outdir}/")

    # Cleanup
    tapper.deactivate()


if __name__ == "__main__":
    main()
