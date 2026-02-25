import contextlib
import math
import random
from collections import defaultdict
from copy import copy, deepcopy
from pathlib import Path
from typing import Optional
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ednet.data import build_dataloader, build_yolo_dataset
from ednet.engine.replay import (
    DetectionPreLogitTapper,
    FeatureTapConfig,
    TinyReplayBuffer,
    TinyReplayItem,
    build_replay_batch,
)
from ednet.engine.trainer import BaseTrainer
from ednet.models import yolo
from ednet.nn.tasks import DetectionModel
from ednet.utils import LOGGER, RANK
from ednet.utils.tal import dist2bbox, make_anchors
from ednet.utils.plotting import plot_images, plot_labels, plot_results
from ednet.utils.torch_utils import de_parallel, torch_distributed_zero_first


class DetectionTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    Example:
        ```python
        from ednet.models.yolo.detect import DetectionTrainer

        args = dict(model='yolov8n.pt', data='coco8.yaml', epochs=3)
        trainer = DetectionTrainer(overrides=args)
        trainer.train()
        ```
    """

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Construct and return dataloader."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        if self.memory_loader:
            memory_batch = self._next_memory_batch()
            if memory_batch:
                batch = self._merge_memory_batch(batch, memory_batch)
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(self.args.imgsz * 0.5, self.args.imgsz * 1.5 + self.stride)
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        self.active_class_ids = None
        self.prev_class_ids = None
        stage_cfg = getattr(self.args, "stage", None)
        if isinstance(stage_cfg, dict):
            raw_ids = stage_cfg.get("active_classes")
            if raw_ids is not None:
                if isinstance(raw_ids, str):
                    raw_ids = raw_ids.strip("[]")
                    raw_ids = [] if raw_ids == "" else raw_ids.split(",")
                self.active_class_ids = tuple(sorted(int(x) for x in raw_ids))
                LOGGER.info(f"Active classes for this stage: {self.active_class_ids}")
            prev_ids = stage_cfg.get("prev_classes")
            if prev_ids is not None:
                if isinstance(prev_ids, str):
                    prev_ids = prev_ids.strip("[]")
                    prev_ids = [] if prev_ids == "" else prev_ids.split(",")
                self.prev_class_ids = tuple(sorted(int(x) for x in prev_ids))
                LOGGER.info(f"Replay/previous classes for this stage: {self.prev_class_ids}")
            # Pseudo-labels provide explicit supervision for old classes, so extend active_class_ids
            # to include prev_class_ids — otherwise the loss would silently ignore old class annotations.
            if stage_cfg.get("pseudo_labels_dir") and self.prev_class_ids and self.active_class_ids is not None:
                combined = tuple(sorted(set(self.active_class_ids) | set(self.prev_class_ids)))
                self.active_class_ids = combined
                LOGGER.info(f"Pseudo-labels active: extended active_classes → {self.active_class_ids}")
        if weights:
            model.load(weights)
        model.active_class_ids = self.active_class_ids
        model.prev_class_ids = self.prev_class_ids
        replay_args = getattr(self.args, "replay", None)
        self.replay_enabled = bool(isinstance(replay_args, dict) and replay_args.get("enable"))
        # Old-class GT annotations must reach the YOLO detection loss (cls/box/dfl); without
        # this extension, active_classes filtering would zero-gradient those heads on images
        # where old-class objects are present (baked-in replay labels or injected memory).
        # Note: prototype consistency (_record_prev_replay_cells) filters by class ID and is
        # unaffected by this extension.
        if self.prev_class_ids and self.active_class_ids is not None:
            combined = tuple(sorted(set(self.active_class_ids) | set(self.prev_class_ids)))
            if combined != self.active_class_ids:
                self.active_class_ids = combined
                model.active_class_ids = combined
                LOGGER.info(f"Replay active: extended active_classes → {self.active_class_ids}")
        self.feature_tapper = None
        self.replay_teacher_buffer = None
        self.replay_student_buffer = None
        self.replay_strides = {}
        self.replay_levels = []
        self.replay_samples_per_class = 0
        self.replay_loss_weight = 1.0
        self.replay_max_edge = 32.0
        self.replay_scale_weight = "uniform"
        self.replay_debug = False
        self.replay_save_path: Optional[Path] = None
        self.replay_init_buffer: Optional[Path] = None
        self.replay_capacity_growth = 1.5
        self.replay_memory_dir: Optional[Path] = None
        self.replay_memory_ratio = 0.0
        self.memory_batch_size = 0
        self.memory_loader = None
        self._memory_iter = None
        self._feature_tapper_needs_activation = False
        self.replay_slot_hits = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self._replay_fine_wins = 0    # embeddings whose nearest proto was a fine slot (not coarse)
        self._replay_total_queries = 0  # total embeddings processed in loss
        # Box prototype replay defaults (populated inside if self.replay_enabled below)
        self.replay_box_enabled = False
        self.box_feature_tapper = None
        self.replay_box_teacher_buffer = None
        self.replay_box_student_buffer = None
        self.replay_box_loss_weight = 1.0
        self.replay_box_save_path: Optional[Path] = None
        self._box_feature_tapper_needs_activation = False
        if self.replay_enabled:
            tap_layers = replay_args.get("tap_layers") or {}
            layer_map = {k: int(v) for k, v in tap_layers.items()} if tap_layers else FeatureTapConfig().layers.copy()
            default_stride_map = {"P2": 4, "P3": 8, "P4": 16, "P5": 32, "P6": 64}
            stride_overrides = replay_args.get("strides") or {}
            self.replay_strides = {
                level: int(stride_overrides.get(level, default_stride_map.get(level, 8)))
                for level in layer_map.keys()
            }
            self.replay_levels = list(layer_map.keys())
            detect_module = None
            with contextlib.suppress(AttributeError, IndexError):
                detect_module = de_parallel(model).model[-1]
            level_indices = self._resolve_detect_level_indices(detect_module, layer_map.keys()) if detect_module else {}
            if detect_module is not None and level_indices:
                self.feature_tapper = DetectionPreLogitTapper(
                    detect_module,
                    level_to_indices=level_indices,
                    detach=False,
                    auto_activate=False,
                )
                self._feature_tapper_needs_activation = True
                LOGGER.info(
                    "Replay feature tapper initialized for detection head levels: %s",
                    list(level_indices.keys()),
                )
            else:
                LOGGER.warning("Replay feature tapper initialization skipped; detection head mapping unavailable.")
            capacity = int(replay_args.get("buffer_per_class", 64))
            self.replay_capacity_growth = float(replay_args.get("carryover_growth", 1.5))
            dtype = getattr(torch, replay_args.get("dtype", "float16"))
            proto_args = replay_args.get("prototypes") or {}
            proto_kwargs = {
                "num_fine": int(proto_args.get("num_fine", 0)),
                "use_coarse": bool(proto_args.get("use_coarse", True)),
                "ema_alpha": float(proto_args.get("ema_alpha", 0.05)),
                "init_sim_thresh": float(proto_args.get("init_sim_thresh", 0.9)),
                "gate_min_cos": float(proto_args.get("gate_min_cos", 0.0)),
                "init_strategy": str(proto_args.get("init_strategy", "first_k")),
                "weight_by_count": bool(proto_args.get("weight_by_count", False)),
            }
            self.replay_teacher_buffer = TinyReplayBuffer(
                per_class_capacity=capacity,
                dtype=dtype,
                device="cpu",
                carryover_growth=self.replay_capacity_growth,
                **proto_kwargs,
            )
            self.replay_student_buffer = TinyReplayBuffer(
                per_class_capacity=capacity,
                dtype=dtype,
                device="cpu",
                carryover_growth=self.replay_capacity_growth,
                **proto_kwargs,
            )
            self.replay_samples_per_class = max(1, int(replay_args.get("sample_per_batch", 16)))
            self.replay_loss_weight = float(replay_args.get("loss_weight", 1.0))
            self.replay_max_edge = float(replay_args.get("tiny_max_pixels", 32))
            self.replay_student_update_freq = max(1, int(replay_args.get("student_update_freq", 1)))
            self._student_update_counter = 0
            self.replay_scale_weight = replay_args.get("scale_weighting", "uniform")
            self.replay_debug = bool(replay_args.get("debug", False))
            store_dir = replay_args.get("store_dir", "replay")
            buffer_file = replay_args.get("buffer_file", "buffer.pt")
            if store_dir:
                store_root = Path(self.save_dir) / store_dir
                store_root.mkdir(parents=True, exist_ok=True)
                fname = buffer_file or "buffer.pt"
                self.replay_save_path = store_root / fname
            init_buffer = replay_args.get("init_buffer")
            if init_buffer:
                self.replay_init_buffer = Path(init_buffer)
                self._load_teacher_buffer(self.replay_init_buffer)
            memory_dir = replay_args.get("memory_dir")
            memory_ratio = float(replay_args.get("memory_ratio", 0.0) or 0.0)
            if memory_dir and memory_ratio > 0:
                self.replay_memory_dir = Path(memory_dir)
                self.replay_memory_ratio = max(0.0, min(memory_ratio, 1.0))
                self.memory_batch_size = max(1, int(self.args.batch * self.replay_memory_ratio))
            else:
                self.replay_memory_dir = None
                self.replay_memory_ratio = 0.0
            # --- Box prototype replay (SA-AB box branch) ---
            box_args = (replay_args.get("box") or {}) if isinstance(replay_args, dict) else {}
            self.replay_box_enabled = bool(box_args.get("enable", False))
            if self.replay_box_enabled:
                self.replay_box_loss_weight = float(box_args.get("loss_weight", 1.0))
                if detect_module is not None and level_indices:
                    self.box_feature_tapper = DetectionPreLogitTapper(
                        detect_module,
                        level_to_indices=level_indices,
                        detach=False,
                        auto_activate=False,
                        branch_attr="cv2",
                    )
                    self._box_feature_tapper_needs_activation = True
                    LOGGER.info("Box replay feature tapper initialized (cv2) for levels: %s", list(level_indices.keys()))
                else:
                    LOGGER.warning("Box replay tapper skipped; detection head mapping unavailable.")
                self.replay_box_teacher_buffer = TinyReplayBuffer(
                    per_class_capacity=capacity,
                    dtype=dtype,
                    device="cpu",
                    carryover_growth=self.replay_capacity_growth,
                    **proto_kwargs,
                )
                self.replay_box_student_buffer = TinyReplayBuffer(
                    per_class_capacity=capacity,
                    dtype=dtype,
                    device="cpu",
                    carryover_growth=self.replay_capacity_growth,
                    **proto_kwargs,
                )
                if store_dir:
                    box_buffer_file = box_args.get("buffer_file", "box_buffer.pt") or "box_buffer.pt"
                    self.replay_box_save_path = store_root / box_buffer_file
                box_init = box_args.get("init_buffer")
                if box_init:
                    self._load_box_teacher_buffer(Path(box_init))
                LOGGER.info(
                    "Box prototype replay enabled | loss_weight=%.2f",
                    self.replay_box_loss_weight,
                )
        kd_args = getattr(self.args, "kd", None)
        self.kd_enabled = bool(isinstance(kd_args, dict) and kd_args.get("enable"))
        self.kd_teacher = None
        self._kd_student_raw = None
        self._kd_hook_handle = None
        self._kd_hook_pending = False
        self._kd_dbg_left = 0
        kd_dict = kd_args if isinstance(kd_args, dict) else {}
        self.kd_lambda_cls = float(kd_dict.get("lambda_cls", 1.0))
        self.kd_lambda_bbox = float(kd_dict.get("lambda_bbox", 1.0))
        self.kd_top_k = int(kd_dict.get("top_k", 10))
        self._kd_dbg_left = int(kd_dict.get("debug_batches", 10))
        if self.kd_enabled and self.prev_class_ids and weights:
            teacher_source = kd_dict.get("teacher_weights", None)
            if teacher_source is None and isinstance(weights, (str, Path)):
                teacher_source = weights

            if teacher_source is not None:
                # Preferred path: build teacher from previous checkpoint file.
                teacher_source = Path(teacher_source)
                if not teacher_source.exists():
                    LOGGER.warning(f"KD teacher checkpoint not found: {teacher_source}. Falling back to in-memory copy.")
                else:
                    # Load checkpoint once — use it for both nc inference and weight extraction.
                    ckpt = torch.load(str(teacher_source), map_location="cpu", weights_only=False)
                    nc_prev = int(self.data["nc"])  # fallback
                    if isinstance(ckpt, dict):
                        nc_prev = int(ckpt.get("train_args", {}).get("nc", nc_prev))
                    nc_prev = int(kd_dict.get("teacher_nc", nc_prev))  # explicit override wins
                    teacher_module = (ckpt.get("ema") or ckpt.get("model")).float()
                    teacher = DetectionModel(cfg, nc=nc_prev, verbose=False)
                    # BaseModel.load() expects an nn.Module or {"model": module} dict, not a path.
                    teacher.load(teacher_module)
                    for p in teacher.parameters():
                        p.requires_grad_(False)
                    teacher.eval()
                    self.kd_teacher = teacher
                    LOGGER.info(
                        "KD teacher initialized from checkpoint | nc_prev=%d | nc_curr=%d | prev_classes=%s",
                        nc_prev,
                        self.data["nc"],
                        self.prev_class_ids,
                    )

            if self.kd_teacher is None and isinstance(weights, nn.Module):
                # Fallback path: clone in-memory model (safe when checkpoint path is unavailable).
                # With growing nc this may not preserve old-head semantics, so checkpoint path is preferred.
                self.kd_teacher = deepcopy(weights)
                for p in self.kd_teacher.parameters():
                    p.requires_grad_(False)
                self.kd_teacher.eval()
                LOGGER.warning(
                    "KD teacher initialized from in-memory model; provide kd.teacher_weights for strict growing-nc KD."
                )

            if self.kd_teacher is not None:
                # Delay hook registration until after BaseTrainer._setup_train builds EMA.
                # Otherwise deepcopy(model) inside EMA includes hook state and can fail.
                self._kd_hook_pending = True
                LOGGER.info(
                    "KD active | prev_classes=%s | top_k=%d | lambda_cls=%.2f | lambda_bbox=%.2f",
                    self.prev_class_ids,
                    self.kd_top_k,
                    self.kd_lambda_cls,
                    self.kd_lambda_bbox,
                )
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss")
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss")
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        if self._kd_hook_pending and self._kd_hook_handle is None:
            self._attach_kd_student_hook(self.model)
            self._kd_hook_pending = False
        if self.replay_enabled and getattr(self, "feature_tapper", None) and self._feature_tapper_needs_activation:
            self.feature_tapper.activate()
            self._feature_tapper_needs_activation = False
        if getattr(self, "replay_box_enabled", False) and getattr(self, "box_feature_tapper", None) and self._box_feature_tapper_needs_activation:
            self.box_feature_tapper.activate()
            self._box_feature_tapper_needs_activation = False
        if self.replay_enabled:
            self._ensure_replay_tap_config()
        if getattr(self, "kd_teacher", None) is not None:
            self.kd_teacher.to(self.device)
        if self.replay_memory_dir and self.replay_memory_ratio > 0:
            self._build_memory_loader()
        else:
            self.memory_loader = None
            self._memory_iter = None

    def _build_memory_loader(self):
        images_dir = self.replay_memory_dir / "images" if self.replay_memory_dir else None
        if images_dir is None or not images_dir.exists():
            LOGGER.warning(f"Replay memory directory missing or invalid: {images_dir}")
            self.memory_loader = None
            self._memory_iter = None
            return
        batch_size = max(1, int(self.args.batch * self.replay_memory_ratio))
        self.memory_loader = self.get_dataloader(str(images_dir), batch_size=batch_size, rank=RANK, mode="train")
        self._memory_iter = iter(self.memory_loader)
        LOGGER.info(
            f"Memory replay enabled from {images_dir} with batch size {batch_size} "
            f"({self.replay_memory_ratio * 100:.1f}% of primary batch)."
        )

    def _next_memory_batch(self):
        if not self.memory_loader:
            return None
        try:
            return next(self._memory_iter)
        except StopIteration:
            self._memory_iter = iter(self.memory_loader)
            return next(self._memory_iter)

    def _ensure_tensor(self, value, reference):
        if torch.is_tensor(value):
            if torch.is_tensor(reference):
                return value.to(reference.device, dtype=reference.dtype)
            return value
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                if torch.is_tensor(reference):
                    return reference.new_empty((0, *reference.shape[1:]))
                return []
            if isinstance(value, (list, tuple)):
                if torch.is_tensor(value[0]):
                    stacked = torch.stack(value, dim=0)
                    if torch.is_tensor(reference):
                        return stacked.to(reference.device, dtype=reference.dtype)
                    return stacked
            if torch.is_tensor(reference):
                return torch.tensor(value, dtype=reference.dtype, device=reference.device)
            return torch.tensor(value)
        if torch.is_tensor(reference):
            return reference.new_tensor(value)
        return torch.tensor(value)

    def _merge_memory_batch(self, batch, memory_batch):
        if not memory_batch:
            return batch
        merged = batch
        main_imgs = merged["img"].shape[0]
        for key in ("img", "cls", "bboxes"):
            if key in merged and key in memory_batch:
                mem_val = self._ensure_tensor(memory_batch[key], merged[key])
                merged[key] = torch.cat((merged[key], mem_val), 0)
        if "batch_idx" in merged and "batch_idx" in memory_batch:
            mem_idx = memory_batch["batch_idx"]
            if isinstance(mem_idx, (list, tuple)):
                if len(mem_idx) == 0:
                    mem_idx = torch.empty_like(merged["batch_idx"][:0])
                else:
                    mem_idx = torch.tensor(mem_idx, device=merged["batch_idx"].device, dtype=merged["batch_idx"].dtype)
            if not torch.is_tensor(mem_idx):
                mem_idx = torch.tensor(mem_idx, device=merged["batch_idx"].device, dtype=merged["batch_idx"].dtype)
            merged["batch_idx"] = torch.cat((merged["batch_idx"], mem_idx + main_imgs), 0)
        if "im_file" in merged and "im_file" in memory_batch:
            merged["im_file"] = merged["im_file"] + memory_batch["im_file"]
        for key in ("ori_shape", "resized_shape", "ratio_pad"):
            if key in merged and key in memory_batch:
                if torch.is_tensor(merged[key]):
                    mem_val = self._ensure_tensor(memory_batch[key], merged[key])
                    merged[key] = torch.cat((merged[key], mem_val), 0)
                else:
                    mem_val = memory_batch[key]
                    if isinstance(mem_val, (list, tuple)):
                        merged[key] = list(merged[key]) + list(mem_val)
                    else:
                        merged[key] = list(merged[key]) + [mem_val]
        return merged

    def _resolve_detect_level_indices(self, detect_module, level_names):
        if detect_module is None or not hasattr(detect_module, "stride"):
            return {}
        stride_tensor = getattr(detect_module, "stride", torch.tensor([]))
        if hasattr(stride_tensor, "tolist"):
            stride_list = [int(s) for s in stride_tensor.tolist()]
        else:
            stride_list = [int(s) for s in stride_tensor]
        mapping = {}
        for level in level_names:
            stride = int(self.replay_strides.get(level, 0))
            if stride <= 0:
                continue
            with contextlib.suppress(ValueError):
                mapping[level] = stride_list.index(stride)
        return mapping

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def _kd_student_hook(self, module, inp, out):
        """Forward hook on the detect head; captures raw training-mode outputs for KD."""
        self._kd_student_raw = out  # {"one2many": [...], "one2one": [...]}

    def _attach_kd_student_hook(self, model):
        """Attach the student detect-head forward hook used by KD."""
        if getattr(self, "_kd_hook_handle", None) is not None:
            self._kd_hook_handle.remove()
            self._kd_hook_handle = None
        detect_head = de_parallel(model).model[-1]
        self._kd_hook_handle = detect_head.register_forward_hook(self._kd_student_hook)

    def _before_checkpoint(self):
        if getattr(self, "feature_tapper", None):
            self.feature_tapper.deactivate()
        if getattr(self, "box_feature_tapper", None):
            self.box_feature_tapper.deactivate()
        if getattr(self, "_kd_hook_handle", None) is not None:
            self._kd_hook_handle.remove()
            self._kd_hook_handle = None
        return {}

    def _after_checkpoint(self, ctx):
        if getattr(self, "feature_tapper", None):
            self.feature_tapper.activate()
        if getattr(self, "box_feature_tapper", None):
            self.box_feature_tapper.activate()
        if getattr(self, "kd_teacher", None) is not None and self._kd_hook_handle is None:
            self._attach_kd_student_hook(self.model)
        if self.replay_enabled:
            self._maybe_save_replay_buffer(ctx)
        super()._after_checkpoint(ctx)

    def compute_auxiliary_loss(self, batch):
        """Compute replay and/or knowledge-distillation auxiliary losses."""
        kd_active = getattr(self, "kd_enabled", False) and self.kd_teacher is not None
        if not self.replay_enabled and not kd_active:
            return None
        aux_loss = None

        # --- replay path ---
        if self.replay_enabled:
            teacher_ready = (
                self.feature_tapper is not None
                and self.replay_teacher_buffer is not None
                and len(self.replay_teacher_buffer) > 0
            )
            criterion = self._ensure_replay_tap_config()
            # Snapshot index data before _gather_embeddings clears the criterion attributes.
            # These refs remain valid even after the criterion attributes are set to None.
            replay_index_raw = getattr(criterion, "last_replay_cells", None)
            student_index_raw = getattr(criterion, "last_positive_cells", None)
            features = self.feature_tapper.pop() if self.feature_tapper is not None else {}
            box_features = self.box_feature_tapper.pop() if getattr(self, "box_feature_tapper", None) else {}
            if features:
                if teacher_ready:
                    replay_items = self._gather_embeddings(features, attr="last_replay_cells", criterion=criterion, batch=batch)
                    if replay_items:
                        grouped = self._group_embeddings(replay_items)
                        self._log_embedding_stats(grouped)
                        replay_batch = build_replay_batch(
                            self.replay_teacher_buffer,
                            per_class=self.replay_samples_per_class,
                            balance_levels=self.replay_levels,
                            device=self.device,
                        )
                        replay_loss = self._compute_replay_consistency(grouped, replay_batch)
                        if replay_loss is not None:
                            aux_loss = replay_loss * self.replay_loss_weight
                if self.replay_student_buffer is not None:
                    self.replay_student_buffer.current_epoch = self.epoch
                    self._student_update_counter += 1
                    if self._student_update_counter % self.replay_student_update_freq == 0:
                        student_items = self._gather_embeddings(features, attr="last_positive_cells", criterion=criterion, batch=batch)
                        for item in student_items:
                            self.replay_student_buffer.add(item)
                if self.replay_box_student_buffer is not None:
                    self.replay_box_student_buffer.current_epoch = self.epoch
            # --- Box prototype replay path (uses same grid-cell positions as cls replay) ---
            box_teacher_ready = (
                getattr(self, "replay_box_enabled", False)
                and getattr(self, "box_feature_tapper", None) is not None
                and self.replay_box_teacher_buffer is not None
                and len(self.replay_box_teacher_buffer) > 0
            )
            if box_features and getattr(self, "replay_box_enabled", False):
                if box_teacher_ready and replay_index_raw is not None:
                    box_replay_items = self._gather_embeddings_from_index(box_features, replay_index_raw, criterion)
                    if box_replay_items:
                        box_grouped = self._group_embeddings(box_replay_items)
                        box_replay_batch = build_replay_batch(
                            self.replay_box_teacher_buffer,
                            per_class=self.replay_samples_per_class,
                            balance_levels=self.replay_levels,
                            device=self.device,
                        )
                        box_replay_loss = self._compute_replay_consistency(
                            box_grouped, box_replay_batch, log_prefix="replay_box",
                            teacher_buffer=self.replay_box_teacher_buffer,
                        )
                        if box_replay_loss is not None:
                            box_aux = box_replay_loss * self.replay_box_loss_weight
                            aux_loss = box_aux if aux_loss is None else aux_loss + box_aux
                if (self.replay_box_student_buffer is not None
                        and student_index_raw is not None
                        and self._student_update_counter % self.replay_student_update_freq == 0):
                    box_student_items = self._gather_embeddings_from_index(
                        box_features, student_index_raw, criterion, detach=True
                    )
                    for item in box_student_items:
                        self.replay_box_student_buffer.add(item)

        # --- KD path ---
        if kd_active:
            kd_loss = self._compute_kd_loss(batch)
            if kd_loss is not None:
                if self._kd_dbg_left > 0:
                    LOGGER.info("[KD] aux kd_loss=%.6f", float(kd_loss.detach()))
                aux_loss = kd_loss if aux_loss is None else aux_loss + kd_loss

        return aux_loss

    def _compute_kd_loss(self, batch):
        """Compute ILOD/RILOD knowledge distillation loss (both heads)."""
        if self._kd_student_raw is None:
            return None

        if self._kd_dbg_left > 0:
            LOGGER.info(
                "[KD] start | prev=%s | top_k=%d | lambda_cls=%.3f | lambda_bbox=%.3f",
                self.prev_class_ids,
                self.kd_top_k,
                self.kd_lambda_cls,
                self.kd_lambda_bbox,
            )

        student_out = self._kd_student_raw  # captured by hook during main forward
        self._kd_student_raw = None         # clear for next step
        if self._kd_dbg_left > 0:
            LOGGER.info(
                "[KD] student_out type=%s keys=%s",
                type(student_out),
                list(student_out.keys()) if isinstance(student_out, dict) else None,
            )

        # Guard: only v10Detect (end2end) returns the required dict format
        if not isinstance(student_out, dict):
            return None

        imgs = batch["img"]  # (B, 3, H, W) already on device and normalized

        # Run teacher: backbone+neck stay in eval (frozen BN running stats).
        # Only set detect_t.training = True directly (not .train()) so that
        # forward_end2end returns the raw dict without recursively touching child BN.
        t_raw = {}
        detect_t = de_parallel(self.kd_teacher).model[-1]

        def _teacher_hook(module, inp, out):
            t_raw["out"] = out

        handle = detect_t.register_forward_hook(_teacher_hook)
        try:
            with torch.no_grad():
                self.kd_teacher.eval()
                detect_t.training = True       # flag only; child BN stays in eval
                self.kd_teacher.predict(imgs)  # triggers hook
        finally:
            detect_t.training = False          # restore eval
            handle.remove()

        if "out" not in t_raw or not isinstance(t_raw["out"], dict):
            return None

        teacher_out = t_raw["out"]
        if self._kd_dbg_left > 0:
            LOGGER.info(
                "[KD] teacher_out keys=%s | teacher.training=%s | detect_head.training=%s",
                list(teacher_out.keys()) if isinstance(teacher_out, dict) else None,
                self.kd_teacher.training,
                detect_t.training,
            )
        detect_s = de_parallel(self.model).model[-1]
        nc_curr = de_parallel(self.model).nc
        nc_prev = detect_t.nc  # detect_t = de_parallel(self.kd_teacher).model[-1], already computed above
        prev_ids = list(self.prev_class_ids)

        total_loss = None
        for head_key in ("one2many", "one2one"):
            head_loss = self._kd_head_loss(
                student_out[head_key], teacher_out[head_key],
                detect_s, nc_curr, nc_prev, prev_ids, imgs.device,
            )
            if head_loss is not None:
                total_loss = head_loss if total_loss is None else total_loss + head_loss
        if self._kd_dbg_left > 0:
            self._kd_dbg_left -= 1
        return total_loss

    def _kd_head_loss(self, s_feats, t_feats, detect, nc_curr, nc_prev, prev_ids, device):
        """KD loss for one head: L2 cls + Smooth L1 box on teacher top-k anchor positions.

        nc_curr: student head nc (may be larger than teacher for growing-nc scenarios)
        nc_prev: teacher head nc (always matches the saved checkpoint)
        """
        B = s_feats[0].shape[0]
        reg4 = detect.reg_max * 4  # 64

        # Flatten and cat across scales with correct nc per model
        s_cat = torch.cat([f.view(B, nc_curr + reg4, -1) for f in s_feats], dim=2)
        t_cat = torch.cat([f.view(B, nc_prev + reg4, -1) for f in t_feats], dim=2)

        s_box_raw, s_cls_raw = s_cat.split((reg4, nc_curr), dim=1)
        t_box_raw, t_cls_raw = t_cat.split((reg4, nc_prev), dim=1)

        # Anchors from student feature shapes (same grid as teacher — identical architecture)
        anchors, stride_t = (x.transpose(0, 1) for x in make_anchors(s_feats, detect.stride, 0.5))
        # anchors: (2, sum_HW), stride_t: (1, sum_HW)

        # Decode boxes via DFL (DFL weights are fixed/non-learnable, identical in both models)
        s_boxes = dist2bbox(detect.dfl(s_box_raw), anchors.unsqueeze(0), xywh=True, dim=1) * stride_t
        t_boxes = dist2bbox(detect.dfl(t_box_raw), anchors.unsqueeze(0), xywh=True, dim=1) * stride_t
        # Both: (B, 4, sum_HW)

        s_cls = s_cls_raw.sigmoid()  # (B, nc, sum_HW)
        t_cls = t_cls_raw.sigmoid()

        # Top-k anchor selection per image by teacher's max old-class confidence
        t_old_conf = t_cls[:, prev_ids, :].max(dim=1).values  # (B, sum_HW)

        cls_losses, box_losses = [], []
        k = min(self.kd_top_k, t_old_conf.shape[1])

        for b in range(B):
            topk_idx = t_old_conf[b].topk(k).indices  # (k,)

            # L2 on old-class sigmoid probabilities
            s_sel = s_cls[b][:, topk_idx][prev_ids, :]  # (|prev|, k)
            t_sel = t_cls[b][:, topk_idx][prev_ids, :]
            cls_losses.append(F.mse_loss(s_sel, t_sel))

            # Smooth L1 on decoded boxes
            s_bsel = s_boxes[b, :, topk_idx]  # (4, k)
            t_bsel = t_boxes[b, :, topk_idx]
            box_losses.append(F.smooth_l1_loss(s_bsel, t_bsel))

        if not cls_losses:
            return None

        cls_loss = torch.stack(cls_losses).mean()
        box_loss = torch.stack(box_losses).mean()
        total = self.kd_lambda_cls * cls_loss + self.kd_lambda_bbox * box_loss
        if self._kd_dbg_left > 0:
            LOGGER.info(
                "[KD] head loss | k=%d | cls=%.6f | box=%.6f | total=%.6f",
                k,
                float(cls_loss.detach()),
                float(box_loss.detach()),
                float(total.detach()),
            )
        return total

    def _group_embeddings(self, items):
        grouped = {}
        for item in items:
            level_map = grouped.setdefault(item.level, {})
            level_map.setdefault(item.cls, []).append(item)
        return grouped

    def _ensure_replay_tap_config(self):
        model_single = de_parallel(self.model)
        criterion = getattr(model_single, "criterion", None)
        if criterion is None and hasattr(model_single, "init_criterion"):
            criterion = model_single.init_criterion()
            model_single.criterion = criterion
        if (
            criterion is not None
            and hasattr(criterion, "set_replay_tap_config")
            and not getattr(criterion, "_replay_configured", False)
            and self.replay_levels
            and self.replay_strides
        ):
            criterion.set_replay_tap_config(self.replay_levels, self.replay_strides)
        return criterion

    def _gather_embeddings(self, features, attr="last_positive_cells", criterion=None, batch=None):
        """Collect TinyReplayItems from tapped features using stored cell indices."""
        if criterion is None:
            criterion = self._ensure_replay_tap_config()
        loss_module = criterion
        #print("loss_module:", loss_module)
        #print(loss_module.last_replay_cells)
        if loss_module is None:
            #print("No loss module available for gathering embeddings.")
            return []
        if hasattr(loss_module, "set_replay_tap_config") and not getattr(loss_module, "_replay_configured", False):
            if self.replay_levels and self.replay_strides:
                loss_module.set_replay_tap_config(self.replay_levels, self.replay_strides)
        pos = getattr(loss_module, attr, None)
        if not pos:
            #print(f"No replay tap data found for attribute '{attr}'.")
            return []
        indices = pos.get("indices")
        classes = pos.get("classes")
        max_edge = pos.get("max_edge")
        if indices is None or classes is None or max_edge is None:
            #print("No indices, classes, or max_edge found in replay tap data.")
            return []
        if indices.numel() == 0:
            #print("No replay embeddings to gather: indices is empty.")
            return []
        size_mask = max_edge <= self.replay_max_edge
        if not size_mask.any():
            if self.replay_debug and RANK in {-1, 0}:
                LOGGER.info(
                    "Replay tap skipped: max_edge min=%.2f max=%.2f threshold=%.2f",
                    float(max_edge.min()),
                    float(max_edge.max()),
                    float(self.replay_max_edge),
                )
            print("No replay embeddings to gather: all max_edge values exceed threshold.")
            return []
        if self.replay_debug:
            LOGGER.info(
                "Replay tap positives before filter=%d after=%d",
                int(size_mask.numel()),
                int(size_mask.sum()),
            )
        # Extract bboxes_norm and im_file lists for provenance metadata (replay cells only)
        bboxes_norm_all = pos.get("bboxes_norm")
        im_files = (batch or {}).get("im_file", [])

        indices = indices[size_mask]
        classes = classes[size_mask]
        max_edge = max_edge[size_mask]
        bboxes_norm_all = bboxes_norm_all[size_mask] if bboxes_norm_all is not None else None
        level_names = getattr(loss_module, "level_names", [])
        if attr == "last_replay_cells" and getattr(loss_module, "replay_level_order", None):
            level_names = loss_module.replay_level_order
        items = []
        unique_levels = indices[:, 0].unique()
        for level_idx in unique_levels:
            level_mask = indices[:, 0] == level_idx
            if not level_mask.any():
                continue
            name = level_names[level_idx] if level_idx < len(level_names) else str(int(level_idx))
            feat = features.get(name)
            if feat is None:
                continue
            sel_indices = indices[level_mask]
            sel_classes = classes[level_mask]
            sel_sizes = max_edge[level_mask]
            sel_bboxes = bboxes_norm_all[level_mask] if bboxes_norm_all is not None else None
            batch_idx = sel_indices[:, 1].long()
            gy = sel_indices[:, 2].long()
            gx = sel_indices[:, 3].long()
            vecs = feat[batch_idx, :, gy, gx]
            if attr == "last_positive_cells":
                # Batch GPU→CPU transfer for all per-item tensors; avoids N×3 CUDA syncs from
                # scalar .item() calls on GPU tensors in the inner loop below.
                vecs        = vecs.detach().cpu()
                sel_classes = sel_classes.cpu()
                sel_sizes   = sel_sizes.cpu()
                batch_idx   = batch_idx.cpu()
                if sel_bboxes is not None:
                    sel_bboxes = sel_bboxes.cpu()

            for i, (embedding, cls_id, size, bidx) in enumerate(
                zip(vecs, sel_classes, sel_sizes, batch_idx)
            ):
                emb = embedding  # already detached+cpu for student; GPU with grad for teacher
                meta: dict = {"max_edge": float(size.item())}
                b = int(bidx.item())
                if im_files and b < len(im_files):
                    meta["im_file"] = im_files[b]
                if sel_bboxes is not None:
                    bbox_norm = sel_bboxes[i]
                    batch_imgs = (batch or {}).get("img")
                    if batch_imgs is not None:
                        imgsz_h, imgsz_w = batch_imgs.shape[2], batch_imgs.shape[3]
                        cx, cy, bw, bh = [float(x.item()) if hasattr(x, "item") else float(x) for x in bbox_norm]
                        bx1 = int(round((cx - bw / 2) * imgsz_w))
                        by1 = int(round((cy - bh / 2) * imgsz_h))
                        bx2 = int(round((cx + bw / 2) * imgsz_w))
                        by2 = int(round((cy + bh / 2) * imgsz_h))
                        pad = max(20, int(max(bx2 - bx1, by2 - by1) * 1.5))
                        cx1 = max(0, bx1 - pad)
                        cy1 = max(0, by1 - pad)
                        cx2 = min(imgsz_w, bx2 + pad)
                        cy2 = min(imgsz_h, by2 + pad)
                        crop = (batch_imgs[b, :, cy1:cy2, cx1:cx2] * 255).byte().detach().cpu()
                        meta["source_crop"] = crop
                        meta["bbox_in_crop"] = [bx1 - cx1, by1 - cy1, bx2 - cx1, by2 - cy1]
                    else:
                        meta["bbox_xywh_norm"] = bbox_norm.tolist()
                items.append(
                    TinyReplayItem(
                        cls=int(cls_id.item()),
                        level=name,
                        embedding=emb,
                        metadata=meta,
                    )
                )
        if attr == "last_positive_cells":
            loss_module.last_positive_cells = None
        if attr == "last_replay_cells" and hasattr(loss_module, "last_replay_cells"):
            loss_module.last_replay_cells = None
        #print(f"Total TinyReplayItems gathered: {len(items)}")
        return items

    def _gather_embeddings_from_index(self, features, index_info, criterion, detach=False):
        """Extract TinyReplayItems using pre-captured index_info (does not clear criterion attrs).

        Used to gather box-branch embeddings at the same grid positions already captured
        for the classification branch, after criterion attributes have been cleared.
        """
        if not index_info or not features:
            return []
        indices = index_info.get("indices")
        classes = index_info.get("classes")
        max_edge = index_info.get("max_edge")
        if indices is None or classes is None or max_edge is None:
            return []
        if indices.numel() == 0:
            return []
        size_mask = max_edge <= self.replay_max_edge
        if not size_mask.any():
            return []
        indices = indices[size_mask]
        classes = classes[size_mask]
        max_edge = max_edge[size_mask]
        level_names = getattr(criterion, "level_names", [])
        replay_order = getattr(criterion, "replay_level_order", None)
        if replay_order:
            level_names = replay_order
        items = []
        unique_levels = indices[:, 0].unique()
        for level_idx in unique_levels:
            level_mask = indices[:, 0] == level_idx
            if not level_mask.any():
                continue
            name = level_names[level_idx] if level_idx < len(level_names) else str(int(level_idx))
            feat = features.get(name)
            if feat is None:
                continue
            sel_indices = indices[level_mask]
            sel_classes = classes[level_mask]
            sel_sizes = max_edge[level_mask]
            batch_idx = sel_indices[:, 1].long()
            gy = sel_indices[:, 2].long()
            gx = sel_indices[:, 3].long()
            vecs = feat[batch_idx, :, gy, gx]
            if detach:
                # Batch GPU→CPU transfer for all per-item tensors; avoids N×2 CUDA syncs from
                # scalar .item() calls on GPU tensors in the inner loop below.
                vecs        = vecs.detach().cpu()
                sel_classes = sel_classes.cpu()
                sel_sizes   = sel_sizes.cpu()
            for embedding, cls_id, size in zip(vecs, sel_classes, sel_sizes):
                emb = embedding  # already detached+cpu if detach=True
                items.append(
                    TinyReplayItem(
                        cls=int(cls_id.item()),
                        level=name,
                        embedding=emb,
                        metadata={"max_edge": float(size.item())},
                    )
                )
        return items

    def _log_embedding_stats(self, grouped):
        if not (self.replay_debug and RANK in {-1, 0}):
            return
        for level, class_map in grouped.items():
            all_items = [it.embedding for items in class_map.values() for it in items]
            if not all_items:
                continue
            tensors = torch.stack(all_items).to(self.device)
            norm = tensors.norm(dim=1)
            LOGGER.info(
                "Replay tap %s: count=%d mean_norm=%.4f max_norm=%.4f",
                level,
                tensors.shape[0],
                float(norm.mean()),
                float(norm.max()),
            )
        if RANK in {-1, 0} and getattr(self.args, "debug_replay", False):
            teacher = self.replay_teacher_buffer
            if teacher:
                proto_map = teacher.collect_prototypes()
                for level, class_map in proto_map.items():
                    for cls_id, data in class_map.items():
                        counts = data["counts"]
                        protos = data["prototypes"]
                        if protos.shape[0] <= 1:
                            continue
                        cos = F.normalize(protos, dim=1) @ protos.transpose(0, 1)
                        upper = torch.triu(cos, diagonal=1)
                        if torch.any(upper != 0):
                            vals = upper[upper != 0]
                            LOGGER.info(
                                "Replay proto cos level=%s cls=%s slots=%d min=%.3f max=%.3f",
                                level,
                                int(cls_id),
                                protos.shape[0],
                                float(vals.min()),
                                float(vals.max()),
                            )

    def _compute_replay_consistency(self, current_groups, replay_batch, log_prefix="replay", teacher_buffer=None):
        if not replay_batch:
            return None
        buf = teacher_buffer if teacher_buffer is not None else self.replay_teacher_buffer
        apply_counts = getattr(buf, "weight_by_count", False)
        total_loss = 0.0
        total_weight = 0.0
        for level, class_map in current_groups.items():
            level_targets = replay_batch.get(level)
            if not level_targets:
                continue
            for cls_id, items in class_map.items():
                proto_info = level_targets.get(int(cls_id))
                if not proto_info:
                    continue
                prototypes = proto_info["prototypes"].to(self.device)
                if prototypes.ndim == 1:
                    prototypes = prototypes.unsqueeze(0)
                current_emb = torch.stack([it.embedding.to(self.device) for it in items])
                if current_emb.numel() == 0:
                    continue
                current_norm = F.normalize(current_emb, dim=1)
                cosine = torch.matmul(current_norm, prototypes.transpose(0, 1))
                max_cos, winners = torch.max(cosine, dim=1)
                slot_counts = torch.bincount(winners.cpu(), minlength=prototypes.shape[0]).tolist()
                self._accumulate_replay_hits(level, int(cls_id), slot_counts)
                if RANK in {-1, 0} and getattr(self.args, "debug_replay", False):
                    LOGGER.info(
                        "Replay cosine level=%s cls=%s mean=%.3f median=%.3f min=%.3f max=%.3f",
                        level,
                        int(cls_id),
                        float(max_cos.mean()),
                        float(max_cos.median()),
                        float(max_cos.min()),
                        float(max_cos.max()),
                    )
                weight = self._scale_weight(level)
                loss_vec = (1.0 - max_cos) * weight
                class_weight = 1.0
                if apply_counts:
                    counts = proto_info.get("counts")
                    if counts is not None and counts.numel() > 0:
                        class_weight = float(counts.sum().item())
                total_loss += loss_vec.sum() * class_weight
                total_weight += loss_vec.shape[0] * class_weight
        if total_weight == 0:
            return None
        replay_loss = total_loss / total_weight
        if RANK in {-1, 0}:
            self.auxiliary_info = self.auxiliary_info or {}
            self.auxiliary_info.update(
                {
                    f"{log_prefix}/loss": float(replay_loss.detach()),
                    f"{log_prefix}/examples": float(total_weight),
                }
            )
        return replay_loss

    def _scale_weight(self, level: str) -> float:
        """Compute per-level weighting for replay consistency."""
        mode = (self.replay_scale_weight or "uniform").lower()
        if mode == "stride":
            stride = self.replay_strides.get(level, 8)
            return 1.0 / max(stride, 1)
        if mode == "level":
            mapping = {lvl: idx + 1 for idx, lvl in enumerate(sorted(self.replay_levels))}
            return 1.0 / mapping.get(level, 1)
        return 1.0

    def _accumulate_replay_hits(self, level, cls_id, slot_counts):
        if not slot_counts:
            return
        level_map = self.replay_slot_hits[level]
        cls_map = level_map[int(cls_id)]
        for idx, count in enumerate(slot_counts):
            if count:
                cls_map[idx] += int(count)
        # Track how many embeddings preferred a fine proto over the coarse (slot 0).
        # This is only meaningful when use_coarse=True and there are fine slots.
        self._replay_total_queries += sum(slot_counts)
        if len(slot_counts) > 1:
            self._replay_fine_wins += sum(slot_counts[1:])

    def _write_replay_hits(self, epoch: int):
        if not self.replay_slot_hits:
            return
        log_path = Path(self.save_dir) / "replay_slot_hits.log"
        lines = [f"Epoch {epoch + 1}"]

        # ── Loss-time hits (which prototype won each cosine race) ─────────────
        # Slot 0 = coarse (global EMA mean), slots 1+ = fine prototypes.
        lines.append("  [loss hits: which prototype was nearest each embedding]")
        for level in sorted(self.replay_slot_hits.keys()):
            class_map = self.replay_slot_hits[level]
            for cls_id in sorted(class_map.keys()):
                slot_map = class_map[cls_id]
                parts = []
                for idx, count in sorted(slot_map.items()):
                    label = "coarse" if idx == 0 else f"fine_{idx - 1}"
                    parts.append(f"{label}:{count}")
                lines.append(f"    level={level} cls={cls_id} | {', '.join(parts)}")

        # ── Fine-win summary across all classes/levels ────────────────────────
        total_q = self._replay_total_queries
        fine_w = self._replay_fine_wins
        fine_pct = 100.0 * fine_w / total_q if total_q > 0 else 0.0
        lines.append(
            f"  [fine-win rate] {fine_w}/{total_q} embeddings preferred a fine proto "
            f"over the coarse ({fine_pct:.1f}%)"
        )

        # ── Buffer state (how many times each prototype was updated) ──────────
        # This is independent of the loss race — it shows whether fine slots
        # are populated and how diverse their training coverage is.
        buf = getattr(self, "replay_teacher_buffer", None)
        if buf is not None and hasattr(buf, "_data"):
            lines.append("  [buffer state: prototype update counts]")
            for level, cls_map in sorted(buf._data.items()):
                for cls_id, entry in sorted(cls_map.items()):
                    coarse_count = int(entry.coarse.count) if entry.coarse.vector is not None else 0
                    fine_info = ", ".join(
                        f"fine_{i}:{int(slot.count)}" for i, slot in enumerate(entry.fine)
                    ) if entry.fine else "no fine slots"
                    lines.append(
                        f"    level={level} cls={cls_id} | coarse:{coarse_count}, {fine_info}"
                    )

        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n\n")

        self.replay_slot_hits = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self._replay_fine_wins = 0
        self._replay_total_queries = 0

    def save_metrics(self, metrics):
        aux = getattr(self, "auxiliary_info", None)
        if isinstance(aux, dict):
            for key, value in aux.items():
                metrics[key] = value
                if isinstance(getattr(self, "metrics", None), dict):
                    self.metrics[key] = value
        self._maybe_update_per_class_tb_metrics()
        super().save_metrics(metrics)
        if getattr(self, "replay_enabled", False):
            self._write_replay_hits(self.epoch)

    def _maybe_update_per_class_tb_metrics(self):
        interval = int(getattr(self.args, "per_class_tb_interval", 1) or 1)
        if interval <= 0:
            interval = 1
        epoch = int(self.epoch) + 1
        if epoch % interval != 0:
            return
        metrics_file = Path(self.save_dir) / "per_class_metrics.json"
        if not metrics_file.exists():
            return
        try:
            import json

            entries = json.loads(metrics_file.read_text())
        except Exception:
            return
        if not entries:
            return
        entry = next((e for e in reversed(entries) if e.get("epoch") == epoch), entries[-1])
        per_class = entry.get("metrics", {})
        if not isinstance(per_class, dict):
            return
        self.metrics = self.metrics or {}
        for name, vals in per_class.items():
            if not isinstance(vals, dict):
                continue
            self.metrics[f"val/precision/{name}"] = float(vals.get("precision", 0.0))
            self.metrics[f"val/recall/{name}"] = float(vals.get("recall", 0.0))
            self.metrics[f"val/mAP50/{name}"] = float(vals.get("map50", 0.0))
            self.metrics[f"val/mAP50-95/{name}"] = float(vals.get("map", 0.0))
            self.metrics[f"val/targets/{name}"] = float(vals.get("num_targets", 0))

    def _load_teacher_buffer(self, path: Path):
        buffer = self.replay_teacher_buffer
        if buffer is None:
            return
        try:
            count = buffer.load(path, allow_growth=True)
        except FileNotFoundError:
            LOGGER.warning(f"Replay init buffer not found at {path}")
            return
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(f"Failed to load replay buffer from {path}: {exc}")
            return
        if count > 0:
            LOGGER.info(f"Loaded {count} replay embeddings from {path}")

    def _load_box_teacher_buffer(self, path: Path):
        buffer = self.replay_box_teacher_buffer
        if buffer is None:
            return
        try:
            count = buffer.load(path, allow_growth=True)
        except FileNotFoundError:
            LOGGER.warning(f"Box replay init buffer not found at {path}")
            return
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(f"Failed to load box replay buffer from {path}: {exc}")
            return
        if count > 0:
            LOGGER.info(f"Loaded {count} box replay embeddings from {path}")

    def _maybe_save_replay_buffer(self, ctx=None):
        tag = ctx.get("type") if isinstance(ctx, dict) else None
        suffix = f" ({tag})" if tag else ""
        self._save_one_buffer(
            self.replay_teacher_buffer,
            self.replay_student_buffer,
            self.replay_save_path,
            label="replay",
            suffix=suffix,
        )
        if getattr(self, "replay_box_enabled", False):
            self._save_one_buffer(
                self.replay_box_teacher_buffer,
                self.replay_box_student_buffer,
                getattr(self, "replay_box_save_path", None),
                label="box replay",
                suffix=suffix,
            )

    def _save_one_buffer(self, teacher, student, save_path, label="replay", suffix=""):
        """Merge teacher+student buffers and persist to disk. Independent of other buffers."""
        if not save_path:
            return
        has_teacher = bool(teacher and len(teacher) > 0)
        has_student = bool(student and len(student) > 0)
        if not (has_teacher or has_student):
            return
        base_buffer = student or teacher
        combined = TinyReplayBuffer(
            per_class_capacity=base_buffer.capacity,
            dtype=base_buffer.dtype,
            device="cpu",
            carryover_growth=self.replay_capacity_growth,
            num_fine=base_buffer.num_fine,
            use_coarse=base_buffer.use_coarse,
            ema_alpha=base_buffer.ema_alpha,
            init_sim_thresh=base_buffer.init_sim_thresh,
            gate_min_cos=base_buffer.gate_min_cos,
            init_strategy=base_buffer.init_strategy,
            weight_by_count=base_buffer.weight_by_count,
        )
        if has_teacher:
            combined.merge_from(teacher)
        if has_student:
            combined.merge_from(student)
        try:
            saved = combined.save(save_path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(f"Failed to save {label} buffer to {save_path}: {exc}")
            return
        LOGGER.info(f"Saved {saved} {label} embeddings to {save_path}{suffix}")
        # Persist fine prototype source metadata for visualization
        if combined.num_fine > 0:
            try:
                sources_path = Path(save_path).with_name(
                    Path(save_path).stem + "_sources.json"
                )
                combined.save_sources(sources_path)
                LOGGER.info(f"Saved fine prototype sources to {sources_path}")
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(f"Failed to save fine prototype sources: {exc}")
