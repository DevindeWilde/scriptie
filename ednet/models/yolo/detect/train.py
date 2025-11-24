import math
import random
from copy import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ednet.data import build_dataloader, build_yolo_dataset
from ednet.engine.replay import (
    FeatureTapConfig,
    FeatureTapper,
    TinyReplayBuffer,
    build_replay_batch,
    extract_tiny_embeddings,
)
from ednet.engine.trainer import BaseTrainer
from ednet.models import yolo
from ednet.nn.lora import LoRAConfig
from ednet.nn.tasks import DetectionModel
from ednet.utils import LOGGER, RANK
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
        if weights:
            model.load(weights)
        lora_args = getattr(self.args, "lora", None)
        self.lora_enabled = bool(isinstance(lora_args, dict) and lora_args.get("enable"))
        if self.lora_enabled:
            lora_config = LoRAConfig(
                rank=int(lora_args.get("rank", 8)),
                alpha=float(lora_args.get("alpha", 16.0)),
                dropout=float(lora_args.get("dropout", 0.0)),
                feature_pyramid_indices=tuple(int(idx) for idx in lora_args.get("feature_pyramid_indices", (16, 22, 25))),
                include_detection_head=bool(lora_args.get("include_detection_head", True)),
            )
            freeze_backbone = bool(lora_args.get("freeze_backbone", True))
            adapters = model.enable_lora(lora_config, freeze_backbone=freeze_backbone)
            LOGGER.info(
                f"LoRA enabled: {len(adapters)} adapters (rank={lora_config.rank}, alpha={lora_config.alpha}, "
                f"freeze_backbone={freeze_backbone})"
            )
            adapter_path = lora_args.get("init_adapter")
            if adapter_path:
                self._load_adapter_weights(model, adapter_path)
        replay_args = getattr(self.args, "replay", None)
        self.replay_enabled = bool(isinstance(replay_args, dict) and replay_args.get("enable"))
        self.feature_tapper = None
        self.replay_buffer = None
        self.replay_strides = {}
        self.replay_levels = []
        self.replay_samples_per_class = 0
        self.replay_loss_weight = 1.0
        self.replay_max_edge = 32.0
        self.replay_context_scale = 1.5
        self.replay_scale_weight = "uniform"
        self._feature_tapper_needs_activation = False
        if self.replay_enabled:
            tap_layers = replay_args.get("tap_layers") or {}
            layer_map = {k: int(v) for k, v in tap_layers.items()} if tap_layers else FeatureTapConfig().layers.copy()
            tap_config = FeatureTapConfig(layers=layer_map, detach=True)
            self.feature_tapper = FeatureTapper(model, tap_config, auto_activate=False)
            self._feature_tapper_needs_activation = True
            LOGGER.info(f"Replay feature tapper initialized for layers: {list(tap_config.layers.keys())}")
            capacity = int(replay_args.get("buffer_per_class", 64))
            self.replay_buffer = TinyReplayBuffer(per_class_capacity=capacity, dtype=torch.float16, device="cpu")
            default_stride_map = {"P2": 4, "P3": 8, "P4": 16, "P5": 32, "P6": 64}
            stride_overrides = replay_args.get("strides") or {}
            self.replay_strides = {
                level: int(stride_overrides.get(level, default_stride_map.get(level, 8)))
                for level in layer_map.keys()
            }
            self.replay_levels = list(layer_map.keys())
            self.replay_samples_per_class = max(1, int(replay_args.get("sample_per_batch", 16)))
            self.replay_loss_weight = float(replay_args.get("loss_weight", 1.0))
            self.replay_max_edge = float(replay_args.get("tiny_max_pixels", 32))
            self.replay_context_scale = float(replay_args.get("context_scale", 1.5))
            self.replay_scale_weight = replay_args.get("scale_weighting", "uniform")
        return model

    def _load_adapter_weights(self, model, adapter_path):
        """Load LoRA adapter weights from disk if available."""
        adapter_path = Path(adapter_path)
        if not adapter_path.exists():
            LOGGER.warning(f"LoRA adapter initialization skipped; file not found: {adapter_path}")
            return
        state = torch.load(adapter_path, map_location="cpu")
        model.load_lora_state_dict(state)
        LOGGER.info(f"Loaded LoRA adapter weights from {adapter_path}")

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        if self.replay_enabled and getattr(self, "feature_tapper", None) and self._feature_tapper_needs_activation:
            self.feature_tapper.activate()
            self._feature_tapper_needs_activation = False

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

    def _before_checkpoint(self):
        if getattr(self, "feature_tapper", None):
            self.feature_tapper.deactivate()
        return {}

    def _after_checkpoint(self, ctx):
        if getattr(self, "feature_tapper", None):
            self.feature_tapper.activate()
        super()._after_checkpoint(ctx)

    def compute_auxiliary_loss(self, batch):
        """Compute replay-based feature consistency loss."""
        if not self.replay_enabled or self.feature_tapper is None or self.replay_buffer is None:
            return None
        features = self.feature_tapper.pop()
        if not features:
            return None
        boxes = batch.get("bboxes")
        classes = batch.get("cls")
        batch_idx = batch.get("batch_idx")
        imgs = batch.get("img")
        if boxes is None or classes is None or batch_idx is None or imgs is None:
            return None
        if boxes.numel() == 0:
            return None
        base_level = next(iter(features.values()))
        target_device = base_level.device
        img_h, img_w = imgs.shape[2:]
        boxes = boxes.to(target_device)
        classes = classes.to(target_device).view(-1)
        batch_idx = batch_idx.to(target_device).long()
        current_items = extract_tiny_embeddings(
            features,
            boxes,
            classes,
            batch_idx,
            self.replay_strides,
            self.replay_max_edge,
            self.replay_context_scale,
            (img_h, img_w),
        )
        if not current_items:
            return None
        grouped = self._group_embeddings(current_items)
        replay_batch = build_replay_batch(
            self.replay_buffer,
            per_class=self.replay_samples_per_class,
            balance_levels=self.replay_levels,
            device=self.device,
        )
        aux_loss = self._compute_replay_consistency(grouped, replay_batch)
        for item in current_items:
            self.replay_buffer.add(item)
        if aux_loss is None:
            return None
        return aux_loss * self.replay_loss_weight

    def _group_embeddings(self, items):
        grouped = {}
        for item in items:
            level_map = grouped.setdefault(item.level, {})
            level_map.setdefault(item.cls, []).append(item)
        return grouped

    def _compute_replay_consistency(self, current_groups, replay_batch):
        if not replay_batch:
            return None
        total_loss = 0.0
        loss_terms = 0
        for level, class_map in current_groups.items():
            if level not in replay_batch:
                continue
            targets = replay_batch[level]
            target_cls = targets["cls"]
            for cls_id, items in class_map.items():
                mask = target_cls == int(cls_id)
                if not mask.any():
                    continue
                current_tiny = torch.stack([it.tiny.to(self.device) for it in items]).mean(dim=0)
                current_context = torch.stack([it.context.to(self.device) for it in items]).mean(dim=0)
                target_tiny = targets["tiny"][mask].mean(dim=0)
                target_context = targets["context"][mask].mean(dim=0)
                weight = self._scale_weight(level)
                total_loss += weight * F.mse_loss(current_tiny, target_tiny)
                total_loss += weight * F.mse_loss(current_context, target_context)
                loss_terms += 2
        if loss_terms == 0:
            return None
        return total_loss / loss_terms

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
