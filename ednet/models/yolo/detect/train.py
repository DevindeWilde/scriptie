import contextlib
import itertools
import math
import random
from collections import defaultdict
from copy import copy
from pathlib import Path
from typing import Optional

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
        if weights:
            model.load(weights)
        model.active_class_ids = self.active_class_ids
        model.prev_class_ids = self.prev_class_ids
        lora_args = getattr(self.args, "lora", None)
        self.lora_enabled = bool(isinstance(lora_args, dict) and lora_args.get("enable"))
        self.lora_freeze_backbone = False
        if self.lora_enabled:
            lora_config = LoRAConfig(
                rank=int(lora_args.get("rank", 8)),
                alpha=float(lora_args.get("alpha", 16.0)),
                dropout=float(lora_args.get("dropout", 0.0)),
                feature_pyramid_indices=tuple(int(idx) for idx in lora_args.get("feature_pyramid_indices", (19, 22, 25))),
                include_detection_head=bool(lora_args.get("include_detection_head", False)),
            )
            freeze_backbone = bool(lora_args.get("freeze_backbone", True))
            self.lora_freeze_backbone = freeze_backbone
            adapters = model.enable_lora(lora_config)
            LOGGER.info(
                f"LoRA enabled: {len(adapters)} adapters (rank={lora_config.rank}, alpha={lora_config.alpha}, "
            )
            if adapters and RANK in {-1, 0}:
                adapter_names = [name for name, _ in adapters]
                LOGGER.info("LoRA adapter registry: %d modules -> %s", len(adapter_names), adapter_names)
            trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            total_params = sum(p.numel() for _, p in trainable)
            head = [(n, p.numel()) for n, p in itertools.islice(trainable, 20)]
            LOGGER.info(
                "Trainable params: %d tensors (%d weights) example=%s",
                len(trainable), total_params, head,
            )
            adapter_path = lora_args.get("init_adapter")
            if adapter_path:
                self._load_adapter_weights(model, adapter_path)
        replay_args = getattr(self.args, "replay", None)
        self.replay_enabled = bool(isinstance(replay_args, dict) and replay_args.get("enable"))
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
        base_loss_names = ["box_loss", "cls_loss", "dfl_loss"]
        if self.replay_enabled:
            base_loss_names.append("replay_loss")
        self.loss_names = tuple(base_loss_names)
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
        base_loss_names = ["box_loss", "cls_loss", "dfl_loss"]
        if getattr(self, "replay_enabled", False):
            base_loss_names.append("replay_loss")
        self.loss_names = tuple(base_loss_names)
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        if self.replay_enabled and getattr(self, "feature_tapper", None) and self._feature_tapper_needs_activation:
            self.feature_tapper.activate()
            self._feature_tapper_needs_activation = False
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
                    if isinstance(mem_val, list):
                        merged[key] = merged[key] + mem_val
                    else:
                        merged[key].append(mem_val)
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

    def _before_checkpoint(self):
        if getattr(self, "feature_tapper", None):
            self.feature_tapper.deactivate()
        return {}

    def _after_checkpoint(self, ctx):
        if getattr(self, "feature_tapper", None):
            self.feature_tapper.activate()
        if self.replay_enabled:
            self._maybe_save_replay_buffer(ctx)
        super()._after_checkpoint(ctx)

    def compute_auxiliary_loss(self, batch):
        """Compute replay-based feature consistency loss."""
        print("DetectionTrainer.compute_auxiliary_loss called.")
        teacher_ready = (
            self.replay_enabled
            and self.feature_tapper is not None
            and self.replay_teacher_buffer is not None
            and len(self.replay_teacher_buffer) > 0
        )
        student_ready = self.replay_student_buffer is not None and self.feature_tapper is not None
        if not teacher_ready and not student_ready:
            print("No replay auxiliary loss computed: teacher_ready =", teacher_ready, "student_ready =", student_ready)
            return None
        criterion = self._ensure_replay_tap_config()
        features = self.feature_tapper.pop()
        if not features:
            return None
        aux_loss = None
        if teacher_ready:
            print("Computing replay auxiliary loss with teacher buffer...")
            replay_items = self._gather_embeddings(features, attr="last_replay_cells", criterion=criterion)
            print(f"  Gathered {len(replay_items)} replay embeddings from tapped features.")
            if replay_items:
                grouped = self._group_embeddings(replay_items)
                self._log_embedding_stats(grouped)
                replay_batch = build_replay_batch(
                    self.replay_teacher_buffer,
                    per_class=self.replay_samples_per_class,
                    balance_levels=self.replay_levels,
                    device=self.device,
                )
                aux_loss = self._compute_replay_consistency(grouped, replay_batch)
        if self.replay_student_buffer is not None:
            student_items = self._gather_embeddings(features, attr="last_positive_cells", criterion=criterion)
            for item in student_items:
                self.replay_student_buffer.add(item)
        if aux_loss is None:
            print("No replay auxiliary loss computed: aux_loss is None")
            return None
        print("Replay auxiliary loss computed:", float(aux_loss))
        return aux_loss * self.replay_loss_weight

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

    def _gather_embeddings(self, features, attr="last_positive_cells", criterion=None):
        """Collect TinyReplayItems from tapped features using stored cell indices."""
        if criterion is None:
            criterion = self._ensure_replay_tap_config()
        loss_module = criterion
        if loss_module is None:
            return []
        if hasattr(loss_module, "set_replay_tap_config") and not getattr(loss_module, "_replay_configured", False):
            if self.replay_levels and self.replay_strides:
                loss_module.set_replay_tap_config(self.replay_levels, self.replay_strides)
        pos = getattr(loss_module, attr, None)
        if not pos:
            return []
        indices = pos.get("indices")
        classes = pos.get("classes")
        max_edge = pos.get("max_edge")
        if indices is None or classes is None or max_edge is None:
            return []
        if indices.numel() == 0:
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
            return []
        if self.replay_debug:
            LOGGER.info(
                "Replay tap positives before filter=%d after=%d",
                int(size_mask.numel()),
                int(size_mask.sum()),
            )
        indices = indices[size_mask]
        classes = classes[size_mask]
        max_edge = max_edge[size_mask]
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
            batch_idx = sel_indices[:, 1].long()
            gy = sel_indices[:, 2].long()
            gx = sel_indices[:, 3].long()
            vecs = feat[batch_idx, :, gy, gx]
            for embedding, cls_id, size in zip(vecs, sel_classes, sel_sizes):
                items.append(
                    TinyReplayItem(
                        cls=int(cls_id.item()),
                        level=name,
                        embedding=embedding.detach(),
                        metadata={"max_edge": float(size.item())},
                    )
                )
        if attr == "last_positive_cells":
            loss_module.last_positive_cells = None
        if attr == "last_replay_cells" and hasattr(loss_module, "last_replay_cells"):
            loss_module.last_replay_cells = None
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

    def _compute_replay_consistency(self, current_groups, replay_batch):
        if not replay_batch:
            return None
        apply_counts = getattr(self.replay_teacher_buffer, "weight_by_count", False)
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
                    "replay/loss": float(replay_loss.detach()),
                    "replay/examples": float(total_weight),
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

    def _write_replay_hits(self, epoch: int):
        if not self.replay_slot_hits:
            return
        log_path = Path(self.save_dir) / "replay_slot_hits.log"
        lines = [f"Epoch {epoch + 1}"]
        for level in sorted(self.replay_slot_hits.keys()):
            class_map = self.replay_slot_hits[level]
            for cls_id in sorted(class_map.keys()):
                slot_map = class_map[cls_id]
                slot_str = ", ".join(f"{idx}:{count}" for idx, count in sorted(slot_map.items()))
                lines.append(f"  level={level} cls={cls_id} hits={slot_str}")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        self.replay_slot_hits = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    def save_metrics(self, metrics):
        aux = getattr(self, "auxiliary_info", None)
        if isinstance(aux, dict):
            for key, value in aux.items():
                metrics[key] = value
        super().save_metrics(metrics)
        if getattr(self, "replay_enabled", False):
            self._write_replay_hits(self.epoch)

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

    def _maybe_save_replay_buffer(self, ctx=None):
        if not self.replay_save_path:
            return
        teacher = self.replay_teacher_buffer
        student = self.replay_student_buffer
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
        )
        if has_teacher:
            combined.load_state_dict(teacher.state_dict())
        if has_student:
            for item in student.iter_items():
                combined.add(item)
        try:
            saved = combined.save(self.replay_save_path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(f"Failed to save replay buffer to {self.replay_save_path}: {exc}")
            return
        tag = ctx.get("type") if isinstance(ctx, dict) else None
        suffix = f" ({tag})" if tag else ""
        LOGGER.info(f"Saved {saved} replay embeddings to {self.replay_save_path}{suffix}")
