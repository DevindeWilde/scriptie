from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass
class FeatureTapConfig:
    """
    Configuration describing which layers in the detection backbone/FPN should be tapped.

    Attributes:
        layers (Dict[str, int]): Mapping from semantic level name (e.g., "P3") to the layer index
            in the parsed EDNet model (`model.model[idx]`).
        detach (bool): Whether to detach features from the autograd graph before caching them.
        clone (bool): Whether to clone the tensor before caching (useful if downstream code
            modifies tensors in place).
    """

    layers: Dict[str, int] = field(default_factory=lambda: {"P3": 16, "P4": 25})
    detach: bool = True
    clone: bool = False


class FeatureTapper:
    """
    Utility that registers forward hooks on selected layers and stores their outputs for replay.

    Example:
        cfg = FeatureTapConfig(layers={"P3": 16, "P4": 25})
        tapper = FeatureTapper(model, cfg)
        ...
        features = tapper.pop()
    """

    def __init__(self, model: nn.Module, config: FeatureTapConfig, auto_activate: bool = True) -> None:
        self.model = model
        self.config = config
        self._features: Dict[str, torch.Tensor] = {}
        self._hooks = []
        self._layers = {}
        seq = getattr(model, "model", None)
        if not isinstance(seq, nn.Sequential):
            raise TypeError("FeatureTapper expects `model.model` to be an nn.Sequential container.")

        for level, idx in config.layers.items():
            if not isinstance(idx, int):
                raise TypeError(f"Layer index for level '{level}' must be an integer, got {type(idx)}")
            if idx < 0 or idx >= len(seq):
                raise IndexError(f"Layer index {idx} for level '{level}' is out of bounds (len={len(seq)})")
            module = seq[idx]
            self._layers[level] = module

        if auto_activate:
            self.activate()

    def _build_hook(self, level: str):
        def hook(_, __, output):
            tensor = output
            if isinstance(tensor, (tuple, list)):
                tensor = tensor[0]
            if self.config.detach and isinstance(tensor, torch.Tensor):
                tensor = tensor.detach()
            if self.config.clone and isinstance(tensor, torch.Tensor):
                tensor = tensor.clone()
            self._features[level] = tensor

        return hook

    def get(self, level: str) -> Optional[torch.Tensor]:
        """Return the cached tensor for the requested level without clearing it."""
        return self._features.get(level)

    def pop(self) -> Dict[str, torch.Tensor]:
        """Return all cached features and clear the internal buffer."""
        features = self._features
        self._features = {}
        return features

    def clear(self) -> None:
        """Clear cached features."""
        self._features.clear()

    def close(self) -> None:
        """Remove all registered hooks to avoid dangling references."""
        self.deactivate()

    def deactivate(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def activate(self) -> None:
        if self._hooks:
            return
        for level, module in self._layers.items():
            handle = module.register_forward_hook(self._build_hook(level))
            self._hooks.append(handle)

    def __del__(self):
        self.close()


@dataclass
class TinyReplayItem:
    """Container for a single replay entry consisting of aggregated tiny/context embeddings."""

    cls: int
    level: str
    tiny: torch.Tensor
    context: torch.Tensor
    metadata: Dict = field(default_factory=dict)


class TinyReplayBuffer:
    """
    Memory-efficient buffer that stores a limited number of TinyEx + ContextEx feature patches per class.
    """

    def __init__(
        self,
        per_class_capacity: int = 64,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = torch.device("cpu"),
    ) -> None:
        self.capacity = per_class_capacity
        self.dtype = dtype
        self.device = torch.device(device)
        self._storage: Dict[int, List[TinyReplayItem]] = defaultdict(list)

    def __len__(self) -> int:
        return sum(len(items) for items in self._storage.values())

    def classes(self) -> List[int]:
        return list(self._storage.keys())

    def add(self, item: TinyReplayItem) -> None:
        """Insert a new sample while respecting per-class capacity."""
        entry = TinyReplayItem(
            cls=item.cls,
            level=item.level,
            tiny=item.tiny.detach().to(self.device, dtype=self.dtype),
            context=item.context.detach().to(self.device, dtype=self.dtype),
            metadata=item.metadata,
        )
        bucket = self._storage[entry.cls]
        bucket.append(entry)
        if len(bucket) > self.capacity:
            bucket.pop(0)

    def sample_balanced(
        self,
        max_per_class: int,
        levels: Optional[Iterable[str]] = None,
    ) -> List[TinyReplayItem]:
        """Sample up to `max_per_class` entries per class, optionally filtering by feature level."""
        samples: List[TinyReplayItem] = []
        allowed_levels = set(levels) if levels else None
        for cls, items in self._storage.items():
            pool = (
                [item for item in items if (allowed_levels is None or item.level in allowed_levels)]
                if allowed_levels
                else items
            )
            if not pool:
                continue
            k = min(max_per_class, len(pool))
            samples.extend(random.sample(pool, k) if k < len(pool) else pool[:])
        return samples

    def clear(self) -> None:
        self._storage.clear()


def build_replay_batch(
    buffer: TinyReplayBuffer,
    per_class: int,
    balance_levels: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Sample embeddings and organize them by feature level for downstream losses.

    Returns:
        Dict level -> {"cls": tensor, "tiny": tensor, "context": tensor}
    """
    samples = buffer.sample_balanced(per_class, levels=balance_levels)
    by_level: Dict[str, Dict[str, List[torch.Tensor]]] = {}
    for item in samples:
        entry = by_level.setdefault(item.level, {"cls": [], "tiny": [], "context": []})
        entry["cls"].append(torch.tensor(item.cls, dtype=torch.long))
        entry["tiny"].append(item.tiny)
        entry["context"].append(item.context)

    batched = {}
    for level, data in by_level.items():
        level_batch = {
            "cls": torch.stack(data["cls"]),
            "tiny": torch.stack(data["tiny"]),
            "context": torch.stack(data["context"]),
        }
        if device is not None:
            level_batch = {k: v.to(device) for k, v in level_batch.items()}
        batched[level] = level_batch
    return batched


def extract_tiny_embeddings(
    features: Dict[str, torch.Tensor],
    boxes: torch.Tensor,
    classes: torch.Tensor,
    batch_indices: torch.Tensor,
    strides: Dict[str, int],
    max_edge: float,
    context_scale: float = 1.5,
    image_hw: Tuple[int, int] = (640, 640),
) -> List[TinyReplayItem]:
    """
    Extract TinyEx and ContextEx embeddings for tiny objects by pooling feature crops.
    """

    if boxes.numel() == 0:
        return []

    img_h, img_w = image_hw
    boxes_px = boxes.clone()
    boxes_px[:, [0, 2]] *= img_w
    boxes_px[:, [1, 3]] *= img_h

    items: List[TinyReplayItem] = []
    levels_sorted = sorted(strides.items(), key=lambda kv: kv[1])  # smallest stride first

    for box, cls, b_idx in zip(boxes_px, classes, batch_indices):
        width = float(box[2] - box[0])
        height = float(box[3] - box[1])
        if max(width, height) > max_edge:
            continue

        img_index = int(b_idx)
        for level, stride in levels_sorted:
            feat = features.get(level)
            if feat is None or img_index >= feat.shape[0]:
                continue
            sample_feat = feat[img_index]
            h, w = sample_feat.shape[-2], sample_feat.shape[-1]

            x1 = (box[0] / stride).clamp(0, w - 1)
            y1 = (box[1] / stride).clamp(0, h - 1)
            x2 = (box[2] / stride).clamp(0, w - 1)
            y2 = (box[3] / stride).clamp(0, h - 1)

            tiny_patch = crop_feature(sample_feat, x1, y1, x2, y2)
            cx1, cy1, cx2, cy2 = expand_box(x1, y1, x2, y2, context_scale, w, h)
            context_patch = crop_feature(sample_feat, cx1, cy1, cx2, cy2)

            tiny_emb = adaptive_pool(tiny_patch)
            context_emb = adaptive_pool(context_patch)

            items.append(
                TinyReplayItem(
                    cls=int(cls),
                    level=level,
                    tiny=tiny_emb,
                    context=context_emb,
                    metadata={"stride": stride},
                )
            )
            break  # assign to the first available level
    return items


def adaptive_pool(patch: torch.Tensor) -> torch.Tensor:
    """Compute a channel-level embedding from a spatial patch."""
    if patch.numel() == 0:
        return torch.zeros(patch.shape[0], device=patch.device, dtype=patch.dtype)
    pooled = patch.mean(dim=(1, 2), keepdim=False)
    return pooled.detach()


def crop_feature(feature: torch.Tensor, x1, y1, x2, y2) -> torch.Tensor:
    """Crop a spatial region from a single feature map."""
    x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2) + 1, int(y2) + 1
    x1i, y1i = max(0, x1i), max(0, y1i)
    x2i, y2i = min(feature.shape[-1], x2i), min(feature.shape[-2], y2i)
    if x2i <= x1i or y2i <= y1i:
        return torch.zeros_like(feature[:, :1, :1])
    return feature[:, y1i:y2i, x1i:x2i]


def expand_box(x1, y1, x2, y2, scale, width, height):
    """Expand box coordinates by a scale factor while clamping to feature map size."""
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale
    new_x1 = max(0.0, cx - w / 2)
    new_y1 = max(0.0, cy - h / 2)
    new_x2 = min(width - 1.0, cx + w / 2)
    new_y2 = min(height - 1.0, cy + h / 2)
    return new_x1, new_y1, new_x2, new_y2
