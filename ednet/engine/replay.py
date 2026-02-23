from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.hooks import RemovableHandle


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

    layers: Dict[str, int] = field(default_factory=lambda: {"P2": 19, "P3": 16, "P4": 25})
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


class DetectionPreLogitTapper:
    """Tapper that records detection head pre-logit tensors via forward pre-hooks.

    By default taps the classification branch (``cv3``).  Pass ``branch_attr="cv2"``
    to tap the box-regression branch instead.
    """

    def __init__(
        self,
        detect_module: nn.Module,
        level_to_indices: Dict[str, int] | None = None,
        detach: bool = False,
        clone: bool = False,
        auto_activate: bool = True,
        branch_attr: str = "cv3",
    ) -> None:
        self.detect_module = detect_module
        self.level_to_indices = level_to_indices or {}
        self.detach = detach
        self.clone = clone
        self.branch_attr = branch_attr
        self._features: Dict[str, torch.Tensor] = {}
        self._hooks: List[RemovableHandle] = []
        if auto_activate:
            self.activate()

    def _classification_branches(self) -> Optional[nn.ModuleList]:
        branches = getattr(self.detect_module, self.branch_attr, None)
        if branches is None:
            return None
        return branches

    def _build_hook(self, level: str):
        def hook(_, inputs):
            if not inputs:
                return
            tensor = inputs[0]
            if isinstance(tensor, (list, tuple)):
                tensor = tensor[0]
            if isinstance(tensor, torch.Tensor):
                #print(f"[Replay tap] level={level} shape={tuple(tensor.shape)}")
                if self.detach:
                    tensor = tensor.detach()
                if self.clone:
                    tensor = tensor.clone()
                self._features[level] = tensor

        return hook

    def activate(self) -> None:
        if self._hooks:
            return
        branches = self._classification_branches()
        if branches is None:
            return
        for level, idx in self.level_to_indices.items():
            if idx < 0 or idx >= len(branches):
                continue
            branch = branches[idx]
            if not isinstance(branch, nn.Sequential) or len(branch) == 0:
                continue
            final_conv = branch[-1]
            handle = final_conv.register_forward_pre_hook(self._build_hook(level))
            self._hooks.append(handle)

    def deactivate(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def pop(self) -> Dict[str, torch.Tensor]:
        features = self._features
        self._features = {}
        return features

    def clear(self) -> None:
        self._features.clear()

    def close(self) -> None:
        self.deactivate()

    def __del__(self):
        self.close()


@dataclass
class TinyReplayItem:
    """Container for a single replay sample consisting of a detection head pre-logit embedding."""

    cls: int
    level: str
    embedding: torch.Tensor
    metadata: Dict = field(default_factory=dict)


@dataclass
class PrototypeEntry:
    """Single prototype statistics."""

    vector: torch.Tensor
    count: int = 1
    source: Optional[Dict] = None  # {"im_file": str, "bbox_xywh_norm": list, "epoch": int}

    def clone(self) -> torch.Tensor:
        return self.vector.clone()


@dataclass
class MultiPrototypeEntry:
    """Collection of coarse and fine prototypes for a class/level."""

    coarse: PrototypeEntry
    fine: List[PrototypeEntry] = field(default_factory=list)


class TinyReplayBuffer:
    """
    Buffer that stores running prototypes per (level, class) instead of raw feature crops.
    """

    def __init__(
        self,
        per_class_capacity: int = 64,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = torch.device("cpu"),
        carryover_growth: float = 1.5,
        num_fine: int = 0,
        use_coarse: bool = True,
        ema_alpha: float = 0.05,
        init_sim_thresh: float = 0.9,
        gate_min_cos: float = 0.0,
        init_strategy: str = "first_k",
        weight_by_count: bool = False,
    ) -> None:
        self.capacity = per_class_capacity
        self.dtype = dtype
        self.device = torch.device(device)
        self.carryover_growth = float(max(carryover_growth, 1.0))
        self.num_fine = max(0, int(num_fine))
        self.use_coarse = bool(use_coarse)
        self.ema_alpha = float(ema_alpha)
        self.init_sim_thresh = float(init_sim_thresh)
        self.gate_min_cos = float(gate_min_cos)
        self.init_strategy = init_strategy
        self.weight_by_count = bool(weight_by_count)
        self._storage: Dict[str, Dict[int, MultiPrototypeEntry]] = defaultdict(dict)

    def __len__(self) -> int:
        total = 0
        for class_map in self._storage.values():
            for entry in class_map.values():
                total += entry.coarse.count
                for slot in entry.fine:
                    total += slot.count
        return total

    def classes(self) -> List[int]:
        cls_ids = set()
        for class_map in self._storage.values():
            cls_ids.update(class_map.keys())
        return sorted(cls_ids)

    def add(self, item: TinyReplayItem) -> None:
        """Insert a sample by updating prototypes for its level and class."""
        if item.embedding is None or item.embedding.numel() == 0:
            print("Warning: Attempted to add an item with empty embedding to TinyReplayBuffer; skipping.")
            return
        embedding = self._normalize(item.embedding)
        entry, created = self._get_or_create_entry(item.level, item.cls, embedding)
        if not created:
            self._update_coarse(entry.coarse, embedding)
        meta = dict(item.metadata or {})
        epoch = getattr(self, "current_epoch", None)
        if epoch is not None:
            meta["epoch"] = int(epoch)
        self._update_fine(entry, embedding, metadata=meta)

    def merge_from(self, other: "TinyReplayBuffer" | None) -> int:
        """Merge all entries from another buffer while preserving prototype counts."""
        if other is None:
            return 0
        merged = 0
        for level, class_map in other._storage.items():
            for cls_id, entry in class_map.items():
                merged += self._merge_entry(level, int(cls_id), entry)
        return merged

    def sample_balanced(
        self,
        max_per_class: int,
        levels: Optional[Iterable[str]] = None,
    ) -> List[TinyReplayItem]:
        """Return prototypes per (level, class), optionally filtering by feature level."""
        if max_per_class <= 0:
            return []
        samples: List[TinyReplayItem] = []
        allowed_levels = set(levels) if levels else None
        for level, class_map in self._storage.items():
            if allowed_levels is not None and level not in allowed_levels:
                continue
            for cls_id, entry in class_map.items():
                if self.use_coarse and entry.coarse.vector is not None:
                    samples.append(
                        TinyReplayItem(
                            cls=int(cls_id),
                            level=level,
                            embedding=entry.coarse.clone().to(self.device),
                            metadata={"count": entry.coarse.count, "slot": "coarse"},
                        )
                    )
        return samples

    def clear(self) -> None:
        self._storage = defaultdict(dict)

    def counts(self) -> Dict[int, int]:
        """Return aggregated sample counts per class across all levels."""
        aggregated: Dict[int, int] = defaultdict(int)
        for class_map in self._storage.values():
            for cls_id, entry in class_map.items():
                total = entry.coarse.count + sum(slot.count for slot in entry.fine)
                aggregated[int(cls_id)] += int(total)
        return dict(aggregated)

    def iter_items(self, levels: Optional[Iterable[str]] = None):
        allowed_levels = set(levels) if levels else None
        for level, class_map in self._storage.items():
            if allowed_levels is not None and level not in allowed_levels:
                continue
            for cls_id, entry in class_map.items():
                if entry.coarse.vector is not None and entry.coarse.vector.numel() > 0:
                    yield TinyReplayItem(
                        cls=int(cls_id),
                        level=level,
                        embedding=entry.coarse.clone(),
                        metadata={"count": entry.coarse.count, "slot": "coarse"},
                    )
                for idx, slot in enumerate(entry.fine):
                    if slot.vector is None or slot.vector.numel() == 0:
                        continue
                    yield TinyReplayItem(
                        cls=int(cls_id),
                        level=level,
                        embedding=slot.vector.clone(),
                        metadata={"count": slot.count, "slot": f"fine_{idx}"},
                    )

    def collect_prototypes(
        self,
        levels: Optional[Iterable[str]] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Dict[int, Dict[str, torch.Tensor]]]:
        allowed_levels = set(levels) if levels else None
        result: Dict[str, Dict[int, Dict[str, torch.Tensor]]] = {}
        target_device = torch.device(device) if device is not None else self.device
        for level, class_map in self._storage.items():
            if allowed_levels is not None and level not in allowed_levels:
                continue
            level_dict: Dict[int, Dict[str, torch.Tensor]] = {}
            for cls_id, entry in class_map.items():
                vectors: List[torch.Tensor] = []
                counts: List[float] = []
                if self.use_coarse and entry.coarse.vector is not None and entry.coarse.vector.numel() > 0:
                    vectors.append(entry.coarse.vector.to(target_device, dtype=torch.float32))
                    counts.append(float(entry.coarse.count))
                for slot in entry.fine:
                    if slot.vector is None or slot.vector.numel() == 0:
                        continue
                    vectors.append(slot.vector.to(target_device, dtype=torch.float32))
                    counts.append(float(slot.count))
                if not vectors:
                    continue
                proto_tensor = torch.stack(vectors, dim=0)
                proto_tensor = F.normalize(proto_tensor, dim=1, eps=1e-6)
                count_tensor = torch.tensor(counts, device=proto_tensor.device, dtype=proto_tensor.dtype)
                level_dict[int(cls_id)] = {"prototypes": proto_tensor, "counts": count_tensor}
            if level_dict:
                result[level] = level_dict
        return result

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        vec = tensor.detach().to(self.device, dtype=torch.float32)
        return F.normalize(vec, dim=0, eps=1e-6)

    def _get_or_create_entry(self, level: str, cls: int, embedding: torch.Tensor) -> Tuple[MultiPrototypeEntry, bool]:
        bucket = self._storage[level]
        entry = bucket.get(cls)
        if entry is None:
            entry = MultiPrototypeEntry(
                coarse=PrototypeEntry(vector=embedding.to(self.dtype), count=1),
                fine=[],
            )
            bucket[cls] = entry
            return entry, True
        return entry, False

    def _update_coarse(self, coarse: PrototypeEntry, embedding: torch.Tensor) -> None:
        vec = coarse.vector.to(torch.float32)
        count = coarse.count
        #print("Updating coarse prototype with count", count)
        if count < self.capacity:
            count += 1
            vec = vec + (embedding - vec) / count
        else:
            momentum = (self.capacity - 1) / float(self.capacity)
            vec = vec * momentum + embedding * (1.0 - momentum)
        coarse.vector = F.normalize(vec, dim=0, eps=1e-6).to(self.dtype)
        coarse.count = count

    def _update_fine(
        self,
        entry: MultiPrototypeEntry,
        embedding: torch.Tensor,
        metadata: Optional[Dict] = None,
    ) -> Optional[tuple]:
        """Update fine prototype slots. Returns (action, slot_idx) when a slot is created or
        reinitialized (useful for provenance logging), or None on a plain EMA update."""
        if self.num_fine <= 0:
            return None
        slots = entry.fine
        if len(slots) > self.num_fine:
            entry.fine = slots = slots[: self.num_fine]
        if len(slots) < self.num_fine:
            if self.init_strategy == "first_k" or not slots:
                slot_idx = len(slots)
                slots.append(PrototypeEntry(vector=embedding.to(self.dtype), count=1, source=metadata))
                return ("opened", slot_idx)
            sims = [self._cosine(slot.vector, embedding) for slot in slots]
            if max(sims, default=-1.0) < self.init_sim_thresh:
                slot_idx = len(slots)
                slots.append(PrototypeEntry(vector=embedding.to(self.dtype), count=1, source=metadata))
                return ("opened", slot_idx)
        if not slots:
            slots.append(PrototypeEntry(vector=embedding.to(self.dtype), count=1, source=metadata))
            return ("opened", 0)
        sims = torch.tensor([self._cosine(slot.vector, embedding) for slot in slots], device=self.device)
        best_idx = int(torch.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim < self.gate_min_cos:
            if len(slots) < self.num_fine:
                slot_idx = len(slots)
                slots.append(PrototypeEntry(vector=embedding.to(self.dtype), count=1, source=metadata))
                return ("opened", slot_idx)
            replace_idx = min(range(len(slots)), key=lambda idx: slots[idx].count)
            slots[replace_idx] = PrototypeEntry(vector=embedding.to(self.dtype), count=1, source=metadata)
            return ("reinitialized", replace_idx)
        slot = slots[best_idx]
        vec = slot.vector.to(torch.float32)
        count = slot.count
        if count < self.capacity:
            count += 1
            vec = vec + (embedding - vec) / count
        else:
            alpha = self.ema_alpha if self.ema_alpha > 0 else (1.0 / float(self.capacity))
            vec = vec * (1.0 - alpha) + embedding * alpha
        slot.vector = F.normalize(vec, dim=0, eps=1e-6).to(self.dtype)
        slot.count = count
        return None

    def _cosine(self, proto: torch.Tensor, embedding: torch.Tensor) -> float:
        v = proto.to(torch.float32)
        return float(torch.clamp(torch.dot(v, embedding), -1.0, 1.0))

    def _merge_entry(self, level: str, cls_id: int, source_entry: MultiPrototypeEntry) -> int:
        total = int(source_entry.coarse.count) + sum(int(slot.count) for slot in source_entry.fine)
        bucket = self._storage[level]
        target_entry = bucket.get(cls_id)
        if target_entry is None:
            bucket[cls_id] = self._clone_entry(source_entry)
            return total
        self._merge_prototype_entry(target_entry, source_entry)
        return total

    def _clone_entry(self, entry: MultiPrototypeEntry) -> MultiPrototypeEntry:
        coarse_vec = entry.coarse.vector.to(self.device, dtype=self.dtype).clone()
        clone = MultiPrototypeEntry(coarse=PrototypeEntry(vector=coarse_vec, count=int(entry.coarse.count)), fine=[])
        if self.num_fine > 0:
            for slot in entry.fine[: self.num_fine]:
                if slot.vector is None or slot.vector.numel() == 0:
                    continue
                clone.fine.append(
                    PrototypeEntry(
                        vector=slot.vector.to(self.device, dtype=self.dtype).clone(),
                        count=int(slot.count),
                    )
                )
        return clone

    def _merge_prototype_entry(self, target: MultiPrototypeEntry, source: MultiPrototypeEntry) -> None:
        if self.use_coarse and source.coarse.vector is not None and source.coarse.vector.numel() > 0:
            self._merge_single_prototype(target.coarse, source.coarse)
        if self.num_fine <= 0:
            return
        for slot in source.fine:
            if slot.vector is None or slot.vector.numel() == 0 or slot.count <= 0:
                continue
            vec = slot.vector.to(torch.float32)
            count = int(slot.count)
            if len(target.fine) < self.num_fine:
                target.fine.append(
                    PrototypeEntry(vector=F.normalize(vec, dim=0, eps=1e-6).to(self.dtype), count=count)
                )
                continue
            sims = torch.tensor(
                [self._cosine(existing.vector, vec) for existing in target.fine],
                device=self.device,
            )
            best_idx = int(torch.argmax(sims)) if len(target.fine) else 0
            best_sim = float(sims[best_idx]) if len(target.fine) else 1.0
            if best_sim < self.gate_min_cos:
                replace_idx = min(range(len(target.fine)), key=lambda idx: target.fine[idx].count)
                target.fine[replace_idx] = PrototypeEntry(
                    vector=F.normalize(vec, dim=0, eps=1e-6).to(self.dtype),
                    count=count,
                )
                continue
            merged_vec, merged_count = self._merge_vectors(
                target.fine[best_idx].count,
                target.fine[best_idx].vector,
                count,
                vec,
            )
            target.fine[best_idx].vector = merged_vec
            target.fine[best_idx].count = merged_count

    def _merge_single_prototype(self, target_proto: PrototypeEntry, source_proto: PrototypeEntry) -> None:
        if source_proto.vector is None or source_proto.vector.numel() == 0 or source_proto.count <= 0:
            return
        if target_proto.vector is None or target_proto.vector.numel() == 0:
            target_proto.vector = source_proto.vector.to(self.device, dtype=self.dtype).clone()
            target_proto.count = int(source_proto.count)
            return
        merged_vec, merged_count = self._merge_vectors(
            target_proto.count,
            target_proto.vector,
            source_proto.count,
            source_proto.vector,
        )
        target_proto.vector = merged_vec
        target_proto.count = merged_count

    def _merge_vectors(
        self,
        count_a: int,
        vec_a: torch.Tensor,
        count_b: int,
        vec_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        total = int(count_a) + int(count_b)
        if total <= 0:
            return vec_a.to(self.dtype), int(count_a)
        merged = vec_a.to(torch.float32) * float(count_a) + vec_b.to(torch.float32) * float(count_b)
        merged = F.normalize(merged, dim=0, eps=1e-6)
        return merged.to(self.dtype), total

    # ------------------------------------------------------------------ Persistence helpers
    def state_dict(self) -> Dict:
        storage = {
            level: {
                int(cls): {
                    "coarse": {
                        "prototype": entry.coarse.vector.cpu(),
                        "count": int(entry.coarse.count),
                    },
                    "fine": [
                        {"prototype": slot.vector.cpu(), "count": int(slot.count), "source": slot.source}
                        for slot in entry.fine
                    ],
                }
                for cls, entry in class_map.items()
            }
            for level, class_map in self._storage.items()
        }
        return {
            "capacity": self.capacity,
            "dtype": str(self.dtype),
            "storage": storage,
        }

    def load_state_dict(self, state: Dict) -> int:
        self._storage.clear()
        storage = state.get("storage", {})
        total = 0
        # Detect legacy format (class -> list of items)
        legacy_format = storage and all(isinstance(v, list) for v in storage.values())
        if legacy_format:
            for cls, entries in storage.items():
                cls_id = int(cls)
                for entry in entries:
                    level = entry.get("level", "P3")
                    embedding = entry.get("embedding", torch.empty(0))
                    tensor = embedding.to(self.device, dtype=self.dtype)
                    self.add(TinyReplayItem(cls=cls_id, level=level, embedding=tensor))
                    total += 1
            return total

        for level, class_map in storage.items():
            for cls, data in class_map.items():
                cls_id = int(cls)
                coarse_data = data.get("coarse", data)
                proto_tensor = coarse_data.get("prototype", torch.empty(0))
                proto = proto_tensor.to(self.device, dtype=self.dtype)
                count = int(coarse_data.get("count", 1))
                if proto.numel() == 0:
                    continue
                entry = MultiPrototypeEntry(coarse=PrototypeEntry(vector=proto, count=count), fine=[])
                fine_list = data.get("fine", [])
                for slot in fine_list:
                    vec = slot.get("prototype", torch.empty(0)).to(self.device, dtype=self.dtype)
                    if vec.numel() == 0:
                        continue
                    entry.fine.append(PrototypeEntry(vector=vec, count=int(slot.get("count", 1)), source=slot.get("source")))
                    total += int(slot.get("count", 1))
                total += count
                self._storage[level][cls_id] = entry
        return total

    def save_sources(self, path: str | Path) -> None:
        """Persist fine prototype source metadata to a JSON file.

        For each (level, class, slot_index) triple, records the image path and
        bounding box of the sample that last opened or reinitialized that slot.
        Slots that have never been opened appear as null in the output list.
        """
        import json

        out: Dict[str, Dict[str, list]] = {}
        for level, class_map in self._storage.items():
            out[level] = {}
            for cls_id, entry in class_map.items():
                slots: list = []
                for slot in entry.fine:
                    slots.append(slot.source if slot.source else None)
                if slots:
                    out[level][str(cls_id)] = slots
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as f:
            json.dump(out, f, indent=2)

    def save(self, path: str | Path) -> int:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        state = self.state_dict()
        torch.save(state, target)
        return len(self)

    def load(self, path: str | Path, allow_growth: bool = True) -> int:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)
        state = torch.load(source, map_location="cpu")
        storage = state.get("storage", {})
        if allow_growth and storage:
            max_bucket = max(len(entries) for entries in storage.values())
            growth_limit = int(math.ceil(max_bucket * self.carryover_growth))
            self.capacity = max(self.capacity, growth_limit)
        return self.load_state_dict(state)


def build_replay_batch(
    buffer: TinyReplayBuffer,
    per_class: int,
    balance_levels: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[int, Dict[str, torch.Tensor]]]:
    """Return stored prototypes per level/class for downstream losses."""

    return buffer.collect_prototypes(levels=balance_levels, device=device)


def extract_tiny_embeddings(
    features: Dict[str, torch.Tensor],
    boxes: torch.Tensor,
    classes: torch.Tensor,
    batch_indices: torch.Tensor,
    strides: Dict[str, int],
    max_edge: float,
    image_hw: Tuple[int, int] = (640, 640),
) -> List[TinyReplayItem]:
    """
    Extract detection head pre-logit embeddings for tiny objects by pooling feature crops.
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
            tiny_emb = adaptive_pool(tiny_patch)

            items.append(
                TinyReplayItem(
                    cls=int(cls),
                    level=level,
                    embedding=tiny_emb,
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
