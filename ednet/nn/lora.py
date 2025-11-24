from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ednet.nn.modules import C2f, Detect, LoRAConv2d, v10Detect


@dataclass
class LoRAConfig:
    """
    Configuration for integrating LoRA adapters into EDNet detection models.

    Attributes:
        rank (int): Intrinsic rank of the low-rank update.
        alpha (float): Scaling factor for the LoRA update.
        dropout (float): Dropout probability between LoRA projections.
        feature_pyramid_indices (Sequence[int]): Layer indices (as defined in the parsed model)
            within the feature pyramid where LoRA adapters should be injected.
        include_detection_head (bool): Whether to inject LoRA adapters into the detection head.
    """

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    feature_pyramid_indices: Sequence[int] = (16, 22, 25)
    include_detection_head: bool = True


def _wrap_conv_with_lora(module: nn.Module, attr: str, cfg: LoRAConfig, registry: List[Tuple[str, LoRAConv2d]]):
    """Replace a Conv module attribute with a LoRA-enabled convolution."""
    conv = getattr(module, attr, None)
    if conv is None:
        return
    if isinstance(conv, nn.Conv2d) and getattr(conv, "groups", 1) == 1:
        lora = LoRAConv2d(conv, rank=cfg.rank, alpha=cfg.alpha, dropout=cfg.dropout)
        setattr(module, attr, lora)
        registry.append((f"{module.__class__.__name__}.{attr}", lora))
    elif isinstance(conv, LoRAConv2d):
        registry.append((f"{module.__class__.__name__}.{attr}", conv))


def inject_lora_ednet(model: nn.Module, cfg: Optional[LoRAConfig] = None) -> List[Tuple[str, LoRAConv2d]]:
    """
    Inject LoRA adapters into the EDNet detection model.

    Args:
        model (nn.Module): DetectionModel instance containing the parsed architecture.
        cfg (LoRAConfig, optional): Configuration controlling where adapters are added.

    Returns:
        List[Tuple[str, LoRAConv2d]]: Registry of inserted adapters (name, module) for bookkeeping.
    """
    if cfg is None:
        cfg = LoRAConfig()

    if not hasattr(model, "model"):
        raise AttributeError("Expected model to have a `model` attribute containing the sequential layers.")

    registry: List[Tuple[str, LoRAConv2d]] = []

    # Inject adapters into the selected feature pyramid layers (P3–P4 regions defined in the YAML).
    for m in getattr(model, "model", []):
        if not hasattr(m, "i"):
            continue
        if isinstance(m, C2f) and m.i in cfg.feature_pyramid_indices:
            _wrap_conv_with_lora(m.cv1, "conv", cfg, registry)
            _wrap_conv_with_lora(m.cv2, "conv", cfg, registry)
            for idx, bottleneck in enumerate(m.m):
                if hasattr(bottleneck, "cv1"):
                    _wrap_conv_with_lora(bottleneck.cv1, "conv", cfg, registry)
                if hasattr(bottleneck, "cv2"):
                    _wrap_conv_with_lora(bottleneck.cv2, "conv", cfg, registry)

    # Optionally inject adapters into the detection head convolutions.
    if cfg.include_detection_head and isinstance(getattr(model, "model", [])[-1], Detect):
        head = model.model[-1]
        if isinstance(head, v10Detect):
            for name, seq in enumerate(head.cv2):
                for block in seq:
                    if isinstance(block, nn.Sequential):
                        for layer in block:
                            if hasattr(layer, "conv"):
                                _wrap_conv_with_lora(layer, "conv", cfg, registry)
                    elif hasattr(block, "conv"):
                        _wrap_conv_with_lora(block, "conv", cfg, registry)
            for name, seq in enumerate(head.cv3):
                for block in seq:
                    if isinstance(block, nn.Sequential):
                        for layer in block:
                            if hasattr(layer, "conv"):
                                _wrap_conv_with_lora(layer, "conv", cfg, registry)
                    elif hasattr(block, "conv"):
                        _wrap_conv_with_lora(block, "conv", cfg, registry)

    return registry


def freeze_model_except_lora(model: nn.Module, adapters: Iterable[Tuple[str, LoRAConv2d]]) -> None:
    """
    Freeze all model parameters except the LoRA adapters.
    """
    for param in model.parameters():
        param.requires_grad = False
    for _, adapter in adapters:
        for param in adapter.lora_parameters():
            param.requires_grad = True


def iter_lora_modules(model: nn.Module) -> Iterator[Tuple[str, LoRAConv2d]]:
    """Yield all LoRA adapters registered within the model."""
    for name, module in model.named_modules():
        if isinstance(module, LoRAConv2d):
            yield name, module


def lora_state_dict(model: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Build a state dict containing only the LoRA adapter weights.
    """
    return {name: module.lora_state_dict() for name, module in iter_lora_modules(model)}


def load_lora_state_dict(model: nn.Module, state_dict: Dict[str, Dict[str, torch.Tensor]]) -> None:
    """
    Load a state dict produced by `lora_state_dict` back into the model.
    """
    for name, weights in state_dict.items():
        module = dict(iter_lora_modules(model)).get(name)
        if module is None:
            continue
        module.load_lora_state_dict(weights)
