import math
from typing import Dict, Iterator

import torch
import torch.nn as nn


class LoRAConv2d(nn.Module):
    """
    Lightweight Low-Rank Adapter for 2D convolutions.

    The module wraps a frozen convolutional layer and adds a trainable low-rank update that is
    applied during the forward pass. This enables parameter-efficient fine-tuning where only the
    low-rank adapters are optimized while the original weights remain fixed.
    """

    def __init__(
        self,
        base_layer: nn.Conv2d,
        rank: int = 4,
        alpha: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        """
        Args:
            base_layer (nn.Conv2d): Convolutional layer to augment with LoRA.
            rank (int): Intrinsic rank of the low-rank update.
            alpha (float): Scaling factor applied to the LoRA update (LoRA alpha).
            dropout (float): Dropout probability applied between A and B projections.
        """
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be a positive integer.")
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Clone the base convolution so weights remain in their original shape but are frozen.
        self.base = nn.Conv2d(
            in_channels=base_layer.in_channels,
            out_channels=base_layer.out_channels,
            kernel_size=base_layer.kernel_size,
            stride=base_layer.stride,
            padding=base_layer.padding,
            dilation=base_layer.dilation,
            groups=base_layer.groups,
            bias=base_layer.bias is not None,
        )
        self.base.load_state_dict(base_layer.state_dict())
        for param in self.base.parameters():
            param.requires_grad_(False)

        # LoRA decomposition: A projects to a low-rank space; B projects back to output channels.
        self.lora_A = nn.Conv2d(
            in_channels=base_layer.in_channels,
            out_channels=rank,
            kernel_size=base_layer.kernel_size,
            stride=base_layer.stride,
            padding=base_layer.padding,
            dilation=base_layer.dilation,
            groups=base_layer.groups,
            bias=False,
        )
        self.lora_B = nn.Conv2d(
            in_channels=rank,
            out_channels=base_layer.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        # Initialize so that the adapter produces zero output before training.
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.dropout = nn.Identity() if dropout == 0.0 else nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with frozen base convolution plus low-rank adaptation."""
        base_out = self.base(x)
        lora_update = self.lora_B(self.dropout(self.lora_A(x)))
        return base_out + self.scaling * lora_update

    def lora_parameters(self) -> Iterator[nn.Parameter]:
        """Returns an iterator over trainable LoRA parameters."""
        yield from self.lora_A.parameters()
        yield from self.lora_B.parameters()

    def lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Collect only the LoRA weights for checkpointing adapters separately from the base model.
        """
        return {
            "lora_A.weight": self.lora_A.weight.data.clone(),
            "lora_B.weight": self.lora_B.weight.data.clone(),
        }

    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Loads LoRA weights from a state dict produced by `lora_state_dict`."""
        if "lora_A.weight" in state_dict:
            self.lora_A.weight.data.copy_(state_dict["lora_A.weight"])
        if "lora_B.weight" in state_dict:
            self.lora_B.weight.data.copy_(state_dict["lora_B.weight"])

