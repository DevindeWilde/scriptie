from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ednet.models.yolo.detect.train import DetectionTrainer
from ednet.utils import LOGGER


class ContinualDetectionPipeline:
    """
    Simple orchestrator that reuses DetectionTrainer for continual learning.

    It sequentially streams dataset batches, optionally carrying over LoRA adapters between
    runs so that new data can be assimilated without retraining from scratch.
    """

    def __init__(
        self,
        base_overrides: Optional[Dict] = None,
        trainer_cls=DetectionTrainer,
    ) -> None:
        self.base_overrides = base_overrides or {}
        self.trainer_cls = trainer_cls

    @staticmethod
    def _merge_overrides(base: Dict, update: Dict) -> Dict:
        """Recursively merge override dictionaries while preserving nested structures."""
        merged = deepcopy(base)
        for key, value in (update or {}).items():
            if key == "lora":
                base_lora = deepcopy(merged.get("lora", {}))
                if value:
                    base_lora.update(value)
                merged["lora"] = base_lora
            elif isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = ContinualDetectionPipeline._merge_overrides(merged[key], value)
            else:
                merged[key] = value
        return merged

    def run(self, batch_sequence: Iterable[Dict]) -> List[Dict]:
        """
        Iterate over data batches, training adapters sequentially while reusing previous checkpoints.

        Args:
            batch_sequence (Iterable[Dict]): Iterable of override dictionaries. Each entry can update any
                trainer argument for that batch (e.g., data path, epochs, learning rate, etc.).

        Returns:
            List[Dict]: Metadata for each batch including save directories and adapter checkpoints.
        """
        results: List[Dict] = []
        adapter_path: Optional[str] = self.base_overrides.get("lora", {}).get("init_adapter")

        for batch_idx, overrides in enumerate(batch_sequence):
            merged_overrides = self._merge_overrides(self.base_overrides, overrides or {})
            lora_cfg = merged_overrides.get("lora", {})
            if lora_cfg.get("enable"):
                if adapter_path and "init_adapter" not in lora_cfg:
                    lora_cfg["init_adapter"] = adapter_path
                merged_overrides["lora"] = lora_cfg

            trainer = self.trainer_cls(overrides=merged_overrides)
            LOGGER.info(f"🚀 Continual batch {batch_idx}: training with data={trainer.args.data}")
            trainer.train()

            batch_result = {
                "batch_index": batch_idx,
                "save_dir": str(trainer.save_dir),
                "adapter_path": None,
            }

            if lora_cfg.get("enable"):
                adapters_dir = Path(trainer.save_dir) / (lora_cfg.get("adapter_dir") or "adapters")
                strategy = (
                    (merged_overrides.get("continual") or {}).get("adapter_strategy")
                    or self.base_overrides.get("continual", {}).get("adapter_strategy")
                    or "last"
                )
                candidate_name = "best-adapter.pt" if strategy == "best" else "last-adapter.pt"
                candidate = adapters_dir / candidate_name
                if candidate.exists():
                    adapter_path = str(candidate)
                    batch_result["adapter_path"] = adapter_path
                    LOGGER.info(f"  ↳ Adapter for next batch: {candidate}")
                else:
                    LOGGER.warning(f"  ↳ Expected adapter checkpoint not found at {candidate}")

            results.append(batch_result)

        return results


def run_continual_training(
    base_overrides: Optional[Dict] = None,
    batch_sequence: Optional[Iterable[Dict]] = None,
    trainer_cls=DetectionTrainer,
) -> List[Dict]:
    """
    Convenience wrapper to execute continual learning directly from configuration dictionaries.
    """
    pipeline = ContinualDetectionPipeline(base_overrides=base_overrides or {}, trainer_cls=trainer_cls)
    continual_cfg = (base_overrides or {}).get("continual", {}) if base_overrides else {}
    sequence = batch_sequence or continual_cfg.get("batches", [])
    return pipeline.run(sequence)
