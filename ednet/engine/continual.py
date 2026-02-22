from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ednet.models.yolo.detect.train import DetectionTrainer
from ednet.utils import LOGGER


class ContinualDetectionPipeline:
    """
    Simple orchestrator that reuses DetectionTrainer for continual learning.

    It sequentially streams dataset batches, carrying over weights between stages
    so that new data can be assimilated without retraining from scratch.
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
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
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

        for batch_idx, overrides in enumerate(batch_sequence):
            merged_overrides = self._merge_overrides(self.base_overrides, overrides or {})

            trainer = self.trainer_cls(overrides=merged_overrides)
            LOGGER.info(f"🚀 Continual batch {batch_idx}: training with data={trainer.args.data}")
            trainer.train()

            batch_result = {
                "batch_index": batch_idx,
                "save_dir": str(trainer.save_dir),
            }

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
