"""Utility to inspect cosine alignment of stored replay embeddings."""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F

from ednet.engine.replay import TinyReplayBuffer


def collect_embeddings(buffer: TinyReplayBuffer):
    """Yield (level, class_id, tensor[N, D]) for every buffer bucket."""
    grouped = defaultdict(lambda: defaultdict(list))
    for cls_id, items in buffer._storage.items():  # pylint: disable=protected-access
        for item in items:
            grouped[item.level][cls_id].append(item.embedding.float())
    for level, cls_map in grouped.items():
        for cls_id, tensors in cls_map.items():
            yield level, cls_id, torch.stack(tensors)


def cosine_stats(vectors: torch.Tensor) -> dict[str, float]:
    normed = F.normalize(vectors, dim=1)
    proto = F.normalize(normed.mean(dim=0, keepdim=True), dim=1)
    cosine = (normed * proto).sum(dim=1)
    return {
        "count": int(vectors.shape[0]),
        "dim": int(vectors.shape[1]),
        "mean_norm": float(vectors.norm(dim=1).mean()),
        "mean": float(cosine.mean()),
        "median": float(cosine.median()),
        "min": float(cosine.min()),
        "max": float(cosine.max()),
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect replay buffer cosine alignment")
    parser.add_argument("buffer", type=str, help="Path to replay buffer .pt file")
    parser.add_argument("--min_samples", type=int, default=2, help="Skip classes with fewer samples")
    args = parser.parse_args()

    buffer = TinyReplayBuffer()
    buffer.load(args.buffer)

    any_printed = False
    for level, cls_id, tensors in collect_embeddings(buffer):
        if tensors.shape[0] < args.min_samples:
            continue
        stats = cosine_stats(tensors)
        print(
            f"level={level:>3} cls={cls_id:>2} count={stats['count']:>4} dim={stats['dim']:>4} "
            f"mean_norm={stats['mean_norm']:.3f} mean={stats['mean']:.3f} "
            f"median={stats['median']:.3f} min={stats['min']:.3f} max={stats['max']:.3f}"
        )
        any_printed = True

    if not any_printed:
        print("No classes met the min_samples threshold; try lowering --min_samples")


if __name__ == "__main__":
    main()
