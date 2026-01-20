"""Utility to inspect stored replay prototypes."""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from ednet.engine.replay import TinyReplayBuffer


def main():
    parser = argparse.ArgumentParser(description="Inspect replay buffer prototypes")
    parser.add_argument("buffer", type=str, help="Path to replay buffer .pt file")
    parser.add_argument("--min_samples", type=int, default=2, help="Skip classes with fewer samples")
    parser.add_argument("--show-cos", action="store_true", help="Show pairwise cosine of prototypes")
    args = parser.parse_args()

    buffer = TinyReplayBuffer()
    buffer.load(args.buffer)

    any_printed = False
    proto_map = buffer.collect_prototypes()
    for level, class_map in proto_map.items():
        for cls_id, data in class_map.items():
            counts = data["counts"]
            if counts.sum() < args.min_samples:
                continue
            protos = data["prototypes"]
            count_str = ", ".join(f"{int(c):>4}" for c in counts.tolist())
            print(
                f"level={level:>3} cls={cls_id:>2} slots={protos.shape[0]:>2} dim={protos.shape[1]:>4} "
                f"counts=[{count_str}]"
            )
            any_printed = True
            if args.show_cos and protos.shape[0] > 1:
                normed = F.normalize(protos, dim=1)
                cos = normed @ normed.t()
                upper = torch.triu(cos, diagonal=1)
                if torch.any(upper != 0):
                    vals = upper[upper != 0]
                    min_cos = float(vals.min())
                    max_cos = float(vals.max())
                else:
                    min_cos = max_cos = math.nan
                print(f"  pairwise cos min={min_cos:.3f} max={max_cos:.3f}")

    if not any_printed:
        print("No classes met the min_samples threshold; try lowering --min_samples")


if __name__ == "__main__":
    main()
