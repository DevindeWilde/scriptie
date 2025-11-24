import argparse
import ast
from copy import deepcopy

from ednet import EDNet


def _assign_nested(dct, keys, value):
    """Assign value into nested dict following keys list."""
    current = dct
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


def _parse_value(raw):
    """Best-effort conversion of CLI string to Python type."""
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered == "none":
        return None
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def parse_override_list(pairs):
    """Convert key=value CLI overrides (with dot notation) to nested dict."""
    overrides = {}
    for item in pairs or []:
        if "=" not in item:
            continue
        key, raw = item.split("=", 1)
        value = _parse_value(raw)
        keys = key.split(".")
        _assign_nested(overrides, keys, value)
    return overrides

def parse_args():
    parser = argparse.ArgumentParser(description="Train EDNet model")
    parser.add_argument("--cfg", type=str, required=True, help="Path to model config (.yaml or .pt)")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--weights", type=str, default="", help="Optional pretrained weights (.pt)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--project", type=str, default="runs/train", help="Output directory")
    parser.add_argument("--name", type=str, default="exp", help="Experiment name")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional training overrides in key=value format (supports dot notation)",
    )
    return parser.parse_args()

def main(opt):
    # If weights are provided, load from them
    if opt.weights:
        model = EDNet(opt.weights)
    else:
        model = EDNet(opt.cfg)

    overrides = parse_override_list(opt.override)
    train_params = {
        "data": opt.data,
        "epochs": opt.epochs,
        "imgsz": opt.imgsz,
        "batch": opt.batch_size,
        "workers": opt.workers,
        "project": opt.project,
        "name": opt.name,
    }
    train_params.update(overrides)
    results = model.train(**train_params)
    save_dir = getattr(model.trainer, "save_dir", None)
    if save_dir is not None:
        print("Training finished. Results saved in:", save_dir)
    else:
        print("Training finished. Metrics:", results)

if __name__ == "__main__":
    opt = parse_args()
    main(opt)
