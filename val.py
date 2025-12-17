import argparse
import os
from ednet import EDNet

def parse_args():
    parser = argparse.ArgumentParser(description="Validate EDNet model")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained weights (.pt)")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for inference")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to validate on")
    parser.add_argument("--outdir", type=str, default="validation_results", help="Folder to save validation results")
    return parser.parse_args()

def main(opt):
    # Ensure results directory exists
    os.makedirs(opt.outdir, exist_ok=True)

    # Load trained model
    model = EDNet(opt.weights)
    # Disable Conv+BN fusion for LoRA checkpoints to avoid attribute errors
    try:
        if hasattr(model, "model") and hasattr(model.model, "args"):
            model.model.args.fuse = False
    except AttributeError:
        pass

    # Run validation
    results = model.val(
        data=opt.data,
        imgsz=opt.imgsz,
        split=opt.split,
        project=opt.outdir,
        fuse=False   # save logs, plots, etc. here
    )

    # Save metrics to a file
    results_file = os.path.join(opt.outdir, "metrics.txt")
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    metrics_attr = getattr(results, "results_dict", {})
    metrics_dict = metrics_attr() if callable(metrics_attr) else (metrics_attr or {})
    with open(results_file, "w") as f:
        for k, v in metrics_dict.items():
            f.write(f"{k}: {v}\n")

    print(f"✅ Validation finished. Results saved to {opt.outdir}")

if __name__ == "__main__":
    opt = parse_args()
    main(opt)
