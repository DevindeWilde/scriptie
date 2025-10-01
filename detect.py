import argparse
import os
from ednet import EDNet

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with EDNet")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained weights (.pt)")
    parser.add_argument("--source", type=str, nargs="+", required=True, help="Paths to input images")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--outdir", type=str, default="inference_results", help="Folder to save detections")
    return parser.parse_args()

def main(opt):
    os.makedirs(opt.outdir, exist_ok=True)

    # Load trained model
    model = EDNet(opt.weights)

    # Run inference
    results = model.predict(
        source=opt.source,
        imgsz=opt.imgsz,
        save=True,          # save images with bboxes
        project=opt.outdir,
        name="exp"
    )

    print(f"✅ Inference done. Check results in {opt.outdir}/exp")

if __name__ == "__main__":
    opt = parse_args()
    main(opt)
