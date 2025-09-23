import argparse
from ednet import EDNet

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
    return parser.parse_args()

def main(opt):
    # If weights are provided, load from them
    if opt.weights:
        model = EDNet(opt.weights)
    else:
        model = EDNet(opt.cfg)

    results = model.train(
        data=opt.data,
        epochs=opt.epochs,
        imgsz=opt.imgsz
    )

    print("Training finished. Results saved in:", results)

if __name__ == "__main__":
    opt = parse_args()
    main(opt)
