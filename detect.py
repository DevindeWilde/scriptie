import matplotlib.pyplot as plt
from ednet import EDNet
from pathlib import Path

def main():
    # Load your trained model
    model = EDNet("runs/detect/train4/weights/best.pt")

    # List of images to test
    images = [
        "datasets/MVRSD_dataset/data/val/images/AUAU030334.jpg",
        "datasets/MVRSD_dataset/data/val/images/AUAU030336.jpg",
        "datasets/MVRSD_dataset/data/val/images/AUAU030340.jpg",
        "datasets/MVRSD_dataset/data/val/images/AUAU030350.jpg",
    ]

    # Run inference, save annotated images in runs/predict/exp/
    results = model(images, save=True, imgsz=640)

    # Plot results inline (optional, only works in notebooks/local Python)
    for i, result in enumerate(results):
        im = result.plot()   # get numpy array with bboxes
        plt.figure(figsize=(8, 8))
        plt.imshow(im)
        plt.axis("off")
        plt.title(f"Prediction {i+1}")
        plt.show()

if __name__ == "__main__":
    main()
