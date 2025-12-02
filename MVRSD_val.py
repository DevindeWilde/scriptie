import random
from pathlib import Path
import shutil

root = Path('datasets/MVRSD_dataset/data')
train_imgs = root / 'train/images'
train_labels = root / 'train/labels'
val_imgs = root / 'val/images'
val_labels = root / 'val/labels'

val_imgs.mkdir(parents=True, exist_ok=True)
val_labels.mkdir(parents=True, exist_ok=True)

images = sorted(train_imgs.glob('*.jpg'))
random.seed(42)
random.shuffle(images)

val_count = int(0.1 * len(images))
val_subset = images[:val_count]

for img in val_subset:
    lbl = train_labels / f"{img.stem}.txt"
    shutil.move(str(img), str(val_imgs / img.name))
    if lbl.exists():
        shutil.move(str(lbl), str(val_labels / lbl.name))
