import os
import glob
from PIL import Image

# === CONFIGURATION ===
# Adjust these paths as needed
ann_dir = 'datasets/visdrone/VisDrone2019-DET-val/annotations'
img_dir = 'datasets/visdrone/VisDrone2019-DET-val/images'
out_dir = 'datasets/visdrone/VisDrone2019-DET-val/labels'

# Create output directory if it doesn't exist
os.makedirs(out_dir, exist_ok=True)

# Get all annotation files
ann_files = glob.glob(os.path.join(ann_dir, '*.txt'))

print(f"Found {len(ann_files)} annotation files.")

for ann_path in ann_files:
    base_name = os.path.splitext(os.path.basename(ann_path))[0]
    img_path = os.path.join(img_dir, base_name + '.jpg')
    out_path = os.path.join(out_dir, base_name + '.txt')

    # Skip if corresponding image does not exist
    if not os.path.exists(img_path):
        print(f"Skipping {base_name}: image not found.")
        continue

    # Load image to get width/height
    try:
        with Image.open(img_path) as im:
            img_w, img_h = im.size
    except Exception as e:
        print(f"Error opening image {img_path}: {e}")
        continue

    yolo_labels = []

    with open(ann_path, 'r') as f:
        for line in f:
            fields = line.strip().split(',')
            if len(fields) < 8:
                continue

            if len(fields) != 8 or '' in fields:
                continue  # skip incomplete or invalid lines
            try:
                x, y, w, h, score, cls_id, trunc, occ = map(int, fields)
            except ValueError:
                continue  # skip if conversion fails


            # Skip ignored objects
            if score != 1 or cls_id == 0 or cls_id == 11:
                continue

            # Convert to YOLO format
            xc = (x + w / 2) / img_w
            yc = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
            cls_id -= 1  # YOLO classes should start at 0

            yolo_labels.append(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

    # Write YOLO label file
    with open(out_path, 'w') as out_file:
        out_file.writelines(yolo_labels)

    print(f"Converted {base_name}.txt → {len(yolo_labels)} objects")
