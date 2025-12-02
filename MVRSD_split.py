import shutil
from pathlib import Path

data_root = Path("datasets/MVRSD_dataset/data")
stage2_root = Path("datasets/MVRSD-stage2")
stage3_root = Path("datasets/MVRSD-stage3")

# Map original MVRSD IDs (0=SMV,1=LMV,2=AFV,3=CV,4=MCV) to final IDs (10–13)
stage2_mapping = {0: 10, 1: 11}
stage3_mapping = {0: 10, 1: 11, 2: 12, 4: 13}  # skip 3=CV entirely

for stage_dir in (stage2_root, stage3_root):
    if stage_dir.exists():
        shutil.rmtree(stage_dir)

for stage_dir, mapping in (
    (stage2_root, stage2_mapping),
    (stage3_root, stage3_mapping),
):
    for split in ("train", "val", "test"):
        img_src = data_root / split / "images"
        lbl_src = data_root / split / "labels"
        img_dst = stage_dir / split / "images"
        lbl_dst = stage_dir / split / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        for label_file in lbl_src.glob("*.txt"):
            new_lines = []
            with label_file.open() as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls = int(parts[0])
                    if cls in mapping:
                        parts[0] = str(mapping[cls])
                        new_lines.append(" ".join(parts))
            if new_lines:
                dst_label = lbl_dst / label_file.name
                dst_label.write_text("\n".join(new_lines) + "\n")
                src_img = img_src / f"{label_file.stem}.jpg"
                if src_img.exists():
                    shutil.copy2(src_img, img_dst / src_img.name)
