import os
import cv2
from glob import glob

# === CONFIGURATION ===
input_video_dir = "datasets/drone_beelden/download"
output_image_dir = "datasets/drone_beelden/images"
fps_interval = 1  # seconds between saved frames

# === Ensure output folder exists ===
os.makedirs(output_image_dir, exist_ok=True)

# === Process each video ===
def video_to_frames(video_path, output_dir, fps_interval=4):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        print(f"⚠️ Could not read FPS from {video_path}")
        return

    frame_interval = int(fps * fps_interval)
    success, image = vidcap.read()
    count = 0
    saved = 0

    while success:
        if count % frame_interval == 0:
            filename = f"{video_name}_frame_{saved:05d}.jpg"
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, image)
            saved += 1
        success, image = vidcap.read()
        count += 1

    vidcap.release()
    print(f"✅ {saved} frames extracted from {video_name}")

# === Main loop over all .mp4 files ===
video_files = glob(os.path.join(input_video_dir, "*.MP4"))

if not video_files:
    print("⚠️ No .mp4 files found in", input_video_dir)
else:
    for video_file in video_files:
        video_to_frames(video_file, output_image_dir, fps_interval=fps_interval)
