import cv2
import os
import numpy as np


def create_video_from_keyframes_and_masks(
    keyframe_dir="keyframes",
    mask_dir="masks",
    output_path="output/masked_video.mp4",
    fps=10,
    seconds_per_frame=5,  # 👈 each keyframe lasts 5 seconds
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    keyframes = sorted([f for f in os.listdir(keyframe_dir) if f.endswith(".jpg")])
    height, width = None, None
    frames = []

    for filename in keyframes:
        key_path = os.path.join(keyframe_dir, filename)
        mask_path = os.path.join(mask_dir, filename.replace(".jpg", "_mask.png"))

        key_img = cv2.imread(key_path)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if key_img is None or mask_img is None:
            print(f"[!] Skipping {filename} due to missing files.")
            continue

        # Resize mask to match key image
        mask_resized = cv2.resize(mask_img, (key_img.shape[1], key_img.shape[0]))

        # Apply mask as red overlay
        overlay = key_img.copy()
        overlay[mask_resized > 128] = [0, 0, 255]
        alpha = 0.5
        blended = cv2.addWeighted(key_img, 1 - alpha, overlay, alpha, 0)

        # Repeat frame to match desired duration
        repeat_count = fps * seconds_per_frame
        frames.extend([blended] * repeat_count)

        if height is None:
            height, width = blended.shape[:2]

    if not frames:
        print("[!] No valid frames to write. Video not created.")
        return

    # Write video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"[🎞] Saved mask-overlay video to: {output_path}")
