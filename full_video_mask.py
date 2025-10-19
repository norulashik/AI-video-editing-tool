import cv2
import os
import numpy as np
from scripts.generate_mask_sam import load_sam_model, generate_mask_from_numpy


def process_full_video_with_masks(
    video_path,
    output_video_path="output/full_masked_video.mp4",
    fps_out=10,
    resize_width=None,
):
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Cannot open video.")
        return

    predictor = load_sam_model()
    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if resize_width:
            scale = resize_width / frame.shape[1]
            frame = cv2.resize(frame, (resize_width, int(frame.shape[0] * scale)))

        try:
            mask = generate_mask_from_numpy(frame, predictor)
        except Exception as e:
            print(f"[!] Skipping frame {frame_count}: {e}")
            continue

        # Overlay mask on frame
        overlay = frame.copy()
        overlay[mask > 128] = [0, 0, 255]
        blended = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)

        frames.append(blended)

    cap.release()

    if not frames:
        print("[!] No frames processed.")
        return

    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (w, h)
    )

    for f in frames:
        out.write(f)

    out.release()
    print(f"[🎞] Full masked video saved to {output_video_path}")
