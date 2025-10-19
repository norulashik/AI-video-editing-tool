# scripts/extract_keyframes.py

import os
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import cv2


def extract_keyframes(video_path: str, output_dir: str, threshold: float = 30.0):
    os.makedirs(output_dir, exist_ok=True)

    # Create video and scene manager
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    # Start scene detection
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()

    cap = cv2.VideoCapture(video_path)

    # If no scenes detected, fallback to single mid-frame
    if len(scene_list) == 0:
        print("[INFO] No scenes detected. Extracting a single fallback keyframe.")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid_frame = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        if ret:
            out_path = os.path.join(output_dir, "scene_1.jpg")
            cv2.imwrite(out_path, frame)
            print(f"[✓] Saved fallback keyframe: {out_path}")
        else:
            print("[!] Failed to extract fallback frame.")
        cap.release()
        return

    print(f"[INFO] Found {len(scene_list)} scenes in the video.")

    for i, (start, end) in enumerate(scene_list):
        mid_frame_num = (start.get_frames() + end.get_frames()) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_num)
        ret, frame = cap.read()
        if ret:
            out_path = os.path.join(output_dir, f"scene_{i + 1}.jpg")
            cv2.imwrite(out_path, frame)
            print(f"[✓] Saved keyframe: {out_path}")
        else:
            print(f"[!] Failed to read frame at {mid_frame_num}")

    cap.release()
