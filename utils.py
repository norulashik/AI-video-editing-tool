# scripts/utils.py

import os


def validate_video_path(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[ERROR] Video file not found: {path}")


def create_output_folder(path: str):
    os.makedirs(path, exist_ok=True)
