# main.py
import argparse
from rokey_pipeline import process_video_with_rokey


def main():
    parser = argparse.ArgumentParser(description="🔧 RotoKey CLI")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default="keyframes")
    parser.add_argument("--threshold", type=float, default=30.0)
    parser.add_argument("--generate-masks", action="store_true")
    parser.add_argument("--make-video", action="store_true")
    parser.add_argument("--full-mask-video", action="store_true")
    parser.add_argument("--remove-mask", action="store_true")
    parser.add_argument("--genai-bg", type=str)

    args = parser.parse_args()

    if args.full_mask_video:
        from scripts.full_video_mask import process_full_video_with_masks

        process_full_video_with_masks(args.video)
        return

    process_video_with_rokey(args)


if __name__ == "__main__":
    main()
