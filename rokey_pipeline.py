# rokey_pipeline.py
import os
from scripts.extract_keyframes import extract_keyframes
from scripts.utils import validate_video_path, create_output_folder
from scripts.generate_mask_sam import load_sam_model, generate_mask_for_image
from scripts.make_video_from_masks import create_video_from_keyframes_and_masks
from scripts.full_video_mask import process_full_video_with_masks
from scripts.remove_mask import remove_object_with_mask
from scripts.genai_background import (
    generate_background,
    composite_subject_on_background,
)


def generate_masks_all(keyframe_dir="keyframes", mask_out_dir="masks"):
    os.makedirs(mask_out_dir, exist_ok=True)
    predictor = load_sam_model()

    for filename in os.listdir(keyframe_dir):
        if filename.lower().endswith(".jpg"):
            input_path = os.path.join(keyframe_dir, filename)
            output_path = os.path.join(
                mask_out_dir, filename.replace(".jpg", "_mask.png")
            )
            generate_mask_for_image(input_path, predictor, output_path)


def remove_objects_from_keyframes(
    keyframe_dir="keyframes", mask_dir="masks", output_dir="removed"
):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(keyframe_dir):
        if filename.lower().endswith(".jpg"):
            base_name = filename.replace(".jpg", "")
            image_path = os.path.join(keyframe_dir, filename)
            mask_path = os.path.join(mask_dir, f"{base_name}_mask.png")
            output_path = os.path.join(output_dir, f"{base_name}_removed.jpg")

            if os.path.exists(mask_path):
                remove_object_with_mask(image_path, mask_path, output_path)
            else:
                print(f"[WARN] Mask not found for {filename}, skipping.")


def process_video_with_rokey(args):
    validate_video_path(args.video)
    create_output_folder(args.out)

    print("[PHASE 1] Extracting Keyframes...")
    extract_keyframes(args.video, args.out, args.threshold)

    if args.generate_masks:
        print("[PHASE 2] Generating Masks...")
        generate_masks_all(args.out, "masks")

    if args.remove_mask:
        print("[PHASE 3] Removing Masked Objects...")
        remove_objects_from_keyframes(args.out, "masks", "removed")

    if args.genai_bg:
        print("[PHASE 4] Generating AI Background and Compositing...")
        bg_path = generate_background(args.genai_bg, save_path="output/ai_bg.png")

        if not bg_path or not os.path.exists(bg_path):
            print("[✗] Skipping compositing due to background generation failure.")
        else:
            os.makedirs("output/genai_composite", exist_ok=True)
            for filename in os.listdir(args.out):
                if filename.endswith(".jpg"):
                    base = filename.replace(".jpg", "")
                    subject_path = os.path.join(args.out, filename)
                    mask_path = os.path.join("masks", f"{base}_mask.png")
                    output_path = os.path.join(
                        "output/genai_composite", f"{base}_composite.jpg"
                    )
                    if os.path.exists(mask_path):
                        composite_subject_on_background(
                            subject_path, mask_path, bg_path, output_path
                        )
                    else:
                        print(f"[WARN] Missing mask for {base}, skipping.")

    if args.make_video:
        print("[PHASE 5] Creating Mask Overlay Video...")
        create_video_from_keyframes_and_masks(
            args.out, "masks", "output/masked_video.mp4"
        )
