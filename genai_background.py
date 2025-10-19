import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import cv2
import os
import numpy as np
from PIL import Image

# === [SETUP] ===

# Use SD 1.5 (lighter, stable on CPU)
model_id = "runwayml/stable-diffusion-v1-5"

# Force CPU
device = "cpu"
dtype = torch.float32

print(f"[INFO] Loading model: {model_id} on CPU")

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=dtype,
    safety_checker=None,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)


# === [GENERATION] ===


def generate_background(prompt, size=(512, 512), save_path="output/ai_bg.png"):
    print(f"[INFO] Generating background for prompt: {prompt} (size={size})")
    result = pipe(prompt, height=size[1], width=size[0])

    image = result.images[0]
    np_img = np.array(image)

    if np_img.max() == 0 or np.isnan(np_img).any():
        print("[✗] Generated image is empty or invalid.")
        return None

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)
    print(f"[✓] Generated background saved to {save_path}")
    return save_path


# === [COMPOSITING] ===


def composite_subject_on_background(
    subject_path, mask_path, bg_path, output_path, size=(512, 512)
):
    subject = cv2.imread(subject_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    background = cv2.imread(bg_path)

    if subject is None or mask is None or background is None:
        print(
            f"[ERROR] Could not load image(s):\n  - {subject_path}\n  - {mask_path}\n  - {bg_path}"
        )
        return

    subject = cv2.resize(subject, size)
    mask = cv2.resize(mask, size)
    background = cv2.resize(background, size)

    # Debug: Save intermediate masks
    os.makedirs("output/debug", exist_ok=True)
    print("[DEBUG] Unique mask values:", np.unique(mask))
    cv2.imwrite("output/debug/resized_mask.png", mask)

    # Ensure binary mask
    _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(binary_mask)

    # Optional debug visuals
    cv2.imwrite("output/debug/binary_mask.png", binary_mask)
    cv2.imwrite("output/debug/inverted_mask.png", mask_inv)

    # Foreground and background blending
    fg = cv2.bitwise_and(subject, subject, mask=binary_mask)
    bg = cv2.bitwise_and(background, background, mask=mask_inv)
    final = cv2.add(fg, bg)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, final)
    print(f"[✓] Composite saved to {output_path}")
