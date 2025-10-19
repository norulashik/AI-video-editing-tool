# scripts/generate_mask_sam.py

import os
import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor


def load_sam_model(model_type="vit_b", checkpoint_path="sam_vit_b.pth"):
    print("[INFO] Loading SAM model...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    return SamPredictor(sam)


def generate_mask_for_image(image_path, predictor, output_path):
    """
    Generate a segmentation mask from an image file path and save it.
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width, _ = image.shape
    center_point = np.array([[width // 2, height // 2]])  # middle point

    predictor.set_image(image_rgb)
    masks, scores, _ = predictor.predict(
        point_coords=center_point,
        point_labels=np.array([1]),
        multimask_output=True,
    )

    best_mask = masks[0]  # just pick the first mask
    mask_img = (best_mask * 255).astype(np.uint8)
    cv2.imwrite(output_path, mask_img)
    print(f"[✓] Saved mask: {output_path}")

    return mask_img, image  # for optional overlay previews


def generate_mask_from_numpy(image_np, predictor):
    """
    Generate a mask from a NumPy image array using the SAM predictor.
    """
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    input_point = np.array([[image_rgb.shape[1] // 2, image_rgb.shape[0] // 2]])
    input_label = np.array([1])

    predictor.set_image(image_rgb)
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    best_mask = masks[np.argmax(scores)].astype(np.uint8) * 255
    return best_mask
