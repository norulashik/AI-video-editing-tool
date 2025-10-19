import cv2
import numpy as np
import os


def preprocess_mask(mask, dilate_iter=1, blur_size=7):
    """Optional: expand and smooth mask for better inpainting."""
    kernel = np.ones((5, 5), np.uint8)

    # Dilate the mask to give inpainting more area
    mask = cv2.dilate(mask, kernel, iterations=dilate_iter)

    # Apply Gaussian blur for smoother transition
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

    # Re-binarize to keep it usable
    _, mask = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)
    return mask


def remove_object_with_mask(image_path, mask_path, output_path, dilate=True):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"[ERROR] Could not load: {image_path} or {mask_path}")
        return

    if dilate:
        mask = preprocess_mask(mask)

    # Inpaint with TELEA method (or try NS for alternative)
    inpainted = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, inpainted)
    print(f"[✓] Saved object-removed image: {output_path}")
