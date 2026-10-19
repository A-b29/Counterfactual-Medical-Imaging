"""
image_io.py - Image input/output and preprocessing for Medical Imaging Counterfactual Lab.

Ensures strict normalization to float32 [0.0, 1.0] with channel consistency.
"""

from pathlib import Path
from typing import Union, Tuple
import numpy as np
import cv2
from PIL import Image


def load_image(
    file_path: Union[str, Path],
    target_size: Tuple[int, int] = (256, 256),
    as_gray: bool = False
) -> np.ndarray:
    """
    Load an image from disk, validate dimensions, resize, and normalize to [0.0, 1.0] float32.

    Args:
        file_path: Path to the image file.
        target_size: Tuple of (height, width) for resizing.
        as_gray: If True, convert image to single-channel (H, W, 1). If False, ensure (H, W, 3).

    Returns:
        np.ndarray: float32 image normalized in [0.0, 1.0].
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {file_path}")

    # Use cv2 with fallback to PIL
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        try:
            pil_img = Image.open(path)
            img = np.array(pil_img)
        except Exception as e:
            raise ValueError(f"Failed to decode image from {file_path}: {e}")

    # Normalize dtype to float32 [0.0, 1.0]
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    elif np.issubdtype(img.dtype, np.floating):
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / img.max()
        img = np.clip(img, 0.0, 1.0)
    else:
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Dimensionality adjustment
    if as_gray:
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
    else:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.ndim == 3 and img.shape[2] == 3:
            # OpenCV loads as BGR by default
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to target
    if (img.shape[0], img.shape[1]) != target_size:
        if img.shape[2] == 1:
            resized = cv2.resize(img[:, :, 0], (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
            img = np.expand_dims(resized, axis=-1)
        else:
            img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)

    return np.clip(img, 0.0, 1.0).astype(np.float32)


def save_image(img: np.ndarray, output_path: Union[str, Path]) -> None:
    """Save normalized float32 [0.0, 1.0] image to disk as 8-bit PNG."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    clipped = np.clip(img, 0.0, 1.0)
    uint8_img = (clipped * 255.0).astype(np.uint8)

    if uint8_img.ndim == 3 and uint8_img.shape[2] == 3:
        # Convert RGB to BGR for cv2
        uint8_img = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(path), uint8_img)
