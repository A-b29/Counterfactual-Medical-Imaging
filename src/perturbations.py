"""
perturbations.py - Pure, deterministic image transformations with explicit severity parameters.

Every function adheres to:
- Input: float32 image in [0.0, 1.0], severity in [0.0, 1.0]
- Identity guarantee: severity == 0.0 returns an exact copy of the input.
- Output: float32 image strictly clipped in [0.0, 1.0], matching input shape.
"""

from typing import Tuple, Optional, Callable, Dict
import numpy as np
import cv2


def gaussian_noise(
    image: np.ndarray,
    severity: float,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Add zero-mean Gaussian noise.
    At severity=1.0, std dev = 0.35 of normalized intensity.
    """
    if severity <= 0.0:
        return image.copy()
    
    severity = np.clip(severity, 0.0, 1.0)
    rng = np.random.default_rng(seed)
    sigma = severity * 0.35
    noise = rng.normal(loc=0.0, scale=sigma, size=image.shape).astype(np.float32)
    perturbed = image + noise
    return np.clip(perturbed, 0.0, 1.0).astype(np.float32)


def gaussian_blur(
    image: np.ndarray,
    severity: float
) -> np.ndarray:
    """
    Apply Gaussian spatial smoothing.
    At severity=1.0, kernel size corresponds to ~10% of image size.
    """
    if severity <= 0.0:
        return image.copy()
    
    severity = np.clip(severity, 0.0, 1.0)
    max_ksize = max(3, int(min(image.shape[0], image.shape[1]) * 0.12))
    ksize = int(severity * max_ksize)
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(3, ksize)
    
    sigma = severity * 5.0 + 0.1
    if image.ndim == 3 and image.shape[2] == 1:
        blurred = cv2.GaussianBlur(image[:, :, 0], (ksize, ksize), sigmaX=sigma)
        return np.expand_dims(blurred, axis=-1).astype(np.float32)
    
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma)
    return np.clip(blurred, 0.0, 1.0).astype(np.float32)


def contrast_scaling(
    image: np.ndarray,
    severity: float,
    mode: str = "decrease"
) -> np.ndarray:
    """
    Scale contrast around the image mean.
    If mode='decrease', severity 1.0 shrinks contrast to 10% of original.
    If mode='increase', severity 1.0 amplifies contrast up to 2.5x.
    """
    if severity <= 0.0:
        return image.copy()
    
    severity = np.clip(severity, 0.0, 1.0)
    mean_val = np.mean(image)
    if mode == "decrease":
        factor = 1.0 - (0.9 * severity)
    else:
        factor = 1.0 + (1.5 * severity)
        
    adjusted = (image - mean_val) * factor + mean_val
    return np.clip(adjusted, 0.0, 1.0).astype(np.float32)


def intensity_shift(
    image: np.ndarray,
    severity: float,
    direction: float = -1.0
) -> np.ndarray:
    """
    Shift global intensity (brightness/underexposure or overexposure).
    direction: -1.0 for darkening, +1.0 for brightening.
    At severity=1.0, shifts intensity by +/- 0.5.
    """
    if severity <= 0.0:
        return image.copy()
    
    severity = np.clip(severity, 0.0, 1.0)
    shift = np.sign(direction) * severity * 0.5
    shifted = image + shift
    return np.clip(shifted, 0.0, 1.0).astype(np.float32)


def local_occlusion(
    image: np.ndarray,
    severity: float,
    center_rel: Tuple[float, float] = (0.5, 0.5),
    fill_value: float = 0.0
) -> np.ndarray:
    """
    Apply a localized square occlusion (e.g. simulating foreign body, radio-opaque marker, or blind spot).
    Severity scales occlusion box side from 0% to 50% of the image size.
    center_rel: (y_rel, x_rel) relative center in [0.0, 1.0].
    """
    if severity <= 0.0:
        return image.copy()
    
    severity = np.clip(severity, 0.0, 1.0)
    out = image.copy()
    h, w = image.shape[:2]
    
    box_size = int(severity * min(h, w) * 0.5)
    if box_size < 1:
        return out

    cy = int(center_rel[0] * h)
    cx = int(center_rel[1] * w)

    half = box_size // 2
    y1 = max(0, cy - half)
    y2 = min(h, cy + half)
    x1 = max(0, cx - half)
    x2 = min(w, cx + half)

    out[y1:y2, x1:x2, ...] = fill_value
    return out.astype(np.float32)


def salt_and_pepper_noise(
    image: np.ndarray,
    severity: float,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Apply salt-and-pepper (impulse sensor) noise.
    At severity=1.0, 15% of pixels are corrupted.
    """
    if severity <= 0.0:
        return image.copy()

    severity = np.clip(severity, 0.0, 1.0)
    rng = np.random.default_rng(seed)
    out = image.copy()
    
    prob = severity * 0.15
    mask = rng.uniform(0.0, 1.0, size=image.shape[:2])
    
    pepper_mask = mask < (prob / 2.0)
    salt_mask = (mask >= (prob / 2.0)) & (mask < prob)
    
    out[pepper_mask, ...] = 0.0
    out[salt_mask, ...] = 1.0
    return out.astype(np.float32)


PERTURBATION_REGISTRY: Dict[str, Callable[..., np.ndarray]] = {
    "Gaussian Noise": gaussian_noise,
    "Gaussian Blur": gaussian_blur,
    "Contrast Reduction": lambda img, s: contrast_scaling(img, s, mode="decrease"),
    "Intensity Darkening": lambda img, s: intensity_shift(img, s, direction=-1.0),
    "Intensity Brightening": lambda img, s: intensity_shift(img, s, direction=1.0),
    "Local Occlusion": local_occlusion,
    "Salt and Pepper": salt_and_pepper_noise,
}
