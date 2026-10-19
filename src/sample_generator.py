"""
sample_generator.py - Generate realistic, synthetic medical imaging phantoms.

Creates clean, self-contained test images with known ground-truth patterns:
1. Synthetic Chest Radiograph (bilateral lung fields, cardiac silhouette, clavicles, simulated nodule)
2. Synthetic Dermoscopy Phantom (pigmented asymmetric lesion, pigment network, erythema)
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import cv2
from .image_io import save_image


def generate_chest_xray_phantom(
    size: Tuple[int, int] = (256, 256),
    with_nodule: bool = True,
    seed: int = 42
) -> np.ndarray:
    """
    Generate an anatomically structured synthetic chest radiograph phantom.
    Pixel values are in float32 [0.0, 1.0].
    """
    rng = np.random.default_rng(seed)
    h, w = size
    img = np.zeros((h, w), dtype=np.float32)

    # Base soft-tissue background
    img += 0.25

    # Thoracic cage ellipse
    cv2.ellipse(img, (w // 2, h // 2), (int(w * 0.42), int(h * 0.45)), 0, 0, 360, 0.45, -1)

    # Left & Right lung fields (air-filled, low radiopacity -> darker)
    cv2.ellipse(img, (int(w * 0.32), int(h * 0.48)), (int(w * 0.14), int(h * 0.32)), -5, 0, 360, 0.08, -1)
    cv2.ellipse(img, (int(w * 0.68), int(h * 0.48)), (int(w * 0.14), int(h * 0.32)), 5, 0, 360, 0.08, -1)

    # Mediastinum and spine (dense central strip)
    cv2.rectangle(img, (int(w * 0.44), 0), (int(w * 0.56), h), 0.55, -1)

    # Cardiac silhouette (dense, shifted left)
    cv2.ellipse(img, (int(w * 0.52), int(h * 0.60)), (int(w * 0.13), int(h * 0.16)), -20, 0, 360, 0.65, -1)

    # Diaphragmatic domes
    cv2.ellipse(img, (int(w * 0.30), int(h * 0.82)), (int(w * 0.22), int(h * 0.12)), 0, 0, 360, 0.50, -1)
    cv2.ellipse(img, (int(w * 0.70), int(h * 0.84)), (int(w * 0.22), int(h * 0.12)), 0, 0, 360, 0.50, -1)

    # Clavicles / rib shadows
    for y_rib in np.linspace(h * 0.25, h * 0.75, 5):
        cv2.line(img, (int(w * 0.15), int(y_rib)), (int(w * 0.45), int(y_rib + 15)), 0.38, 3)
        cv2.line(img, (int(w * 0.85), int(y_rib)), (int(w * 0.55), int(y_rib + 15)), 0.38, 3)

    # Optional pathology: focal lung nodule/consolidation in right lower zone
    if with_nodule:
        cv2.circle(img, (int(w * 0.30), int(h * 0.55)), int(w * 0.05), 0.72, -1)

    # Anatomical tissue blur to simulate X-ray scatter
    img = cv2.GaussianBlur(img, (11, 11), 3.0)

    # Quantum mottle / detector noise
    mottle = rng.normal(0.0, 0.015, size=(h, w)).astype(np.float32)
    img = np.clip(img + mottle, 0.0, 1.0)

    return np.expand_dims(img, axis=-1)


def generate_dermoscopy_phantom(
    size: Tuple[int, int] = (256, 256),
    seed: int = 101
) -> np.ndarray:
    """
    Generate a synthetic dermoscopy phantom (RGB skin lesion).
    """
    rng = np.random.default_rng(seed)
    h, w = size
    # Base skin tone (fair skin RGB)
    skin = np.zeros((h, w, 3), dtype=np.float32)
    skin[:, :, 0] = 0.92  # Red
    skin[:, :, 1] = 0.76  # Green
    skin[:, :, 2] = 0.68  # Blue

    # Vignetting (dermatoscope circular lens)
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2
    radius = min(h, w) * 0.48
    mask = dist_sq > radius ** 2
    skin[mask] = 0.05

    # Pigmented lesion center
    lesion_mask = dist_sq < (radius * 0.5) ** 2
    skin[lesion_mask, 0] *= 0.45
    skin[lesion_mask, 1] *= 0.25
    skin[lesion_mask, 2] *= 0.15

    # Micro-texture noise
    noise = rng.normal(0.0, 0.01, size=(h, w, 3)).astype(np.float32)
    img = np.clip(skin + noise, 0.0, 1.0)
    return img


def create_default_sample_data(sample_dir: Path) -> None:
    """Generate and save default sample images into sample_data/."""
    sample_dir.mkdir(parents=True, exist_ok=True)
    cxr_path = sample_dir / "chest_xray_nodule.png"
    normal_cxr_path = sample_dir / "chest_xray_normal.png"
    dermo_path = sample_dir / "dermoscopy_lesion.png"

    if not cxr_path.exists():
        cxr = generate_chest_xray_phantom(with_nodule=True, seed=42)
        save_image(cxr, cxr_path)

    if not normal_cxr_path.exists():
        normal_cxr = generate_chest_xray_phantom(with_nodule=False, seed=99)
        save_image(normal_cxr, normal_cxr_path)

    if not dermo_path.exists():
        dermo = generate_dermoscopy_phantom(seed=101)
        save_image(dermo, dermo_path)
