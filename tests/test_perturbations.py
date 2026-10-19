"""
test_perturbations.py - Unit tests for perturbation pure functions.
"""

import numpy as np
import pytest
from src.perturbations import (
    gaussian_noise,
    gaussian_blur,
    contrast_scaling,
    intensity_shift,
    local_occlusion,
    salt_and_pepper_noise,
    PERTURBATION_REGISTRY
)


@pytest.fixture
def sample_image():
    """Deterministic synthetic test image."""
    rng = np.random.default_rng(123)
    return rng.uniform(0.1, 0.9, size=(64, 64, 1)).astype(np.float32)


def test_identity_at_zero_severity(sample_image):
    """Every perturbation must return an exact duplicate at severity=0.0."""
    for name, func in PERTURBATION_REGISTRY.items():
        transformed = func(sample_image, 0.0)
        assert np.allclose(transformed, sample_image, atol=1e-6), (
            f"Perturbation '{name}' failed identity guarantee at severity=0.0"
        )


def test_shape_and_dtype_preservation(sample_image):
    """Perturbations must preserve original shape and float32 dtype."""
    for name, func in PERTURBATION_REGISTRY.items():
        transformed = func(sample_image, 0.5)
        assert transformed.shape == sample_image.shape, f"Shape mismatch in {name}"
        assert transformed.dtype == np.float32, f"Dtype mismatch in {name}"


def test_value_bounding(sample_image):
    """All output pixel values must remain strictly within [0.0, 1.0]."""
    for name, func in PERTURBATION_REGISTRY.items():
        transformed = func(sample_image, 1.0)
        assert np.all(transformed >= 0.0), f"Negative values found in {name}"
        assert np.all(transformed <= 1.0), f"Values > 1.0 found in {name}"


def test_local_occlusion_box(sample_image):
    """Local occlusion should set a nonzero region to fill_value."""
    occluded = local_occlusion(sample_image, severity=0.5, center_rel=(0.5, 0.5), fill_value=0.0)
    assert np.min(occluded) == 0.0
    # Center pixel must be 0.0
    assert occluded[32, 32, 0] == 0.0
