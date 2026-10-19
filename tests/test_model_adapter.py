"""
test_model_adapter.py - Unit tests for model inference adapter.
"""

import numpy as np
import pytest
from src.model_adapter import CalibratedChestXRayAdapter


@pytest.fixture
def adapter():
    return CalibratedChestXRayAdapter()


def test_model_adapter_output_structure(adapter):
    """Adapter output must satisfy the schema and sum probabilities to 1."""
    img = np.zeros((128, 128, 1), dtype=np.float32)
    res = adapter.predict(img)

    assert res.predicted_class in adapter.class_names
    assert res.class_index in [0, 1, 2]
    assert 0.0 <= res.confidence <= 1.0

    prob_sum = sum(res.probabilities.values())
    assert prob_sum == pytest.approx(1.0, abs=1e-5)


def test_model_rgb_compatibility(adapter):
    """Adapter should handle 3-channel RGB image gracefully."""
    img = np.ones((128, 128, 3), dtype=np.float32) * 0.5
    res = adapter.predict(img)
    assert res.predicted_class in adapter.class_names
