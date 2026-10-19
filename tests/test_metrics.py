"""
test_metrics.py - Unit tests for metrics engine.
"""

import numpy as np
import pytest
from src.model_adapter import PredictionResult
from src.metrics import compute_prediction_delta, compute_robustness_curve_metrics


def test_prediction_delta_identical():
    """Identical predictions should have zero drift and zero JS divergence."""
    res = PredictionResult(
        predicted_class="Normal",
        class_index=0,
        probabilities={"Normal": 0.9, "Pneumonia": 0.1},
        confidence=0.9
    )
    delta = compute_prediction_delta(res, res)
    assert delta["is_flipped"] is False
    assert delta["baseline_prob_drift"] == 0.0
    assert delta["max_prob_drift"] == 0.0
    assert delta["jensen_shannon_div"] == pytest.approx(0.0, abs=1e-5)


def test_prediction_delta_flip():
    """Flipped prediction should trigger is_flipped=True and non-zero JS divergence."""
    p1 = PredictionResult(
        predicted_class="Normal",
        class_index=0,
        probabilities={"Normal": 0.8, "Pneumonia": 0.2},
        confidence=0.8
    )
    p2 = PredictionResult(
        predicted_class="Pneumonia",
        class_index=1,
        probabilities={"Normal": 0.1, "Pneumonia": 0.9},
        confidence=0.9
    )
    delta = compute_prediction_delta(p1, p2)
    assert delta["is_flipped"] is True
    assert delta["baseline_prob_drift"] == pytest.approx(0.7, abs=1e-4)
    assert delta["jensen_shannon_div"] > 0.3


def test_curve_metrics_auc():
    """Test trapezoidal AUC and earliest flip detection."""
    severities = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    retained_probs = [1.0, 0.9, 0.8, 0.4, 0.2, 0.1]
    flips = [False, False, False, True, True, True]

    metrics = compute_robustness_curve_metrics(severities, retained_probs, flips)
    assert metrics["critical_flip_severity"] == 0.6
    assert 0.4 < metrics["retention_auc"] < 0.7
    assert metrics["final_retention"] == 0.1
