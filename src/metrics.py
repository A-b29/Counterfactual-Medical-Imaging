"""
metrics.py - Quantitative robustness and perturbation sensitivity metrics.

Calculates:
- Prediction Flip Indicator (binary)
- Baseline Class Probability Drift (delta)
- Jensen-Shannon Divergence between baseline and perturbed probability distributions
- Maximum Confidence Shift
- Area Under the Probability Retention Curve (AUPRC)
"""

from typing import Dict, List, Any
import numpy as np
from scipy.spatial.distance import jensenshannon
from .model_adapter import PredictionResult


def compute_prediction_delta(
    baseline: PredictionResult,
    perturbed: PredictionResult
) -> Dict[str, Any]:
    """
    Compare a baseline prediction against a perturbed prediction.

    Returns:
        dict containing:
            - is_flipped: bool
            - baseline_class: str
            - perturbed_class: str
            - baseline_conf: float
            - perturbed_conf: float
            - baseline_prob_drift: float (P_base(y_base) - P_pert(y_base))
            - max_prob_drift: float (max_c |P_base(c) - P_pert(c)|)
            - jensen_shannon_div: float (symmetric divergence in [0, 1])
    """
    is_flipped = bool(baseline.predicted_class != perturbed.predicted_class)
    
    # Probability vectors aligned by class key
    classes = list(baseline.probabilities.keys())
    p_base = np.array([baseline.probabilities[c] for c in classes], dtype=np.float64)
    p_pert = np.array([perturbed.probabilities.get(c, 0.0) for c in classes], dtype=np.float64)

    # Normalize to avoid numerical drift
    p_base = p_base / (np.sum(p_base) + 1e-12)
    p_pert = p_pert / (np.sum(p_pert) + 1e-12)

    # Baseline class probability drop
    target_cls = baseline.predicted_class
    base_prob_orig = baseline.probabilities[target_cls]
    base_prob_new = perturbed.probabilities.get(target_cls, 0.0)
    baseline_prob_drift = float(base_prob_orig - base_prob_new)

    # Max absolute deviation across all classes
    max_prob_drift = float(np.max(np.abs(p_base - p_pert)))

    # Jensen-Shannon Divergence (base 2 -> bounded in [0, 1])
    js_div = float(jensenshannon(p_base, p_pert, base=2.0))

    return {
        "is_flipped": is_flipped,
        "baseline_class": baseline.predicted_class,
        "perturbed_class": perturbed.predicted_class,
        "baseline_conf": float(baseline.confidence),
        "perturbed_conf": float(perturbed.confidence),
        "baseline_prob_drift": baseline_prob_drift,
        "max_prob_drift": max_prob_drift,
        "jensen_shannon_div": js_div,
    }


def compute_robustness_curve_metrics(
    severity_levels: List[float],
    baseline_probs_retained: List[float],
    flips: List[bool]
) -> Dict[str, Any]:
    """
    Summarize a full sweep across severities for a single transformation.

    Returns:
        dict:
            - critical_flip_severity: float or None (lowest severity causing a flip)
            - retention_auc: float (area under probability retention curve)
            - max_drift: float
    """
    severities = np.array(severity_levels, dtype=np.float64)
    retained = np.array(baseline_probs_retained, dtype=np.float64)

    # Sort by severity
    sort_idx = np.argsort(severities)
    severities = severities[sort_idx]
    retained = retained[sort_idx]
    flips_sorted = [flips[i] for i in sort_idx]

    # Find earliest flip
    critical_flip_severity = None
    for s, flipped in zip(severities, flips_sorted):
        if flipped:
            critical_flip_severity = float(s)
            break

    # Trapezoidal AUC normalized by severity range
    if len(severities) > 1 and (severities[-1] - severities[0]) > 0:
        norm_range = severities[-1] - severities[0]
        auc = float(np.trapezoid(retained, severities) / norm_range)
    else:
        auc = float(retained[0]) if len(retained) > 0 else 0.0

    return {
        "critical_flip_severity": critical_flip_severity,
        "retention_auc": auc,
        "final_retention": float(retained[-1]) if len(retained) > 0 else 0.0,
    }
