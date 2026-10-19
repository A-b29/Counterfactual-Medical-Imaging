"""
cli.py - Command-line interface for batch counterfactual medical robustness audits.
"""

import argparse
from pathlib import Path
import numpy as np

from src.image_io import load_image, save_image
from src.sample_generator import create_default_sample_data
from src.perturbations import PERTURBATION_REGISTRY
from src.model_adapter import CalibratedChestXRayAdapter
from src.metrics import compute_prediction_delta, compute_robustness_curve_metrics
from src.reporting import export_results_csv, export_results_json, generate_markdown_report


def run_audit(image_path: Path, output_dir: Path, steps: int = 6):
    print(f"=== Starting Counterfactual Robustness Audit on {image_path.name} ===")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load image and model
    image = load_image(image_path)
    model = CalibratedChestXRayAdapter()
    
    # 2. Baseline inference
    baseline_pred = model.predict(image)
    print(f"Baseline: {baseline_pred.predicted_class} (Confidence: {baseline_pred.confidence:.2%})")

    severities = np.linspace(0.0, 1.0, steps).tolist()
    records = []
    curve_metrics = {}

    # 3. Sweep perturbations
    for pert_name, pert_func in PERTURBATION_REGISTRY.items():
        print(f"Running perturbation sweep: {pert_name}...")
        retained_probs = []
        flips = []

        for sev in severities:
            perturbed_img = pert_func(image, sev)
            pred = model.predict(perturbed_img)
            delta = compute_prediction_delta(baseline_pred, pred)

            rec = {
                "image_name": image_path.name,
                "perturbation": pert_name,
                "severity": float(sev),
                "predicted_class": pred.predicted_class,
                "confidence": pred.confidence,
                "is_flipped": delta["is_flipped"],
                "baseline_prob_drift": delta["baseline_prob_drift"],
                "jensen_shannon_div": delta["jensen_shannon_div"],
                "baseline_class_retained_prob": pred.probabilities.get(baseline_pred.predicted_class, 0.0)
            }
            records.append(rec)
            retained_probs.append(rec["baseline_class_retained_prob"])
            flips.append(rec["is_flipped"])

        curve_metrics[pert_name] = compute_robustness_curve_metrics(severities, retained_probs, flips)

    # 4. Export artifacts
    csv_file = export_results_csv(records, output_dir / "perturbation_metrics.csv")
    json_summary = {
        "model_name": "CalibratedChestXRayAdapter_v1",
        "image_name": image_path.name,
        "baseline_class": baseline_pred.predicted_class,
        "baseline_conf": baseline_pred.confidence,
        "baseline_probabilities": baseline_pred.probabilities,
        "total_trials": len(records),
        "curve_metrics": curve_metrics,
    }
    json_file = export_results_json(json_summary, output_dir / "audit_summary.json")
    md_file = generate_markdown_report(json_summary, curve_metrics, output_dir / "audit_report.md")

    print(f"\nAudit complete!")
    print(f"-> CSV results: {csv_file}")
    print(f"-> JSON summary: {json_file}")
    print(f"-> Markdown report: {md_file}")


def main():
    parser = argparse.ArgumentParser(description="Counterfactual Medical Imaging Lab CLI")
    parser.add_argument("--image", type=str, default=None, help="Path to input medical image.")
    parser.add_argument("--output-dir", type=str, default="reports", help="Directory for output reports.")
    parser.add_argument("--steps", type=int, default=6, help="Number of severity steps between 0 and 1.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    sample_dir = project_root / "sample_data"
    create_default_sample_data(sample_dir)

    if args.image is None:
        image_path = sample_dir / "chest_xray_nodule.png"
    else:
        image_path = Path(args.image)

    run_audit(image_path, Path(args.output_dir), steps=args.steps)


if __name__ == "__main__":
    main()
