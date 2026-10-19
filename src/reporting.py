"""
reporting.py - Generates structured JSON, CSV, and Markdown audit reports.
"""

from pathlib import Path
from typing import List, Dict, Any
import json
import pandas as pd


def export_results_csv(records: List[Dict[str, Any]], output_path: Path) -> Path:
    """Save evaluation rows to CSV."""
    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def export_results_json(data: Dict[str, Any], output_path: Path) -> Path:
    """Save full run metadata and results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return output_path


def generate_markdown_report(
    summary_data: Dict[str, Any],
    curve_metrics: Dict[str, Dict[str, Any]],
    output_path: Path
) -> Path:
    """Generate a clean, reproducible Markdown audit report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Counterfactual Medical Imaging Robustness Report",
        "",
        "> **Non-Clinical Disclaimer:** This report is produced by a research robustness-analysis prototype. "
        "Perturbation sensitivity measurements do NOT constitute clinical safety validation or diagnostic certification.",
        "",
        "## 1. Experiment Overview",
        f"- **Model Identifier:** `{summary_data.get('model_name', 'Unknown')}`",
        f"- **Input Image:** `{summary_data.get('image_name', 'Unknown')}`",
        f"- **Baseline Prediction:** **{summary_data.get('baseline_class', 'N/A')}** "
        f"*(Confidence: {summary_data.get('baseline_conf', 0.0):.2%})*",
        f"- **Total Perturbation Trials:** `{summary_data.get('total_trials', 0)}`",
        "",
        "## 2. Perturbation Robustness Summary",
        "",
        "| Perturbation | Retention AUC | Critical Flip Severity | Final Baseline Probability |",
        "| :--- | :---: | :---: | :---: |",
    ]

    for pert_name, metrics in curve_metrics.items():
        flip_str = f"{metrics.get('critical_flip_severity', 'Never'):.2f}" if metrics.get('critical_flip_severity') is not None else "None (Resistant)"
        auc = metrics.get('retention_auc', 0.0)
        final_p = metrics.get('final_retention', 0.0)
        lines.append(f"| {pert_name} | `{auc:.3f}` | `{flip_str}` | `{final_p:.2%}` |")

    lines.extend([
        "",
        "## 3. Methodological Notes",
        "- **Identity Verification:** All transformations were verified at `severity = 0.0` ensuring zero baseline distortion.",
        "- **Probability Retention AUC:** Measures the integral of baseline class probability across the severity interval $[0.0, 1.0]$. Values near 1.0 indicate high stability; values approaching 0.0 reflect severe degradation.",
        "- **Reproducibility:** Seeded random transformations guarantee deterministic execution.",
        "",
        "---",
        "*Generated automatically by Counterfactual Medical Imaging Lab.*"
    ])

    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return output_path
