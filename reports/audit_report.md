# Counterfactual Medical Imaging Robustness Report

> **Non-Clinical Disclaimer:** This report is produced by a research robustness-analysis prototype. Perturbation sensitivity measurements do NOT constitute clinical safety validation or diagnostic certification.

## 1. Experiment Overview
- **Model Identifier:** `CalibratedChestXRayAdapter_v1`
- **Input Image:** `chest_xray_nodule.png`
- **Baseline Prediction:** **Normal** *(Confidence: 61.74%)*
- **Total Perturbation Trials:** `42`

## 2. Perturbation Robustness Summary

| Perturbation | Retention AUC | Critical Flip Severity | Final Baseline Probability |
| :--- | :---: | :---: | :---: |
| Gaussian Noise | `0.617` | `None (Resistant)` | `61.71%` |
| Gaussian Blur | `0.617` | `None (Resistant)` | `61.73%` |
| Contrast Reduction | `0.617` | `None (Resistant)` | `61.67%` |
| Intensity Darkening | `0.618` | `None (Resistant)` | `61.77%` |
| Intensity Brightening | `0.616` | `None (Resistant)` | `61.29%` |
| Local Occlusion | `0.617` | `None (Resistant)` | `61.70%` |
| Salt and Pepper | `0.617` | `None (Resistant)` | `61.72%` |

## 3. Methodological Notes
- **Identity Verification:** All transformations were verified at `severity = 0.0` ensuring zero baseline distortion.
- **Probability Retention AUC:** Measures the integral of baseline class probability across the severity interval $[0.0, 1.0]$. Values near 1.0 indicate high stability; values approaching 0.0 reflect severe degradation.
- **Reproducibility:** Seeded random transformations guarantee deterministic execution.

---
*Generated automatically by Counterfactual Medical Imaging Lab.*