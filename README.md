# Counterfactual Medical Imaging Lab

An interactive workbench and CLI tool to measure prediction stability and failure modes under controlled physical and radiological perturbations.

> **Research Prototype Disclaimer:** This system is for algorithmic robustness testing and explainability research. It is **not** a validated medical device and must **not** be used for clinical decision-making.

---

## Features
- **6 Controlled Perturbations:** Gaussian noise, Gaussian blur, contrast scaling, intensity shifts, local occlusion, and salt-and-pepper sensor noise.
- **Strict Invariance:** `severity = 0.0` guarantees mathematical identity.
- **Calibrated Inference Adapter:** Self-contained radiological convolutional model with biological spatial priors.
- **Information-Theoretic Metrics:** Jensen-Shannon divergence, baseline class retention AUC, critical flip threshold, and probability drift.
- **Dual Interface:** Interactive Streamlit web lab and headless batch CLI runner.

---

## Quickstart

### 1. Run Unit Tests
```bash
pytest tests/ -v
```

### 2. Run Headless CLI Audit
```bash
python cli.py --image sample_data/chest_xray_nodule.png --output-dir reports/
```

### 3. Launch Interactive Streamlit Lab
```bash
streamlit run app.py
```

---

## File Structure
```text
counterfactual_medical_imaging/
├── app.py                      # Streamlit interactive visualizer
├── cli.py                      # Headless batch evaluation CLI
├── DECISIONS.md                # Cunning and architectural decisions
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── src/
│   ├── image_io.py             # Normalization and image loading
│   ├── model_adapter.py        # Abstract adapter and calibrated model
│   ├── perturbations.py        # Pure perturbation functions with severity [0, 1]
│   ├── metrics.py              # Divergence, flip detection, and retention AUC
│   ├── reporting.py            # Markdown/JSON/CSV export
│   └── sample_generator.py     # Synthetic medical phantom generator
├── tests/
│   ├── test_perturbations.py
│   ├── test_metrics.py
│   └── test_model_adapter.py
└── sample_data/
    └── README.md
```
