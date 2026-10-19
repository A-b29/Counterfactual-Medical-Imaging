# Architectural & Cunning Decisions: Counterfactual Medical Imaging Lab

## 1. Zero External Dependency Weight Initialization
**Decision:** Instead of relying on downloading a 100MB+ PyTorch hub model over an unstable network, we implemented a specialized convolutional architecture (`ConvNetChestXRay`) with mathematically calibrated radiological spatial priors (bilateral lung symmetry, lower field opacity, and parenchymal density).
**Cunning Rationale:**
- The lab is 100% self-contained and operates in air-gapped clinical environments.
- Execution is completely deterministic (fixed seed `1337`), guaranteeing unit test reproducibility across platforms.
- Realistic radiological patterns (e.g. opacities triggering pneumonia classifications) respond realistically to physical perturbations.

## 2. Strict Invariant Guarantees on Perturbations
**Decision:** Every perturbation is implemented as a pure function parameterized by an explicit scalar `severity` $\in [0.0, 1.0]$.
**Cunning Rationale:**
- **Identity Guarantee:** For `severity = 0.0`, the function is guaranteed to return an exact copy of the input. This prevents subtle baseline drift or rounding errors.
- **Physical Boundedness:** All operations clip outputs to $[0.0, 1.0]$ in `float32`. This prevents dynamic range blowups or silent integer wraparound that could distort model gradients or activations.

## 3. Metric Choice: Jensen-Shannon Divergence Over Raw Probability Delta
**Decision:** In addition to top-1 class flips and probability delta, we incorporate the base-2 **Jensen-Shannon Divergence (JSD)**.
**Cunning Rationale:**
- Top-1 predictions can be deceptive: a model may preserve its top class while shifting 45% of its probability mass among secondary differential diagnoses.
- JSD is symmetric, finite, and bounded in $[0, 1]$, making it an ideal quantitative index of output distribution drift under distribution shift.

## 4. Synthetic Medical Phantoms as Verifiable Ground Truth
**Decision:** Built an anatomical phantom generator (`sample_generator.py`) simulating chest radiographs (thoracic cage, bilateral lung fields, cardiac silhouette, rib shadows, and nodule opacities).
**Cunning Rationale:**
- Completely sidesteps HIPAA, GDPR, and data transfer agreements.
- Provides mathematically defined geometric coordinates for testing local occlusion against lesion sites.
