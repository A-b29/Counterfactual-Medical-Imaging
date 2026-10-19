"""

app.py - Streamlit Interactive Workbench for Counterfactual Medical Imaging Lab.
"""

from pathlib import Path
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

from src.image_io import load_image
from src.sample_generator import create_default_sample_data
from src.perturbations import PERTURBATION_REGISTRY
from src.model_adapter import CalibratedChestXRayAdapter
from src.metrics import compute_prediction_delta, compute_robustness_curve_metrics
from src.reporting import export_results_csv, export_results_json, generate_markdown_report

st.set_page_config(
    page_title="Counterfactual Medical Imaging Lab",
    page_icon="🔬",
    layout="wide"
)

# Custom header
st.title("🔬 Counterfactual Medical Imaging Lab")
st.markdown(
    "Measure model prediction sensitivity and failure modes under controlled physical/radiological perturbations."
)
st.caption("⚠️ Non-Clinical Prototype: Evaluates algorithmic sensitivity, not clinical diagnostic validity.")

# Ensure sample data exists
sample_dir = Path(__file__).resolve().parent / "sample_data"
create_default_sample_data(sample_dir)

@st.cache_resource
def get_model():
    return CalibratedChestXRayAdapter()

model = get_model()

# Sidebar: Image and Perturbation Controls
st.sidebar.header("1. Input Configuration")
input_mode = st.sidebar.radio("Select Image Source:", ["Synthetic Demo Phantoms", "Upload Custom Image"])

current_image = None
image_name = "unknown"

if input_mode == "Synthetic Demo Phantoms":
    available_samples = list(sample_dir.glob("*.png"))
    sample_names = [p.name for p in available_samples]
    selected_sample = st.sidebar.selectbox("Choose Phantom:", sample_names)
    if selected_sample:
        image_path = sample_dir / selected_sample
        current_image = load_image(image_path)
        image_name = selected_sample
else:
    uploaded_file = st.sidebar.file_uploader("Upload Image (PNG/JPEG/TIFF):", type=["png", "jpg", "jpeg", "tif"])
    if uploaded_file is not None:
        temp_path = sample_dir / f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        current_image = load_image(temp_path)
        image_name = uploaded_file.name

if current_image is None:
    st.info("Please select or upload an image to begin.")
    st.stop()

st.sidebar.header("2. Perturbation Controls")
selected_perturbation = st.sidebar.selectbox("Select Transformation:", list(PERTURBATION_REGISTRY.keys()))
severity = st.sidebar.slider("Severity Level (0.0 = Identity, 1.0 = Max):", min_value=0.0, max_value=1.0, value=0.35, step=0.05)

# Execute baseline & perturbed
baseline_pred = model.predict(current_image)
pert_func = PERTURBATION_REGISTRY[selected_perturbation]
perturbed_image = pert_func(current_image, severity)
perturbed_pred = model.predict(perturbed_image)
delta = compute_prediction_delta(baseline_pred, perturbed_pred)

# Layout: Visual Comparison
col_left, col_mid, col_right = st.columns([1.2, 1.2, 1.4])

with col_left:
    st.subheader("Baseline Image")
    st.image(current_image, clamp=True, use_container_width=True)
    st.markdown(f"**Predicted:** `{baseline_pred.predicted_class}`")
    st.markdown(f"**Confidence:** `{baseline_pred.confidence:.2%}`")

with col_mid:
    st.subheader(f"Perturbed ({selected_perturbation} @ {severity:.2f})")
    st.image(perturbed_image, clamp=True, use_container_width=True)
    flip_badge = "🔴 **CLASS FLIPPED**" if delta["is_flipped"] else "🟢 **STABLE**"
    st.markdown(f"**Predicted:** `{perturbed_pred.predicted_class}` ({flip_badge})")
    st.markdown(f"**Confidence:** `{perturbed_pred.confidence:.2%}`")

with col_right:
    st.subheader("Class Probability Comparison")
    classes = list(baseline_pred.probabilities.keys())
    base_probs = [baseline_pred.probabilities[c] for c in classes]
    pert_probs = [perturbed_pred.probabilities.get(c, 0.0) for c in classes]

    fig = go.Figure(data=[
        go.Bar(name='Baseline', x=classes, y=base_probs, marker_color='#2563eb'),
        go.Bar(name='Perturbed', x=classes, y=pert_probs, marker_color='#dc2626' if delta["is_flipped"] else '#16a34a')
    ])
    fig.update_layout(
        barmode='group',
        yaxis_title="Probability",
        yaxis_range=[0, 1.05],
        margin=dict(l=10, r=10, t=20, b=20),
        height=280
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Differential Metrics")
    st.metric("Probability Drift (Baseline Class)", f"{delta['baseline_prob_drift']:+.2%}")
    st.metric("Jensen-Shannon Divergence", f"{delta['jensen_shannon_div']:.4f}")

# Full Severity Sweep
st.divider()
st.subheader("📈 Full Perturbation Sensitivity Sweep")

sweep_steps = np.linspace(0.0, 1.0, 11).tolist()
sweep_records = []
retained_probs = []
flips = []

for s in sweep_steps:
    p_img = pert_func(current_image, s)
    pred_s = model.predict(p_img)
    ret_prob = pred_s.probabilities.get(baseline_pred.predicted_class, 0.0)
    retained_probs.append(ret_prob)
    flipped = pred_s.predicted_class != baseline_pred.predicted_class
    flips.append(flipped)
    sweep_records.append({
        "Severity": round(s, 2),
        "Predicted Class": pred_s.predicted_class,
        "Confidence": f"{pred_s.confidence:.2%}",
        "Baseline Class Prob": f"{ret_prob:.2%}",
        "Flipped": "Yes" if flipped else "No"
    })

curve_metrics = compute_robustness_curve_metrics(sweep_steps, retained_probs, flips)

col_chart, col_stats = st.columns([2, 1])

with col_chart:
    fig_sweep = go.Figure()
    fig_sweep.add_trace(go.Scatter(
        x=sweep_steps,
        y=retained_probs,
        mode='lines+markers',
        name=f'{baseline_pred.predicted_class} Retention',
        line=dict(color='#2563eb', width=3)
    ))
    fig_sweep.update_layout(
        xaxis_title="Perturbation Severity",
        yaxis_title=f"P({baseline_pred.predicted_class})",
        yaxis_range=[0, 1.05],
        height=320,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_sweep, use_container_width=True)

with col_stats:
    st.markdown("#### Curve Summary")
    st.metric("Retention AUC", f"{curve_metrics['retention_auc']:.3f}")
    crit = curve_metrics['critical_flip_severity']
    st.metric("Critical Flip Severity", f"{crit:.2f}" if crit is not None else "Resistant (No Flip)")

    df_sweep = pd.DataFrame(sweep_records)
    csv_bytes = df_sweep.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download CSV Results",
        data=csv_bytes,
        file_name=f"robustness_{selected_perturbation.lower().replace(' ', '_')}.csv",
        mime="text/csv"
    )
