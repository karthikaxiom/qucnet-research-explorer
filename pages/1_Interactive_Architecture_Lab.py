"""Interactive, source-backed QuCNet architecture explanations."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.quantum.simulator import fixed_weights, simulate_patch
from src.evidence.input_representations import ARCHITECTURE_LABELS, render_route

st.set_page_config(page_title="Architecture Lab | QuCNet", page_icon="🧬", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1240px; padding-top: 1.9rem;}
h1 {letter-spacing: -.038em;}
[data-testid="stMetric"] {background: rgba(100, 130, 170, .08); border: 1px solid rgba(120,140,170,.18); border-radius: 14px; padding: 16px;}
</style>
""", unsafe_allow_html=True)

provenance_path = ROOT / "provenance.json"
try:
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    source_commit = provenance["source_commit"]
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Verified provenance unavailable: {exc}")
    st.stop()

repo_root = "https://github.com/karthikaxiom/QCNN_Thyroid_Paper_Reproduction/blob/"

def source_url(path: str) -> str:
    return f"{repo_root}{source_commit}/{path}"

# Each pathway reflects a distinct historical or extension implementation;
# these are not a single interchangeable trained network.
PATHWAYS = {
    "Phase 6 · Binary quantum QuCNet": {
        "context": "Historical 70-image validation-selected holdout; TI-RADS-derived low/high suspicion proxy, not pathology diagnosis.",
        "source": "05_Source_Code/phase6_qucnet_model.py",
        "result": "54.29% historical holdout accuracy (38/70); classical binary baseline 57.14%.",
        "stages": [
            ("Grayscale ultrasound", "128 × 128 × 1", "Historical preprocessed full-image input"),
            ("Non-overlapping patches", "64 × 64 patches", "Each patch contains four 2 × 2 normalized pixel intensities"),
            ("Fixed simulated four-qubit map", "64 × 64 × 4", "RX encoding, two fixed RY/CNOT-ring layers, four Pauli-Z expectations"),
            ("Flatten", "16,384", "Concatenate the four-channel feature tensor"),
            ("Dense + ReLU", "128", "First learned classical classifier layer"),
            ("Dense + ReLU", "64", "Second learned classical classifier layer"),
            ("Softmax", "2", "Two-class TI-RADS-derived suspicion proxy output"),
        ],
    },
    "Phase 9 · Six-class quantum QuCNet": {
        "context": "Historical matched 70-image six-class TI-RADS comparison; earlier image-level splits do not establish independent cases.",
        "source": "05_Source_Code/phase9_paper_tirads_model.py",
        "result": "18.57% accuracy (13/70); implementation-chosen kernel sizes, widths, learning rate and training controls.",
        "stages": [
            ("Quantum feature tensor", "64 × 64 × 4", "Reconstructed four-qubit filter applied to 128 × 128 full images"),
            ("Conv2D · 16 filters", "64 × 64 × 16", "3 × 3 kernel, same padding, ReLU"),
            ("MaxPool2D", "32 × 32 × 16", "2 × 2 pooling"),
            ("Conv2D · 32 filters", "32 × 32 × 32", "3 × 3 kernel, same padding, ReLU"),
            ("MaxPool2D", "16 × 16 × 32", "2 × 2 pooling"),
            ("Flatten", "8,192", "Convert spatial maps into a vector"),
            ("Dense + ReLU", "128", "Learned classical dense representation"),
            ("Dense + ReLU", "64", "Learned classical dense representation"),
            ("Softmax", "6", "TI-RADS 2, 3, 4a, 4b, 4c, 5"),
        ],
    },
    "Phase 10 · Six-class classical CNN": {
        "context": "Historical matched grayscale six-class comparison using the same 70-image evaluation context as Phase 9.",
        "source": "05_Source_Code/phase10_classical_tirads_model.py",
        "result": "24.29% accuracy (17/70); majority-class reference 27.14%.",
        "stages": [
            ("Grayscale ultrasound", "128 × 128 × 1", "Original normalized full-image representation"),
            ("Resizing", "64 × 64 × 1", "Keras Resizing to match the quantum arm's spatial resolution"),
            ("Conv2D · 16 filters", "64 × 64 × 16", "3 × 3 kernel, same padding, ReLU"),
            ("MaxPool2D", "32 × 32 × 16", "2 × 2 pooling"),
            ("Conv2D · 32 filters", "32 × 32 × 32", "3 × 3 kernel, same padding, ReLU"),
            ("MaxPool2D", "16 × 16 × 32", "2 × 2 pooling"),
            ("Flatten", "8,192", "Vectorize convolutional features"),
            ("Dense + ReLU", "128", "Learned representation"),
            ("Dense + ReLU", "64", "Learned representation"),
            ("Softmax", "6", "Six-class TI-RADS prediction"),
        ],
    },
    "Phase 14 · Strict ROI CNN": {
        "context": "Separate historical recorded-case-ID-disjoint ROI evaluation: 71 ROIs / 44 cases; patient identity remains unverified.",
        "source": "05_Source_Code/phase14_train_roi_categorical_baseline.py",
        "result": "ROI 5.63% (4/71), case 4.55% (2/44); all historical test predictions were TI-RADS 3.",
        "stages": [
            ("Cropped grayscale lesion ROI", "64 × 64 × 1", "Square-padded lesion regions from expert annotation geometry"),
            ("Training-only augmentation", "64 × 64 × 1", "Flip, rotation, zoom and translation; inactive for validation/test"),
            ("Conv16 × 2 + BN/ReLU + pool", "32 × 32 × 16", "First convolutional block followed by Dropout 0.15"),
            ("Conv32 × 2 + BN/ReLU + pool", "16 × 16 × 32", "Second block followed by Dropout 0.20"),
            ("Conv64 × 2 + BN/ReLU + pool", "8 × 8 × 64", "Third block followed by Dropout 0.25"),
            ("Conv96 + BN/ReLU", "8 × 8 × 96", "Last convolutional feature map"),
            ("Global average pooling", "96", "Averages each feature channel spatially"),
            ("Dense + ReLU + Dropout", "64", "Classical final representation; Dropout 0.35"),
            ("Softmax", "6", "Six TI-RADS categories"),
        ],
    },
    "Extension · Matched fixed 64-D features": {
        "context": "Separate training-only fivefold case-ID-grouped out-of-fold study: 311 ROIs / 203 case IDs.",
        "source": "05_Source_Code/exp_qmap_001_fixed_feature_compare.py",
        "result": "Compared classical/quantum fixed 64-D features with logistic regression, RBF SVM, random forest and 7-NN. No proven quantum superiority.",
        "stages": [
            ("Cropped grayscale lesion ROI", "64 × 64 × 1", "Phase 13 training ROIs only"),
            ("Mean pooling", "8 × 8", "Common pooled input for both feature representations"),
            ("Classical branch", "64 features", "Flattened 8 × 8 mean-pooled grayscale intensities"),
            ("Fixed quantum branch", "64 features", "16 non-overlapping 2 × 2 patches × four Pauli-Z outputs"),
            ("Matched classifier families", "4 algorithms", "LR, RBF-SVM, random forest and 7-NN; classifiers are separately fitted"),
            ("Case-ID-grouped OOF evaluation", "203 cases", "Fivefold development-only analysis, not independent clinical testing"),
        ],
    },
}

st.sidebar.title("QuCNet Research Explorer")
st.sidebar.caption("Source-backed interactive architecture workbench")
st.sidebar.info("Current module: Architecture Lab")
st.sidebar.caption(f"Research commit: `{source_commit[:12]}`")

st.title("Interactive Architecture Lab")
st.caption("Five distinct implementation pathways · real source references · transparent tensor shapes")
st.info("Architecture inspection and synthetic circuit demonstration only. No medical-image classification or model training.")

selected = st.selectbox("Choose a verified pathway", list(PATHWAYS), key="arch_select")
path = PATHWAYS[selected]
st.markdown(f"**Evaluation context:** {path['context']}")
st.markdown(f"**Recorded finding:** {path['result']}")
st.link_button("Inspect original model source on GitHub", source_url(path["source"]), icon="🔗")
with st.expander("Source ultrasound input and tensor stages", expanded=False):
    render_route(st, ARCHITECTURE_LABELS[selected], source_commit)

st.subheader("Step through the implementation")
stages = path["stages"]
# Per-pathway widget keys avoid stale slider values when pathway lengths differ.
pathway_index = list(PATHWAYS).index(selected)
step = st.slider("Layer / operation", min_value=1, max_value=len(stages),
                 value=1, key=f"arch_slider_{pathway_index}")
name, shape, explanation = stages[step - 1]
left, right = st.columns([1.3, 1], gap="large")
with left:
    st.markdown(f"### {step:02d} / {len(stages):02d} · {name}")
    st.write(explanation)
    st.caption("This is an implementation-stage explanation, not a newly trained model.")
with right:
    st.metric("Tensor / output shape", shape)

with st.expander("Show the complete ordered pathway", expanded=True):
    for idx, (stage_name, stage_shape, _) in enumerate(stages, start=1):
        marker = "🔷" if idx == step else "▫️"
        label = f"**{idx:02d} {stage_name}** — `{stage_shape}`"
        st.markdown(f"{marker} {label}")

st.divider()
st.subheader("Four-qubit circuit · synthetic input demonstration")
st.caption("The simulator follows the archived RX → two fixed RY/CNOT-ring layers → Pauli-Z sequence, using seed 42. It is a NumPy state-vector implementation, not a PennyLane runtime or a clinical predictor.")
st.link_button("Inspect original PennyLane quantum utility", source_url("05_Source_Code/utilities/quantum_utils.py"), icon="🔗")

with st.expander("View the actual fixed quantum gate sequence", expanded=False):
    st.code("""for wire in (0,1,2,3): RX(pixel[wire], wire)
for layer in (0,1):
    for wire in (0,1,2,3): RY(weights[layer,wire], wire)
    CNOT(0,1); CNOT(1,2); CNOT(2,3); CNOT(3,0)
return [expectation(PauliZ(w)) for w in (0,1,2,3)]""", language="python")
    st.caption("Angles are normalized pixel values in radians, with no extra ×π scaling. Fixed RY weights do not train.")

pixels = []
cols = st.columns(4)
for idx, col in enumerate(cols):
    with col:
        pixels.append(st.slider(f"Synthetic pixel {idx + 1}", 0.0, 1.0,
                                value=(0.2, 0.4, 0.6, 0.8)[idx], step=0.05,
                                key=f"synthetic_pixel_{idx}"))

out = simulate_patch(pixels)
fig = go.Figure(go.Bar(x=[f"Z{idx}" for idx in range(4)], y=out,
                       marker_color=["#4894C2", "#7A82C6", "#53A6A0", "#A689C0"],
                       text=[f"{x:+.4f}" for x in out], textposition="auto"))
fig.update_layout(height=310, margin=dict(l=15,r=15,t=20,b=5),
                  yaxis=dict(range=[-1.10,1.10], title="Pauli-Z expectation (−1 to +1)"),
                  xaxis_title="Measured qubit", template="plotly_dark", showlegend=False)
st.plotly_chart(fig, width="stretch")
st.caption("Values are expectation measurements in [−1, +1] — not classification probabilities or accuracy scores.")
with st.expander("Show the two fixed rotation layers"):
    st.dataframe(fixed_weights(), width="stretch",
                 column_config={str(i): st.column_config.NumberColumn(format="%.6f") for i in range(4)})

st.divider()
st.markdown("**Research boundary:** Phase 6/9 are archived full-image quantum-feature models; Phase 10 is the matched grayscale CNN; Phase 14 is a strict cropped-ROI classical CNN; the later fixed-feature extension is a separate development-only case-fold study. Do not compare them as if they share a single test protocol.")
