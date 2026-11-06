"""QuCNet Research Explorer — verified-evidence overview."""
import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="QuCNet Research Explorer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1200px;}
    h1 {letter-spacing: -.04em;}
    [data-testid="stMetric"] {background: rgba(100, 130, 170, 0.08);
       padding: 18px; border-radius: 14px; border: 1px solid rgba(120, 140, 170, .17);}
    </style>
    """,
    unsafe_allow_html=True,
)


def load_json(relative_path: str) -> dict:
    path = ROOT / relative_path
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Cannot read {relative_path}: {exc}")
        st.stop()


research = load_json("data/research_snapshot.json")
extension = load_json("data/extension_snapshot.json")
provenance = load_json("provenance.json")


st.title("QuCNet Research Explorer")
st.markdown(
    "**An Interactive Architecture, Reproducibility and Evaluation-Audit Workbench**  "
    "for Quantum-Inspired Thyroid Ultrasound Classification"
)
st.info(
    "This application explains archived experimental evidence. "
    "It does not retrain models, diagnose patients, or demonstrate quantum advantage."
)

tab1, tab2 = st.tabs(["Research results", "Provenance and scope"])

with tab1:
    st.subheader("Verified historical findings")
    first, second, third = st.columns(3)
    with first:
        st.metric("Binary QuCNet · 70 images", f"{research['binary_qucnet']['accuracy'] * 100:.2f}%")
        st.caption("Validation-selected historical holdout; TI-RADS-derived suspicion proxies")
    with second:
        st.metric("Six-class quantum · 70 images", f"{research['six_class_historical']['quantum_accuracy'] * 100:.2f}%")
        st.caption("Historical image-level six-class comparison")
    with third:
        st.metric("Strict ROI CNN · 71 ROIs", f"{research['strict_roi_cnn']['roi_accuracy'] * 100:.2f}%")
        st.caption("Separate recorded-case-ID-disjoint evaluation")

    st.subheader("Controlled six-class comparison")
    six = research["six_class_historical"]
    names = ["Quantum-feature CNN", "Classical grayscale CNN", "Majority reference"]
    scores = [
        100 * six["quantum_accuracy"],
        100 * six["classical_accuracy"],
        100 * six["majority_accuracy"],
    ]
    chart = go.Figure(go.Bar(
        x=names, y=scores, text=[f"{value:.2f}%" for value in scores],
        textposition="outside", marker_color=["#7268D8", "#258AB6", "#9AABB5"]
    ))
    chart.update_layout(
        yaxis_title="Accuracy (%)", yaxis_range=[0, 38],
        height=360, margin=dict(l=20, r=20, t=30, b=30),
    )
    st.plotly_chart(chart, width="stretch")
    st.caption("All three figures above use the historical 70-image six-class comparison; they are not Phase 14 or development-fold results.")

    st.subheader("Later research extension")
    st.write(
        "A separate development-only analysis evaluated eight classifier/feature "
        f"configurations on **{extension['classifier_comparison']['case_count']} recorded case IDs** "
        "using case-grouped out-of-fold predictions. It is not an independent clinical test."
    )
    left, right = st.columns(2)
    with left:
        model = extension["classifier_comparison"]["models"]["fixed_quantum_64__random_forest"]
        st.metric("Fixed quantum · random forest", f"{model['accuracy'] * 100:.2f}%")
    with right:
        st.metric("Strict ROI · case-level accuracy", f"{research['strict_roi_cnn']['case_accuracy'] * 100:.2f}%")
    st.warning("These two measurements use different tasks and evaluation samples and must not be compared as if they share a test set.")

with tab2:
    st.subheader("Traceable research evidence")
    st.write("**Frozen source repository:**", provenance["research_repository"])
    st.code(provenance["source_commit"], language=None)
    st.write("**Locally verified JSON source files:**", len(provenance["source_file_sha256"]) + len(extension["source_file_sha256"]))
    st.write("**Recorded case-ID overlap in strict partitions:**", research["case_split"]["recorded_case_id_overlap"])
    st.warning("Recorded case-ID separation does not establish independent patient identities. Source TI-RADS labels are not pathology-confirmed diagnoses.")
    st.caption("No raw DDTI JPEG/XML files, case IDs, individual predictions, probability arrays or model checkpoints are loaded by this app.")
