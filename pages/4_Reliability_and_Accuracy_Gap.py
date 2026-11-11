"""Research reliability and accuracy-gap analysis from archived aggregates only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evidence.dataset_correspondence import render_dataset_correspondence
from src.evidence.reliability_math import (  # noqa: E402
    LABELS, classwise_from_confusion, development_scores,
    validate_bn_conditions,
)

st.set_page_config(page_title="Reliability & Accuracy Gap | QuCNet",
                   page_icon="🛡️", layout="wide")
st.markdown("""<style>
.block-container {max-width:1240px;padding-top:2rem;}
h1 {letter-spacing:-.038em;}
[data-testid="stMetric"] {background:rgba(110,135,165,.08);border:1px solid rgba(135,150,185,.18);border-radius:14px;padding:16px;}
</style>""", unsafe_allow_html=True)
REPO = "https://github.com/karthikaxiom/QCNN_Thyroid_Paper_Reproduction"
COLORS = ["#55AED0", "#9D85E4", "#DFA666", "#47B49E", "#DA86AB", "#E16C71"]


def load_json(name):
    try:
        return json.loads((ROOT / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        st.error(f"Cannot load verified evidence: {name}. Restore the locally verified snapshot before continuing.")
        st.stop()


def evidence_link(path, label):
    st.link_button(label, f"{REPO}/blob/{commit}/{path}")


def style(fig, height=390):
    fig.update_layout(template="plotly_dark", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12), margin=dict(l=35, r=25, t=40, b=50),
        legend=dict(orientation="h", y=1.12, x=0))
    fig.update_yaxes(gridcolor="rgba(160,175,200,.14)")
    return fig


hist = load_json("data/research_snapshot.json")
ext = load_json("data/extension_snapshot.json")
prov = load_json("provenance.json")
dataset_snapshot = load_json("data/dataset_integrity_snapshot.json")
commit = prov["source_commit"]
if ext["source_commit"] != commit or dataset_snapshot["source_commit"] != commit:
    st.error("Research-source commit mismatch. Evidence display is blocked.")
    st.stop()

from src.evidence.visual_enhancements import render_phase14_figures

st.title("Reliability & Accuracy-Gap Analysis")
st.caption("Source fidelity · class-specific failures · controlled diagnostics · scientific boundaries")
st.info("This page examines completed research. Observations, possible mechanisms and unsupported claims are kept separate. No training or medical inference occurs.")

m1, m2, m3 = st.columns(3)
m1.metric("Historical binary QuCNet", f"{hist['binary_qucnet']['accuracy']*100:.2f}%",
          help="70-image validation-selected holdout; TI-RADS proxy")
m2.metric("Historical six-class quantum", f"{hist['six_class_historical']['quantum_accuracy']*100:.2f}%",
          help="Different 70-image historical six-class task")
m3.metric("Strict ROI CNN", f"{hist['strict_roi_cnn']['roi_accuracy']*100:.2f}%",
          help="Different case-ID-separated 71-ROI historical evaluation")
st.caption("These three headline scores are not from a common evaluation set and should not be directly ranked.")

comparison, failure, diagnostic, boundaries = st.tabs([
    "Paper vs implementation", "Class-specific failures",
    "Preprocessing & diagnostics", "Findings vs hypotheses"
])

with comparison:
    st.subheader("Why the published accuracy cannot be compared directly")
    st.write("The source paper reported **97.63%** binary tumour classification on **127 images** and **93.87%** six-category suspicious-level classification on **98 images**. The reconstructed study and later audits used different recoverable labels, sample inventories, processing, simulation and evaluation protocols.")
    render_dataset_correspondence(st, dataset_snapshot, hist, commit)
    st.divider()
    aspects = {
        "Target labels": (
            "Tumour classification and six-category suspicion grading were reported.",
            "The recovered binary outcome is a TI-RADS-derived low/high suspicion proxy, not confirmed pathology. Six-class labels are source-assigned TI-RADS 2/3/4a/4b/4c/5.",
            "The binary endpoints are not ground-truth equivalent. Comparing accuracies as if both measured pathology diagnosis is invalid."),
        "Dataset inventory": (
            "Paper describes 478 images / 399 cases, and original test sets of 127 and 98 images.",
            "Recovered snapshot contains 480 JPG files, 390 XML records and 349 labelled source images. Historical local evaluations used 70 images.",
            "The exact original test samples and data-version correspondence are not independently established."),
        "Preprocessing": (
            "Paper describes mean subtraction / standard deviation scaling and specific augmentation.",
            "Historical pipeline used grayscale scaling to [0, 1]. Strict 64x64 ROI experiments later used cropped, padded lesions.",
            "Preprocessing is not source-exact; whether paper-like scaling would improve accuracy has not been measured here."),
        "Quantum execution": (
            "Original publication reported a Qiskit simulator and describes a random four-qubit circuit at a broad level.",
            "Four-qubit PennyLane default.qubit CPU simulation; RX encoding, two fixed seeded RY/CNOT-ring layers and four Pauli-Z outputs.",
            "The archived implementation has structural fidelity, not gate-for-gate evidence of the paper's undisclosed random circuit."),
        "Sampling and selection": (
            "Source describes an 80/20 image-level split; exact case membership not recoverable.",
            "Phase 6 reused the 70-image holdout for checkpoint selection. Earlier image splits shared some case IDs; strict Phase 12/14 splits are recorded-case-ID disjoint.",
            "A case-disjoint split is not the source's original image protocol; recorded case IDs do not verify unique patients."),
        "Classifier configuration": (
            "Two-convolution six-class design; some exact widths and circuit/training settings are unavailable.",
            "Phase 9 uses Conv2D 16/32, pooling and Dense 128/64; Phase 14 is a separate BatchNorm ROI CNN.",
            "These are traceable reconstruction choices, not source-exact original hyperparameters."),
    }
    choice = st.selectbox("Inspect a methodological difference", list(aspects))
    paper, implementation, meaning = aspects[choice]
    left, right = st.columns(2)
    with left:
        st.markdown("#### Published-paper description")
        st.write(paper)
    with right:
        st.markdown("#### Archived local implementation")
        st.write(implementation)
    st.warning("**Interpretation:** " + meaning)
    evidence_link("Research_Extension/ORIGINAL_PAPER_FULLTEXT_FIDELITY_AUDIT.md", "Inspect original paper-to-code audit")
    st.divider()
    st.subheader("Synthetic normalization demonstration")
    st.caption("Educational calculation, not a new training experiment. Synthetic mean and standard deviation are freely chosen, not estimates of the DDTI dataset.")
    pixel = st.slider("Example grayscale pixel", 0, 255, 128)
    c1, c2 = st.columns(2)
    with c1:
        mean = st.slider("Illustrative dataset mean (0–255)", 0.0, 255.0, 110.0, 1.0)
    with c2:
        sigma = st.slider("Illustrative dataset standard deviation", 1.0, 128.0, 50.0, 1.0)
    norm1, norm2 = st.columns(2)
    norm1.metric("Scale to [0, 1]", f"{pixel/255:.4f}")
    norm2.metric("Illustrative (pixel − mean) / std", f"{(pixel-mean)/sigma:.4f}")
    st.caption("Changing input scaling changes numerical representation, but no accuracy improvement can be deduced without a controlled new experiment.")

with failure:
    st.subheader("Recorded Phase 14 prediction collapse")
    st.write("The strict, case-ID-separated ROI CNN predicted **TI-RADS 3 for all 71 test ROIs** and all **44 recorded test cases**. Only **4/71 ROI predictions (5.63%)** and **2/44 case predictions (4.55%)** matched the true label.")
    c1, c2 = st.columns(2)
    c1.metric("Correct ROI predictions", "4 / 71")
    c2.metric("Correct case predictions", "2 / 44")
    st.warning("This is the *classical Phase 14 ROI CNN* failure—not a quantum-model result. The recorded outcome does not isolate its cause.")
    evidence_link("07_Results/phase14_roi_categorical_baseline/evaluation_metrics.json", "Inspect strict ROI evaluation")
    render_phase14_figures(st, ROOT, commit)
    st.divider()
    st.subheader("Later matched-classifier classwise sensitivity")
    st.caption("All options below share 203 recorded training case IDs in fivefold development OOF evaluation; they are not a new untouched test cohort.")
    classifiers = {
        "Classical · logistic regression": "classical_64__linear_logistic",
        "Fixed quantum · logistic regression": "fixed_quantum_64__linear_logistic",
        "Classical · random forest": "classical_64__random_forest",
        "Fixed quantum · random forest": "fixed_quantum_64__random_forest",
    }
    name = st.selectbox("Select an archived classifier", list(classifiers), index=3)
    selection = classifiers[name]
    record = ext["classifier_comparison"]["models"][selection]
    try:
        summary = classwise_from_confusion(record["confusion_matrix"])
    except ValueError:
        st.error("Invalid saved confusion matrix. Classwise chart is blocked.")
        st.stop()
    if summary["n"] != 203 or abs(summary["accuracy"] - record["accuracy"]) > 1e-9:
        st.error("Classwise confusion matrix conflicts with saved metric; chart blocked.")
        st.stop()
    a, b, c = st.columns(3)
    a.metric("Cases in development OOF", summary["n"])
    b.metric("Accuracy", f"{summary['accuracy']*100:.2f}%")
    c.metric("Correct cases", summary["correct"])
    recall = [value*100 if value is not None else None for value in summary["recall"]]
    fig = go.Figure(go.Bar(x=["TI-RADS " + lab for lab in LABELS], y=recall,
        text=[f"{value:.1f}%" if value is not None else "N/A" for value in recall],
        textposition="outside", marker_color=COLORS,
        customdata=summary["support"],
        hovertemplate="%{x}<br>Recall %{y:.1f}%<br>Support %{customdata} cases<extra></extra>"))
    fig.update_yaxes(title="Sensitivity / recall (%)", range=[0,110])
    st.plotly_chart(style(fig), width="stretch")
    st.caption("A higher overall accuracy can coexist with a complete failure to identify a minority class. The two random forest variants had 0/10 TI-RADS 3 recall at all five repeated case-fold seeds.")
    evidence_link("07_Results/exp_models_001/metrics.json", "Inspect matched-classifier aggregate metrics")
    evidence_link("07_Results/exp_stability_001/metrics.json", "Inspect repeated-fold findings")

with diagnostic:
    st.subheader("Development-only preprocessing sensitivity")
    st.caption("These comparisons use the same 203 training case IDs, with case-grouped out-of-fold evaluation. No historical test data was retuned.")
    study = st.radio("Choose one controlled development comparison",
                     ["preprocessing", "fusion"], horizontal=True,
                     format_func=lambda s: "ROI preprocessing" if s == "preprocessing" else "Image vs geometry fusion")
    metric = st.selectbox("Metric", ["accuracy", "balanced_accuracy", "macro_f1"],
                          format_func=lambda value: value.replace("_", " ").title())
    naming = {"median_original": "Median-filled padding", "reflection_padding": "Reflection padding",
              "direct_stretch": "Direct resize (aspect distortion)", "image": "Image features",
              "geometry": "ROI geometry", "fusion": "Image + geometry", "majority": "Majority predictor"}
    try:
        rows = development_scores(ext, study)
    except (KeyError, ValueError):
        st.error("Archived comparison is incomplete or inconsistent.")
        st.stop()
    fig = go.Figure(go.Bar(x=[naming.get(row["name"], row["name"]) for row in rows],
        y=[row[metric]*100 for row in rows],
        text=[f"{row[metric]*100:.2f}%" for row in rows],
        textposition="outside", marker_color=COLORS[:len(rows)]))
    fig.update_yaxes(title=f"{metric.replace('_',' ').title()} (%)", range=[0,max(r[metric] for r in rows)*125])
    st.plotly_chart(style(fig, 380), width="stretch")
    if study == "fusion":
        st.write("Image-only classified **45/203** cases correctly; geometry-only **39/203**; fusion **46/203**. The one-case difference versus image-only was not a statistically established improvement (paired exact McNemar **p = 1.00**).")
        evidence_link("07_Results/exp_fusion_001/metrics.json", "Inspect fusion study evidence")
    else:
        st.write("Direct resizing changed the observed result relative to square padding, but it also distorts lesion aspect ratio. These exploratory results do not establish a best preprocessing strategy.")
        evidence_link("07_Results/exp_preproc_001/metrics.json", "Inspect preprocessing study evidence")
    st.divider()
    st.subheader("Separate BatchNorm running-statistics diagnostic")
    st.caption("Historical *validation*, n=58; not the 203-case development-fold comparison, nor a new test result. Learned network weights were not updated.")
    try:
        bn = validate_bn_conditions(ext["batchnorm"])
    except (KeyError, ValueError):
        st.error("BatchNorm evidence failed validation.")
        st.stop()
    fig = go.Figure(go.Bar(x=["Original stored statistics", "Recalibrated statistics", "Validation majority reference"],
        y=[100*bn["original"], 100*bn["recalibrated"], 100*bn["majority"]],
        marker_color=["#D67980", "#5AB3A5", "#A6AFBD"], text=["2 / 58", "11 / 58", "17 / 58"],
        textposition="outside"))
    fig.update_yaxes(title="Validation accuracy (%)", range=[0,40])
    st.plotly_chart(style(fig, 315), width="stretch")
    st.warning("This supports an effect of BatchNorm running statistics on saved checkpoint behaviour. It does **not** prove that BatchNorm was the sole cause of single-class collapse or establish improved test performance.")
    evidence_link("07_Results/exp_bn_001/metrics.json", "Inspect BatchNorm diagnostic evidence")

with boundaries:
    st.subheader("Evidence-supported conclusions and limitations")
    st.markdown("#### Verified observations")
    st.success("The public research reconstruction did not match the publication's data/label/protocol details; earlier image splits had recorded-case overlaps; the strict ROI CNN collapsed to a single class; classwise failures persisted in later development comparisons.")
    st.markdown("#### Possible explanations, not independently proven causes")
    st.warning("Small and imbalanced TI-RADS classes, target mismatch, incomplete circuit details, preprocessing differences, correlated quantum channels and BatchNorm statistics may contribute to weak generalization. The available experiments do not isolate their causal effects.")
    st.markdown("#### Conclusions the research does not establish")
    st.error("No pathology-confirmed diagnostic accuracy, no quantum computational advantage, no clinical deployment readiness, no uniquely verified patients and no compatible independently validated six-class external cohort.")
    st.divider()
    st.markdown("**Research question:** *Why did the reconstructed experiments yield lower accuracies than the published study?*")
    st.write("**Evidence-based interpretation:** The published and reconstructed studies differ in label provenance, data inventory, implementation choices and evaluation protocols. The reconstructed models achieved low accuracy under their respective documented settings. The audit identifies limitations but does not establish a single causal explanation, nor does it reproduce the authors' original test conditions.")
    evidence_link("Research_Extension/FINAL_RESEARCH_VERIFICATION_MATRIX.md", "Read verified research-evidence matrix")
    evidence_link("Research_Extension/ORIGINAL_PAPER_FULLTEXT_FIDELITY_AUDIT.md", "Read full-text method-fidelity audit")

st.divider()
st.caption(f"Evidence provenance: {commit[:12]} · Aggregate-only educational research workbench · No raw clinical imagery or patient-level data.")
