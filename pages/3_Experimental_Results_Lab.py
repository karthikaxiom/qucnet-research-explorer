"""Source-backed Experimental Results Lab; aggregates only, no inference or training."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.evidence.results_math import BinaryConfusion, multiclass_summary

st.set_page_config(page_title="Experimental Results | QuCNet", page_icon="📊", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1240px; padding-top: 2rem;}
h1 {letter-spacing:-.038em;}
[data-testid="stMetric"] {background:rgba(110,135,165,.08);border:1px solid rgba(135,150,185,.18);border-radius:14px;padding:16px;}
</style>""", unsafe_allow_html=True)

REPO = "https://github.com/karthikaxiom/QCNN_Thyroid_Paper_Reproduction"
LABELS = ["2", "3", "4a", "4b", "4c", "5"]
FEATURES = {"classical_64": "Classical 64-D", "fixed_quantum_64": "Fixed quantum 64-D"}
ALGORITHMS = {"linear_logistic": "Logistic regression", "rbf_svm": "RBF SVM", "random_forest": "Random forest", "knn_7": "7-NN"}
PALETTE = {"classical_64": "#42A6C9", "fixed_quantum_64": "#9781E6"}


def load(name):
    try:
        return json.loads((ROOT / name).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, KeyError) as ex:
        st.error(f"Verified evidence unavailable: {name}")
        st.caption(f"Details: {type(ex).__name__}")
        st.stop()


def chart_style(fig, height=380):
    fig.update_layout(template="plotly_dark", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=25, r=20, t=35, b=40), font=dict(size=12),
        legend=dict(orientation="h", y=1.16, x=0))
    fig.update_yaxes(gridcolor="rgba(160,180,200,.14)")
    return fig


def heatmap(matrix, names, heading):
    """One count per matrix cell, positioned by stable numeric grid indices."""
    n = len(names)
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError("Confusion matrix dimensions do not match the class labels")
    positions = list(range(n))
    hover_labels = [
        [[names[row], names[col]] for col in positions]
        for row in positions
    ]
    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=positions,
        y=positions,
        customdata=hover_labels,
        colorscale=[[0, "#172238"], [0.5, "#1F6689"], [1, "#3D90B4"]],
        texttemplate="%{z}",
        textfont=dict(color="white", size=14),
        showscale=True,
        colorbar=dict(title="Count"),
        hovertemplate=(
            "Actual %{customdata[0]}<br>"
            "Predicted %{customdata[1]}<br>"
            "Count %{z}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=heading,
        xaxis=dict(title="Predicted", tickmode="array", tickvals=positions, ticktext=names),
        yaxis=dict(title="Actual", tickmode="array", tickvals=positions,
                   ticktext=names, autorange="reversed"),
    )
    return chart_style(fig, 385 if n == 6 else 315)


def source_link(path, label):
    st.link_button(label, f"{REPO}/blob/{sha}/{path}")


def percentage(value):
    return "Undefined" if value is None else f"{value*100:.2f}%"


historical = load("data/research_snapshot.json")
extension = load("data/extension_snapshot.json")
prov = load("provenance.json")
sha = prov["source_commit"]
if sha != extension["source_commit"]:
    st.error("Research source commit disagreement; results not displayed.")
    st.stop()

from src.evidence.visual_enhancements import render_binary_provenance, render_additional_gallery

st.title("Experimental Results Laboratory")
st.caption("Measured historical evidence · development-only classifier studies · transparent mathematical what-if experiments")
st.info("This workbench reads saved aggregate results. It does not load checkpoints, predict medical diagnoses, or retrain QuCNet.")

t_binary, t_six, t_models, t_robust, t_figs = st.tabs([
    "Binary audit & what-if", "Six-class comparison", "Eight classifier configurations",
    "Stability & method sensitivity", "Research figure gallery"
])

with t_binary:
    bin_data = historical["binary_qucnet"]
    cm = bin_data["confusion_matrix"]
    original = BinaryConfusion(tn=cm["true_negative"], fp=cm["false_positive"],
                               fn=cm["false_negative"], tp=cm["true_positive"])
    st.subheader("Phase 16 · Binary QuCNet")
    st.caption("70-image validation-selected historical holdout. Positive label = TI-RADS-derived high-suspicion proxy, not pathology-verified malignancy.")
    a,b,c,d = st.columns(4)
    a.metric("Accuracy", percentage(original.metrics()["accuracy"]), help="(TP + TN) / 70")
    b.metric("Precision (not accuracy)", percentage(original.metrics()["precision"]))
    c.metric("Recall / sensitivity", percentage(original.metrics()["recall"]))
    d.metric("Specificity", percentage(original.metrics()["specificity"]))
    l,r = st.columns([1.1,1])
    with l:
        st.plotly_chart(heatmap([[original.tn, original.fp], [original.fn, original.tp]],
                                ["Low suspicion", "High suspicion"], "Verified archived confusion matrix"), width="stretch")
    with r:
        st.markdown("**Interpretation**")
        st.write("The classifier identified **32 of 39** high-suspicion proxy cases, but correctly rejected only **6 of 31** low-suspicion proxy cases. **25 false positives** contributed to weak specificity.")
        st.write(f"ROC-AUC **{bin_data['roc_auc']:.4f}** · Average precision **{bin_data['average_precision']:.4f}** · F1 **{percentage(original.metrics()['f1'])}**.")
        st.write("Derived same-holdout majority reference: **39/70 = 55.71%** (calculated, not a reported model result).")
        source_link("07_Results/phase16_binary_qucnet_evaluation/binary_evaluation_metrics.json", "Inspect original binary metrics")

    st.divider()
    render_binary_provenance(st, ROOT, sha)
    st.subheader("Metric sensitivity analysis · changes in error counts")
    st.warning("Illustrative calculation, not an experimental result. The recorded class totals remain fixed at 31 low-suspicion and 39 high-suspicion images; only the assumed error counts change.")
    st.caption("Adjust assumed false-positive and false-negative counts to examine their effect on metrics. Values of 25 and 7 reproduce the archived confusion matrix.")
    left,right = st.columns(2)
    with left:
        fp = st.slider("False positives (out of 31 low-suspicion images)", 0, 31, 25,
                       key="qucnet_whatif_fp")
    with right:
        fn = st.slider("False negatives (out of 39 high-suspicion images)", 0, 39, 7,
                       key="qucnet_whatif_fn")
    hypothetical = BinaryConfusion(tn=31-fp, fp=fp, fn=fn, tp=39-fn)
    h = hypothetical.metrics()
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Scenario accuracy", percentage(h["accuracy"]),
              delta=f"{(h['accuracy']-original.metrics()['accuracy'])*100:+.2f} pp vs archived")
    m2.metric("Scenario precision", percentage(h["precision"]))
    m3.metric("Scenario recall", percentage(h["recall"]))
    m4.metric("Scenario specificity", percentage(h["specificity"]))
    st.plotly_chart(heatmap([[hypothetical.tn, fp], [fn, hypothetical.tp]],
                            ["Low suspicion", "High suspicion"], "Hypothetical scenario — not a saved prediction"),
                    width="stretch")
    st.caption(f"Calculated F1: {percentage(h['f1'])}. These are arithmetic scenarios, not additional model predictions or evidence of improved performance.")

with t_six:
    six = historical["six_class_historical"]
    st.subheader("Controlled historical six-class TI-RADS comparison")
    st.caption("All three accuracy values below belong to the same historical 70-image six-class comparison—not the ROI or development-fold experiments.")
    rows = [
        ("Quantum-feature QuCNet", six["quantum_accuracy"], six["quantum_balanced_accuracy"], six["quantum_macro_f1"]),
        ("Classical grayscale CNN", six["classical_accuracy"], six["classical_balanced_accuracy"], six["classical_macro_f1"]),
        ("Majority-class reference", six["majority_accuracy"], None, None),
    ]
    select_metric = st.radio("Compare", ["Accuracy", "Balanced accuracy", "Macro-F1"], horizontal=True)
    field = {"Accuracy":1, "Balanced accuracy":2, "Macro-F1":3}[select_metric]
    current = [x for x in rows if x[field] is not None]
    chart = go.Figure(go.Bar(x=[x[0] for x in current], y=[x[field]*100 for x in current],
        marker_color=["#927CE5", "#3AA6C6", "#A6AFBD"][:len(current)],
        text=[f"{x[field]*100:.2f}%" for x in current], textposition="outside"))
    chart.update_layout(yaxis_title=select_metric+" (%)")
    chart.update_yaxes(range=[0, max(x[field]*100 for x in current)*1.30])
    st.plotly_chart(chart_style(chart), width="stretch")
    st.caption("The reconstructed quantum model did not outperform the matched classical model. Historical image-level case overlap limits independence; published-paper results use different tasks and evaluation samples.")
    l,r = st.columns(2)
    with l:
        source_link("07_Results/phase9_paper_tirads_class_weighted/test_metrics.json", "Phase 9 quantum evidence")
    with r:
        source_link("07_Results/phase10_classical_tirads_class_weighted/test_metrics.json", "Phase 10 classical evidence")

with t_models:
    data = extension["classifier_comparison"]
    st.subheader("Matched development-fold classifier comparison")
    st.caption("Fivefold case-grouped out-of-fold analysis using 311 training ROIs from 203 recorded case IDs. The fixed quantum feature map is not a trained quantum CNN.")
    selection = st.selectbox("Classifier family", list(ALGORITHMS), format_func=lambda x:ALGORITHMS[x])
    l,r = st.columns(2)
    with l:
        classical = data["models"][f"classical_64__{selection}"]
        st.metric("Classical 64-D accuracy", percentage(classical["accuracy"]))
    with r:
        quantum = data["models"][f"fixed_quantum_64__{selection}"]
        st.metric("Fixed quantum 64-D accuracy", percentage(quantum["accuracy"]))
    compare_metric = st.selectbox("Comparison metric", ["accuracy", "balanced_accuracy", "macro_f1"],
                                   format_func=lambda x:x.replace("_", " ").capitalize())
    keys = list(ALGORITHMS)
    fig = go.Figure()
    for map_name,map_label in FEATURES.items():
        values = [data["models"][f"{map_name}__{k}"][compare_metric]*100 for k in keys]
        fig.add_bar(name=map_label, x=[ALGORITHMS[k] for k in keys], y=values,
                    marker_color=PALETTE[map_name], text=[f"{v:.2f}%" for v in values],
                    textposition="outside")
    fig.update_layout(barmode="group", yaxis_title=compare_metric.replace("_", " ").title()+" (%)")
    fig.update_yaxes(range=[0, max(36, max(fig.data[0].y)*1.3, max(fig.data[1].y)*1.3)])
    st.plotly_chart(chart_style(fig, 405), width="stretch")
    view = st.radio("Inspect one archived confusion matrix", list(FEATURES), horizontal=True,
                    format_func=lambda x:FEATURES[x])
    selected = data["models"][f"{view}__{selection}"]
    matrix = selected["confusion_matrix"]
    summary = multiclass_summary(matrix)
    if summary["n"] != data["case_count"] or abs(summary["accuracy"]-selected["accuracy"])>1e-12:
        st.error("Confusion-matrix aggregate consistency error; cannot show the matrix.")
        st.stop()
    x,y = st.columns([1.15,1])
    with x:
        st.plotly_chart(heatmap(matrix, LABELS, f"{FEATURES[view]} · {ALGORITHMS[selection]}"), width="stretch")
    with y:
        st.markdown("**Recall by TI-RADS category**")
        rec = summary["class_recall"]
        recall_fig=go.Figure(go.Bar(x=[f"TI-RADS {n}" for n in LABELS], y=[v*100 if v is not None else 0 for v in rec],
            marker_color="#58AFCE", text=[percentage(v) for v in rec], textposition="outside"))
        recall_fig.update_yaxes(title="Recall (%)", range=[0,110])
        st.plotly_chart(chart_style(recall_fig, 385), width="stretch")
    st.warning("A higher aggregate accuracy does not guarantee good minority-class sensitivity. Both random-forest variants missed all 10 TI-RADS 3 cases in the seed-42 comparison. No quantum superiority was established.")
    source_link("07_Results/exp_models_001/metrics.json", "Inspect eight-classifier source metrics")

with t_robust:
    st.subheader("Fold stability · Preprocessing · Fusion · BatchNorm")
    st.caption("These findings are exploratory and do not represent a single common independent test set.")
    option = st.selectbox("Analysis", ["Five-seed case-fold stability", "Preprocessing sensitivity", "Image/geometry fusion", "BatchNorm validation diagnostic"])
    if option == "Five-seed case-fold stability":
        data = extension["stability"]
        family = st.radio("Classifier", ["linear_logistic", "random_forest"], horizontal=True,
                          format_func=lambda x: ALGORITHMS[x])
        metric = st.selectbox("Metric", ["accuracy", "balanced_accuracy", "macro_f1"],
                              format_func=lambda x: x.replace("_", " ").capitalize())
        fig = go.Figure()
        for map_name,map_label in FEATURES.items():
            value = data["models"][f"{map_name}__{family}"][metric]["per_seed"]
            fig.add_scatter(x=data["seeds"], y=[v*100 for v in value],
                            mode="lines+markers+text", name=map_label,
                            line=dict(color=PALETTE[map_name], width=3),
                            text=[f"{v*100:.2f}%" for v in value], textposition="top center")
        fig.update_xaxes(type="category", title="Case-fold assignment seed")
        fig.update_yaxes(title=metric.replace("_", " ").capitalize()+" (%)")
        st.plotly_chart(chart_style(fig,410),width="stretch")
        st.caption("The same 203 cases were evaluated under different fold assignments. These are correlated repeat estimates—not five independent patient cohorts.")
        source_link("07_Results/exp_stability_001/metrics.json", "Inspect stability evidence")
    elif option in ("Preprocessing sensitivity", "Image/geometry fusion"):
        key = "preprocessing" if option.startswith("Preprocessing") else "fusion"
        rec = extension[key]
        titles = {"median_original":"Median square padding", "reflection_padding":"Reflection padding", "direct_stretch":"Direct resizing",
                  "majority":"Majority reference", "image":"Image only", "geometry":"ROI geometry", "fusion":"Image + geometry"}
        fig = go.Figure(go.Bar(x=[titles.get(k,k) for k in rec["models"]],
            y=[v["accuracy"]*100 for v in rec["models"].values()],
            text=[f"{v['accuracy']*100:.2f}%" for v in rec["models"].values()],
            marker_color=["#52AACB", "#9C84DB", "#E1AB68", "#65B4AD"][:len(rec["models"])],
            textposition="outside"))
        fig.update_yaxes(title="Development-case OOF accuracy (%)", range=[0,38])
        st.plotly_chart(chart_style(fig), width="stretch")
        if key == "fusion":
            st.caption("Image-only 45/203; geometry-only 39/203; fusion 46/203. The +1 case gain versus image-only was not statistically established (exact McNemar p=1.00).")
            source_link("07_Results/exp_fusion_001/metrics.json", "Inspect fusion evidence")
        else:
            st.caption("Changing padding/resizing altered development accuracy; direct resizing also distorts lesion aspect ratio. This analysis does not establish a causal explanation for the Phase 14 failure.")
            source_link("07_Results/exp_preproc_001/metrics.json", "Inspect preprocessing evidence")
    else:
        data=extension["batchnorm"]["conditions"]
        c1,c2,c3=st.columns(3)
        c1.metric("Original validation accuracy", percentage(data["original"]["accuracy"]))
        c2.metric("Recalibrated validation accuracy", percentage(data["recalibrated"]["accuracy"]))
        c3.metric("Validation majority reference", percentage(data["original"]["majority_reference"]))
        st.warning("Only seven BatchNorm running-statistic sets were re-estimated from training ROIs. Learned weights were not updated; test data were not accessed. This does not prove BatchNorm was the sole cause of prediction collapse.")
        source_link("07_Results/exp_bn_001/metrics.json", "Inspect BatchNorm diagnostic")

with t_figs:
    st.subheader("Saved figure evidence")
    st.caption("Figures are copied from an explicit aggregate-only whitelist in the frozen research repository. They are optional; no raw medical imagery is included.")
    gallery = [
        ("Binary confusion matrix", "binary_confusion_matrix.png", "Historical binary proxy, n=70"),
        ("Binary ROC curve", "binary_roc_curve.png", "Historical binary proxy ROC-AUC 0.6634"),
        ("Binary precision-recall curve", "binary_precision_recall_curve.png", "Average precision 0.7487"),
        ("Six-class quantum confusion", "phase9_test_confusion_matrix.png", "Historical image-level evaluation, n=70"),
        ("Six-class classical confusion", "phase10_test_confusion_matrix.png", "Historical image-level evaluation, n=70"),
        ("Quantum training history", "phase9_training_accuracy_plot.png", "Saved Phase 9 training curve"),
        ("Classical training history", "phase10_training_accuracy_plot.png", "Saved Phase 10 training curve"),
        ("Matched classifier comparison", "fig_08_matched_classifier_families_color.svg", "Development-only case-fold comparison"),
    ]
    present = [(t,f,description) for t,f,description in gallery if (ROOT/"assets"/"figures"/f).is_file()]
    if not present:
        st.warning("Verified chart files have not been imported yet. Run src/evidence/import_public_figures.py once from the mini-project folder. Other tabs work without figures.")
    for i in range(0,len(present),2):
        columns=st.columns(2)
        for col,(title,file,desc) in zip(columns,present[i:i+2]):
            with col:
                st.markdown(f"**{title}**")
                st.image(str(ROOT/"assets"/"figures"/file), width="stretch")
                st.caption(desc)
    manifest_path=ROOT/"assets"/"figures"/"figure_manifest.json"
    if manifest_path.exists():
        with st.expander("Figure provenance manifest"):
            manifest=load("assets/figures/figure_manifest.json")
            for entry in manifest["figures"]:
                st.markdown(f"- [{entry['source_path']}]({REPO}/blob/{sha}/{entry['source_path']}) — SHA-256 `{entry['sha256'][:12]}`")

    render_additional_gallery(st, ROOT, sha)

st.divider()
st.caption(f"Source commit: {sha[:12]} · Archival evidence only · No patient-level predictions or clinical diagnostic claims")
