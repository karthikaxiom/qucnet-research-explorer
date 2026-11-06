"""QuCNet Research Explorer — privacy-safe dataset and integrity explorer."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data" / "dataset_integrity_snapshot.json"
LABEL_ORDER = ["2", "3", "4a", "4b", "4c", "5"]
COLORS = ["#36B8B5", "#70A1D7", "#8875DC", "#DD83AA", "#E7AF66", "#D56D75"]
SOURCE_REPO = "https://github.com/karthikaxiom/QCNN_Thyroid_Paper_Reproduction"

st.set_page_config(page_title="Dataset & Integrity | QuCNet", page_icon="🗂️", layout="wide")


def source_url(sha: str, path: str) -> str:
    return f"{SOURCE_REPO}/blob/{sha}/{path}"


def style_chart(fig: go.Figure, height: int = 370) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height,
        margin=dict(l=24, r=20, t=35, b=35),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13), legend=dict(orientation="h", y=1.13, x=0),
    )
    fig.update_yaxes(gridcolor="rgba(170,185,210,0.12)", zeroline=False)
    return fig


@st.cache_data(show_spinner=False)
def load_snapshot() -> dict:
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


st.title("Dataset & Integrity Explorer")
st.caption("DDTI inventory · TI-RADS missingness · frozen case-ID partitions · duplicate screening")
st.info("All figures use saved, aggregate research evidence. No medical images, XML or patient-level predictions are loaded.")

try:
    evidence = load_snapshot()
except (FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
    st.error("The verified dataset snapshot is missing or invalid. Run the Step 9 importer first.")
    st.code(".\\.venv\\Scripts\\python.exe src\\evidence\\build_dataset_snapshot.py", language="powershell")
    st.caption(f"Technical detail: {type(exc).__name__}")
    st.stop()

inventory = evidence["inventory"]
strict = evidence["strict_roi"]
dupes = evidence["raw_image_duplicates"]
sha = evidence["source_commit"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Recovered JPG images", f"{inventory['raw_jpg_images']:,}")
m2.metric("XML records", f"{inventory['xml_records']:,}")
m3.metric("Images with TI-RADS", f"{inventory['images_with_tirads']:,}")
m4.metric("Missing TI-RADS", f"{inventory['images_missing_tirads']:,}")

st.markdown("### A. Label coverage and six-category composition")
a, b = st.columns([1.4, 1])
with a:
    label_values = inventory["tirads_image_distribution"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["TI-RADS " + x for x in LABEL_ORDER],
        y=[label_values[x] for x in LABEL_ORDER],
        marker_color=COLORS,
        text=[label_values[x] for x in LABEL_ORDER],
        textposition="outside",
        hovertemplate="%{x}: %{y} labeled images<extra></extra>",
    ))
    fig.update_yaxes(title="Labeled source images")
    st.plotly_chart(style_chart(fig), width="stretch")
with b:
    fig = go.Figure(go.Pie(
        labels=["Recoverable TI-RADS", "Missing TI-RADS"],
        values=[inventory["images_with_tirads"], inventory["images_missing_tirads"]],
        marker=dict(colors=["#4B98C2", "#E6A660"]), hole=0.63,
        textinfo="percent+label",
        hovertemplate="%{label}: %{value} images (%{percent})<extra></extra>",
    ))
    fig.update_layout(showlegend=False)
    st.plotly_chart(style_chart(fig), width="stretch")

st.caption("These counts describe 480 recovered source JPG records. They do not represent 480 independently verified patients.")
st.link_button("Inspect the original TI-RADS inventory audit", source_url(sha, "07_Results/phase7_reproducibility_audit.txt"))

st.divider()
st.markdown("### B. Recorded-case-ID-separated ROI partitions")
q1, q2, q3 = st.columns(3)
q1.metric("Valid ROI geometries", strict["roi_geometries"])
q2.metric("Recorded case IDs", strict["recorded_case_ids"])
q3.metric("Case-ID overlaps", strict["case_id_overlap_between_splits"])

split_names = ["train", "validation", "test"]
split_labels = {"train": "Training", "validation": "Validation", "test": "Historical test"}
unit = st.radio("Compare by", ["ROIs", "Recorded case IDs"], horizontal=True)
key = "rois" if unit == "ROIs" else "case_ids"
counts = [strict["splits"][s][key] for s in split_names]
fig = go.Figure(go.Bar(
    x=[split_labels[s] for s in split_names], y=counts,
    marker_color=["#48A7D0", "#A28BDA", "#E7AF66"],
    text=counts, textposition="outside", hovertemplate="%{x}: %{y}<extra></extra>",
))
fig.update_yaxes(title=unit)
st.plotly_chart(style_chart(fig, 335), width="stretch")

st.markdown("#### TI-RADS composition of the strict ROI splits")
show_percentage = st.toggle("Show percentages instead of ROI counts", value=False)
fig = go.Figure()
for label, color in zip(LABEL_ORDER, COLORS):
    counts = [strict["splits"][name]["tirads_roi_distribution"][label] for name in split_names]
    values = [100 * n / strict["splits"][name]["rois"] for n, name in zip(counts, split_names)] if show_percentage else counts
    fig.add_trace(go.Bar(
        x=[split_labels[s] for s in split_names], y=values, name="TI-RADS " + label,
        marker_color=color, customdata=counts,
        hovertemplate="%{x} · " + "TI-RADS " + label + ": %{customdata} ROIs<extra></extra>",
    ))
fig.update_layout(barmode="stack")
fig.update_yaxes(title="Percentage of split" if show_percentage else "ROI samples", range=[0, 100] if show_percentage else None)
st.plotly_chart(style_chart(fig, 400), width="stretch")

summary = pd.DataFrame([
    {
        "Partition": split_labels[name],
        "Recorded case IDs": strict["splits"][name]["case_ids"],
        "ROIs": strict["splits"][name]["rois"],
        **{f"TI-RADS {label}": strict["splits"][name]["tirads_roi_distribution"][label] for label in LABEL_ORDER},
    }
    for name in split_names
])
st.dataframe(summary, width="stretch", hide_index=True)
st.warning("Zero overlap is verified for recorded case IDs—not for independently verified patient identities. The historical test partition has already been inspected; it is not a fresh independent test for later extensions.")
st.link_button("Inspect frozen split counts", source_url(sha, "07_Results/phase12_case_level_split/split_distribution.json"))

st.divider()
st.markdown("### C. Duplicate and integrity screening")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Distinct raw-file hashes", dupes["distinct_file_sha256"])
c2.metric("Exact duplicate groups", dupes["duplicate_groups"])
c3.metric("Redundant raw files", dupes["redundant_files"])
c4.metric("Cross-split exact groups", dupes["cross_assigned_split_duplicate_groups"])

fig = go.Figure(go.Bar(
    x=["Distinct raw-image hashes", "Redundant raw-file copies"],
    y=[dupes["distinct_file_sha256"], dupes["redundant_files"]],
    marker_color=["#48A7D0", "#E7AF66"],
    text=[dupes["distinct_file_sha256"], dupes["redundant_files"]], textposition="outside",
))
fig.update_yaxes(title="JPG file records")
st.plotly_chart(style_chart(fig, 315), width="stretch")

with st.expander("What exactly was checked?", expanded=True):
    st.markdown(
        "**Raw JPG audit:** SHA-256 byte-identical duplicate detection found six groups "
        "containing 15 files, including nine redundant copies. These are file-level findings. "
        "**ROI audit:** the separate annotation-geometry deduplication retained 440 unique ROIs. "
        "Neither audit proves patient independence. Near-duplicate screening was limited and did not "
        "establish a clinical identity match."
    )
    st.link_button("Inspect raw-file duplicate counts", source_url(sha, "07_Results/exp_duplicate_001/local_sha256/exact_raw_image_duplicates.json"))

st.divider()
st.markdown("### D. Integrity conclusions and source provenance")
st.markdown("**Verified findings:** original-image missing labels, strongly unequal category counts, zero overlap across recorded case-ID partitions, and measurable exact raw-file duplicates.")
st.markdown("**Limits:** dataset licensing and privacy clearance remain unresolved; label categories are TI-RADS grades, not pathology diagnoses; recorded case IDs do not verify unique patient identities.")
with st.expander("Source files used to build this dataset page"):
    st.code(sha, language=None)
    for relative in evidence["source_file_sha256"]:
        st.markdown(f"- [View `{relative}`]({source_url(sha, relative)})")

st.download_button(
    "Download aggregate dataset audit (JSON)",
    data=json.dumps({k: v for k, v in evidence.items() if k != "source_file_sha256"}, indent=2),
    file_name="qucnet_aggregate_dataset_audit.json", mime="application/json",
)
st.caption("QuCNet Research Explorer · Read-only archived evidence · No diagnosis or clinical validation claims")
