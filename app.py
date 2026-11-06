"""QuCNet Research Explorer | unified, evidence-linked multipage entry point.

The original research repository is never imported or executed by this app.
Only saved, privacy-safe aggregate snapshots are loaded at runtime.
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="QuCNet Research Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Consistent design tokens and readable panel-projection sizing.
st.markdown(
    """<style>
    .block-container {max-width: 1240px; padding-top: 1.8rem; padding-bottom: 2.5rem;}
    h1 {letter-spacing: -.04em; font-weight: 750;}
    h2, h3 {letter-spacing: -.023em;}
    [data-testid="stMetric"] {
        background: rgba(115,140,174,.075);
        border: 1px solid rgba(140,159,192,.19);
        border-radius: 14px;
        padding: 15px;
    }
    a:focus-visible, button:focus-visible {
        outline: 2px solid #38BDF8;
        outline-offset: 2px;
    }
    @media (max-width: 768px) {
       .block-container {padding-top: 1rem; padding-left: 1rem; padding-right: 1rem;}
    }
    </style>""",
    unsafe_allow_html=True,
)


def evidence_status() -> tuple[str, bool]:
    """Check only safe local manifests, never source medical data or checkpoints."""
    needed = (
        "provenance.json",
        "data/research_snapshot.json",
        "data/extension_snapshot.json",
        "data/dataset_integrity_snapshot.json",
    )
    try:
        documents = {
            name: json.loads((ROOT / name).read_text(encoding="utf-8-sig"))
            for name in needed
        }
        commit = documents["provenance.json"]["source_commit"]
        if not (isinstance(commit, str) and len(commit) == 40):
            raise ValueError("Invalid source commit")
        for item in ("data/extension_snapshot.json", "data/dataset_integrity_snapshot.json"):
            if documents[item].get("source_commit") != commit:
                raise ValueError("Source commit mismatch")
        return commit, True
    except (OSError, ValueError, TypeError, KeyError):
        return "unavailable", False


# Streamlit's Page + navigation API gives every module an intentional label
# without renaming the canonical `streamlit run app.py` entry point.
OVERVIEW = st.Page(
    "pages/0_Research_Overview.py",
    title="Research Overview",
    icon=":material/home:",
    default=True,
)
ARCHITECTURE = st.Page(
    "pages/1_Interactive_Architecture_Lab.py",
    title="Interactive Architecture Lab",
    icon=":material/memory:",
    url_path="Interactive_Architecture_Lab",
)
DATASET = st.Page(
    "pages/2_Dataset_and_Integrity.py",
    title="Dataset & Integrity Explorer",
    icon=":material/database:",
    url_path="Dataset_and_Integrity",
)
RESULTS = st.Page(
    "pages/3_Experimental_Results_Lab.py",
    title="Experimental Results Lab",
    icon=":material/analytics:",
    url_path="Experimental_Results_Lab",
)
RELIABILITY = st.Page(
    "pages/4_Reliability_and_Accuracy_Gap.py",
    title="Reliability & Accuracy-Gap Analysis",
    icon=":material/shield:",
    url_path="Reliability_and_Accuracy_Gap",
)
PRESENTER = st.Page(
    "pages/5_Presenter_Mode_and_Evidence.py",
    title="Evidence & Interpretation",
    icon=":material/co_present:",
    url_path="Evidence_and_Interpretation",
)

page = st.navigation(
    {
        "RESEARCH": [OVERVIEW, ARCHITECTURE, DATASET],
        "EXPERIMENTS": [RESULTS, RELIABILITY],
        "EVIDENCE REVIEW": [PRESENTER],
    },
    position="sidebar",
)

commit, verified = evidence_status()
with st.sidebar:
    st.divider()
    st.markdown("**QuCNet Research Explorer**")
    st.caption("Interactive architecture · reproducibility · evaluation audit")
    if verified:
        st.caption(f"Research source commit: `{commit[:12]}`")
        st.caption("Aggregate evidence loaded · no medical-image inference")
    else:
        st.error("An evidence snapshot is missing or inconsistent.")

if not verified:
    st.error(
        "The verified research snapshot set is incomplete or has mismatched "
        "source commits. Restore it before using the research workbench."
    )
    st.stop()

page.run()
