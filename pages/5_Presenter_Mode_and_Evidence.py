"""Source-linked evidence and interpretation of the recorded QuCNet research."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evidence.presenter_flow import STEPS, validate_evidence, speaking_script  # noqa: E402

st.set_page_config(page_title="Evidence & Interpretation | QuCNet", page_icon="🎓", layout="wide")
st.markdown("""
<style>
.block-container{max-width:1240px;padding-top:2rem;}
h1{letter-spacing:-.04em;}
[data-testid="stMetric"]{background:rgba(110,135,165,.07);border:1px solid rgba(135,150,185,.18);border-radius:14px;padding:14px;}
</style>
""", unsafe_allow_html=True)


def load_json(path: str) -> dict:
    try:
        return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        st.error(f"Required verified evidence is unavailable: {path} ({type(exc).__name__}).")
        st.stop()


hist = load_json("data/research_snapshot.json")
ext = load_json("data/extension_snapshot.json")
dataset = load_json("data/dataset_integrity_snapshot.json")
provenance = load_json("provenance.json")
try:
    validate_evidence(hist, ext, dataset, provenance)
except (KeyError, ValueError, TypeError) as exc:
    st.error(f"Source-evidence consistency check failed: {exc}. Unverified results cannot be displayed.")
    st.stop()

COMMIT = provenance["source_commit"]
REPO = provenance["research_repository"].rstrip("/")
if "presenter_position" not in st.session_state:
    st.session_state.presenter_position = 0
if "presenter_jump" not in st.session_state:
    st.session_state.presenter_jump = st.session_state.presenter_position


def jump(index: int) -> None:
    new_position = max(0, min(index, len(STEPS) - 1))
    st.session_state.presenter_position = new_position
    st.session_state.presenter_jump = new_position


from src.evidence.visual_enhancements import render_evidence_register

st.title("Evidence & Interpretation")
st.caption("A source-linked account of the research methods, recorded findings, limitations and software implementation")
st.info("Experimental findings are drawn from saved aggregate evidence. Synthetic circuit calculations and hypothetical metric scenarios are separate from historical model evaluations and are not medical predictions.")

with st.sidebar:
    st.markdown("**Research review**")
    st.caption(f"Verified source: `{COMMIT[:12]}`")

left, mid, right = st.columns([2.4, 1, 1])
with left:
    index = st.selectbox(
        "Select a research section",
        range(len(STEPS)),
        format_func=lambda k: f"{k+1:02d} · {STEPS[k].title}",
        index=st.session_state.presenter_position,
        key="presenter_jump",
        on_change=lambda: st.session_state.__setitem__("presenter_position", st.session_state.presenter_jump),
    )
with mid:
    st.metric("Section", f"{st.session_state.presenter_position+1:02d} / {len(STEPS):02d}")
with right:
    st.metric("Source integrity", "Verified", help="Core displayed metrics were checked against the local source snapshots at the recorded research commit.")

position = st.session_state.presenter_position
slide = STEPS[position]
st.progress((position + 1) / len(STEPS), text=f"Research record · {slide.section}")

back, next_col, reset_col = st.columns([1, 1, 1])
with back:
    st.button("← Previous section", on_click=jump, args=(position-1,), disabled=position == 0, width="stretch")
with next_col:
    st.button("Next section →", on_click=jump, args=(position+1,), disabled=position == len(STEPS)-1, type="primary", width="stretch")
with reset_col:
    st.button("Return to beginning", on_click=jump, args=(0,), width="stretch")

st.divider()
st.subheader(f"{position+1:02d}. {slide.title}")
st.caption(slide.section)

c1, c2 = st.columns([1.1, 1], gap="large")
with c1:
    st.markdown("#### What is shown")
    st.write(slide.shown)
    st.markdown("#### Interpretation")
    st.write(slide.interpretation)
with c2:
    st.markdown("#### Verified finding")
    st.write(slide.finding)
    st.markdown("#### Why it matters")
    st.write(slide.significance)

st.warning("Scientific boundary: the historical phases have different evaluation protocols. Source labels are not verified pathology diagnoses; this app cannot claim clinical utility or quantum superiority.")

st.divider()
show_notes = st.toggle("Researcher’s commentary", value=False)
if show_notes:
    st.markdown("#### Researcher’s commentary")
    st.write(slide.speaker_notes)

with st.expander("Methodological question and evidence", expanded=False):
    st.markdown("**Question:**")
    st.write(slide.question)
    st.markdown("**Interpretation:**")
    st.write(slide.answer)

with st.expander("Inspect exact original research evidence", expanded=False):
    st.caption(f"Frozen source commit: `{COMMIT}`")
    for path in slide.source_paths:
        st.markdown(f"- [{path}]({REPO}/blob/{COMMIT}/{path})")

if (ROOT / slide.module_path).is_file() and slide.module_path != "app.py":
    st.page_link(slide.module_path, label="Open the relevant interactive module", icon="🔎")

st.divider()
st.markdown("#### Methodological questions")
question_index = st.selectbox(
    "Select an evidence question",
    range(len(STEPS)),
    format_func=lambda i: f"{STEPS[i].section} · {STEPS[i].question}",
    key="rehearsal_question",
)
st.write(f"**Question:** {STEPS[question_index].question}")
if st.checkbox("Show interpretation", key="reveal_answer"):
    st.success(STEPS[question_index].answer)
    st.caption("Evidence: " + ", ".join(STEPS[question_index].source_paths))

st.download_button(
    "Export evidence summary (.txt)",
    data=speaking_script().encode("utf-8"),
    file_name="qucnet_evidence_summary.txt",
    mime="text/plain",
    help="Contains only aggregate research explanations and source paths; no medical records.",
)
st.caption("Prepared from frozen saved evidence. Not a clinical decision-support or model-training interface.")

render_evidence_register(st, COMMIT)
