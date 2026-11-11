"""Source-grounded ultrasound input pathways without medical-image redistribution.

Every diagram is a schematic of archived implementation steps, not an
ultrasound image, sample reconstruction, inference output or training result.
"""
from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any

SOURCE_COMMIT = "e848013b9ff943d816a8c90840255233a5ae7b8d"
SOURCE_REPO = "https://github.com/karthikaxiom/QCNN_Thyroid_Paper_Reproduction"
PAPER = "https://doi.org/10.1109/ACCESS.2024.3367806"


@dataclass(frozen=True)
class Step:
    title: str
    shape: str
    detail: str


@dataclass(frozen=True)
class InputRoute:
    heading: str
    context: str
    steps: tuple[Step, ...]
    source_paths: tuple[str, ...]
    result: str
    accent: str


ROUTES: dict[str, InputRoute] = {
    "Phase 6 · Binary quantum QuCNet": InputRoute(
        heading="Historical full-image binary pathway",
        context="Recorded 70-image validation-selected holdout; binary labels were TI-RADS-derived suspicion proxies, not pathology-confirmed outcomes.",
        steps=(
            Step("Grayscale ultrasound", "128 × 128 × 1", "Preprocessed full-image input"),
            Step("Patch-grid extraction", "2 × 2", "64 × 64 patch locations"),
            Step("Fixed four-qubit filter", "64 × 64 × 4", "RX / RY–CNOT / Pauli-Z"),
            Step("Dense classifier", "128 / 64 / 2", "Binary proxy Softmax"),
        ),
        source_paths=("05_Source_Code/phase5_quantum_filter.py", "05_Source_Code/phase6_qucnet_model.py", "05_Source_Code/utilities/quantum_utils.py"),
        result="54.29% (38/70) historical validation-selected holdout accuracy.",
        accent="#63B9CB",
    ),
    "Phase 9 · Six-class quantum CNN": InputRoute(
        heading="Historical six-class quantum-feature pathway",
        context="Separate six-category classification trained on saved four-channel quantum features; earlier image-level evaluation did not prove case independence.",
        steps=(
            Step("Full grayscale image", "128 × 128 × 1", "Historical source-image input"),
            Step("Fixed quantum features", "64 × 64 × 4", "Four-qubit patch map"),
            Step("Convolutional classifier", "Conv 16 / Conv 32", "Pooling; Dense 128 / 64"),
            Step("Six-class output", "Softmax × 6", "Six TI-RADS categories"),
        ),
        source_paths=("05_Source_Code/phase5_quantum_filter.py", "05_Source_Code/phase9_paper_tirads_model.py"),
        result="18.57% (13/70) on the historical six-class image-level comparison.",
        accent="#AF97E9",
    ),
    "Phase 10 · Classical grayscale CNN": InputRoute(
        heading="Historical matched classical pathway",
        context="Six-class grayscale CNN; uses a resized grayscale input instead of the quantum-transformed feature tensor.",
        steps=(
            Step("Full grayscale image", "128 × 128 × 1", "Historical normalized input"),
            Step("Spatial resize", "64 × 64 × 1", "No quantum feature channels"),
            Step("Convolutional classifier", "Conv 16 / Conv 32", "Pooling; Dense 128 / 64"),
            Step("Six-class output", "Softmax × 6", "Six TI-RADS categories"),
        ),
        source_paths=("05_Source_Code/phase10_classical_tirads_model.py",),
        result="24.29% (17/70) on the matched historical six-class comparison.",
        accent="#67BEA5",
    ),
    "Phase 14 · Strict ROI CNN": InputRoute(
        heading="Separate annotated-lesion ROI pathway",
        context="Crops from expert annotation geometry are square-padded and resized. Recorded case IDs were separated across splits; patient identity remains unverified.",
        steps=(
            Step("Annotated lesion region", "ROI geometry", "Source image + annotation"),
            Step("Square pad + resize", "64 × 64 × 1", "Grayscale lesion crop"),
            Step("Classical ROI CNN", "Conv / BN / GAP", "Augmentation only in training"),
            Step("Six-class output", "Softmax × 6", "Case-level mean of ROI scores"),
        ),
        source_paths=("05_Source_Code/phase13_prepare_roi_dataset.py", "05_Source_Code/phase14_train_roi_categorical_baseline.py"),
        result="5.63% (4/71 ROIs) and 4.55% (2/44 recorded cases); predicted TI-RADS 3 for every test ROI/case.",
        accent="#E0AF74",
    ),
    "Extension · Fixed 64-D representations": InputRoute(
        heading="Separate development-only feature comparison",
        context="Uses only 311 training ROIs / 203 recorded case IDs in fivefold case-grouped cross-validation. This is not the original end-to-end QuCNet CNN.",
        steps=(
            Step("Phase 13 training ROIs", "64 × 64 × 1", "Same development cases"),
            Step("Spatial mean pooling", "8 × 8", "Shared grayscale pooling"),
            Step("Two distinct fixed maps", "64-D each", "Classical vs simulated quantum"),
            Step("Matched classifiers", "LR / SVM / RF / 7-NN", "Fivefold case-grouped OOF"),
        ),
        source_paths=("05_Source_Code/exp_qmap_001_fixed_feature_compare.py", "07_Results/exp_models_001/metrics.json"),
        result="Development-fold accuracy varies by classifier; no proven quantum superiority.",
        accent="#E38CA5",
    ),
}


ARCHITECTURE_LABELS: dict[str, str] = {
    "Phase 6 · Binary quantum QuCNet": "Phase 6 · Binary quantum QuCNet",
    "Phase 9 · Six-class quantum QuCNet": "Phase 9 · Six-class quantum CNN",
    "Phase 10 · Six-class classical CNN": "Phase 10 · Classical grayscale CNN",
    "Phase 14 · Strict ROI CNN": "Phase 14 · Strict ROI CNN",
    "Extension · Matched fixed 64-D features": "Extension · Fixed 64-D representations",
}


def diagram_svg(route_key: str) -> str:
    """Return an accessible static vector diagram; no externally loaded images."""
    route = ROUTES[route_key]
    accent = route.accent
    width, height = 1100, 208
    blocks = []
    for i, stage in enumerate(route.steps):
        x = 19 + i * 273
        blocks.append(
            f'<rect x="{x}" y="24" width="246" height="158" rx="16" fill="#19273A" stroke="#35506B" stroke-width="1.2"/>'
            f'<rect x="{x}" y="24" width="246" height="5" rx="2" fill="{accent}"/>'
            f'<text x="{x+18}" y="57" font-size="13" font-weight="700" fill="{accent}">STAGE {i+1:02d}</text>'
            f'<text x="{x+18}" y="87" font-size="18" font-weight="700" fill="#F3F7FC">{escape(stage.title)}</text>'
            f'<text x="{x+18}" y="118" font-size="17" font-weight="600" fill="#E2EDF6">{escape(stage.shape)}</text>'
            f'<text x="{x+18}" y="149" font-size="13" fill="#AAC0D3">{escape(stage.detail)}</text>'
        )
        if i < 3:
            arrow_x = x + 253
            blocks.append(f'<path d="M {arrow_x} 101 H {arrow_x+12} M {arrow_x+7} 96 L {arrow_x+12} 101 L {arrow_x+7} 106" fill="none" stroke="{accent}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>')
    title = escape(route.heading)
    descr = escape("Implementation-stage tensor diagram, not a source medical image or prediction. " + route.context)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="pipelineTitle pipelineDesc">'
        f'<title id="pipelineTitle">{title}</title><desc id="pipelineDesc">{descr}</desc>'
        '<rect x="0" y="0" width="1100" height="208" rx="20" fill="#101B2B"/>'
        + "".join(blocks) + '</svg>'
    )


def original_link(commit: str, path: str) -> str:
    if commit != SOURCE_COMMIT:
        raise ValueError("Research source commit disagrees with archived evidence")
    if path.startswith("/") or ".." in path.split("/") or "\\" in path:
        raise ValueError("Unsafe research source path")
    return f"{SOURCE_REPO}/blob/{commit}/{path}"


def render_route(st: Any, route_key: str, commit: str, *, show_links: bool = True) -> None:
    if commit != SOURCE_COMMIT:
        st.error("The research source commit does not match the verified evidence.")
        st.stop()
    route = ROUTES[route_key]
    st.caption(route.context)
    st.image(
        diagram_svg(route_key), width="stretch",
        alt=f"Schematic, not ultrasound: {route.heading}. " + "; ".join(f"{s.title}: {s.shape}" for s in route.steps),
    )
    st.caption("Implementation diagram only. It does not depict a real thyroid ultrasound image, patient ROI or model prediction.")
    st.markdown(f"**Recorded evaluation:** {route.result}")
    if show_links:
        st.caption("Source code and archived-evidence references")
        st.markdown(" · ".join(
            f"[{path.rsplit('/', 1)[-1]}]({original_link(commit, path)})"
            for path in route.source_paths
        ))


def render_dataset_input_section(st: Any, commit: str) -> None:
    """Input pipeline explorer for the Dataset & Integrity page."""
    st.subheader("Ultrasound image inputs and processing pathways")
    st.write(
        "The archived experiments used thyroid ultrasound JPG records and, for the later ROI studies, "
        "annotation-derived lesion crops. The views below show *implementation shapes and operations*, "
        "not images from the clinical dataset. Different experiments use separate input pathways."
    )
    choice = st.selectbox("Inspect a recorded processing pathway", list(ROUTES), key="input_processing_route")
    render_route(st, choice, commit)
    st.markdown("**Where can the original ultrasound examples be seen?**")
    st.write(
        "Figures 1 and 2 of the published QuCNet paper illustrate thyroid ultrasound examples. "
        "The app links to that publication for visual context but does not redistribute paper figures, "
        "source DDTI JPG/XML records or individual lesion crops."
    )
    st.link_button("View published QuCNet paper · Figures 1 and 2", PAPER, icon="🔗")
    st.caption(
        "No raw ultrasound images, patient identifiers, case-level predictions or checkpoints are included. "
        "Image-data redistribution permissions and privacy clearance are not established."
    )
