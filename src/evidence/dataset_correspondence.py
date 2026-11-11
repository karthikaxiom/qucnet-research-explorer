"""Original-paper dataset correspondence, using aggregate source-verified facts only.

This module describes published claims and archived research evidence. It neither
reconstructs unavailable labels nor asserts causal explanations for accuracy gaps.
"""
from __future__ import annotations

SOURCE_COMMIT = "e848013b9ff943d816a8c90840255233a5ae7b8d"
PAPER_DOI = "https://doi.org/10.1109/ACCESS.2024.3367806"
SOURCE_REPO = "https://github.com/karthikaxiom/QCNN_Thyroid_Paper_Reproduction"

# G. Swathi et al., IEEE Access 12 (2024), pp. 27831–27832 and Tables 1–2.
PAPER = {
    "images": 478,
    "medical_cases_described": 399,
    "images_with_benign_malignant_groups": 460,
    "images_described_with_tirads": 478,
    "binary_training": 333,
    "binary_test": 127,
    "six_class_training": 380,
    "six_class_test": 98,
    "binary_accuracy": 0.9763,
    "six_class_accuracy": 0.9387,
}


def verified_comparison(dataset: dict, research: dict, source_commit: str) -> dict:
    """Return display-only comparisons after checking archived aggregate consistency."""
    if source_commit != SOURCE_COMMIT or dataset.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("Dataset comparison source commit mismatch")
    inventory = dataset["inventory"]
    binary = research["binary_qucnet"]
    six = research["six_class_historical"]
    jpg = inventory["raw_jpg_images"]
    labelled = inventory["images_with_tirads"]
    missing = inventory["images_missing_tirads"]
    xml = inventory["xml_records"]
    if (jpg, xml, labelled, missing) != (480, 390, 349, 131) or labelled + missing != jpg:
        raise ValueError("Recovered dataset inventory differs from archived verification")
    if binary["n_images"] != 70 or abs(binary["accuracy"] - 38 / 70) > 1e-10:
        raise ValueError("Historical binary evaluation is inconsistent")
    if six["n_images"] != 70 or abs(six["quantum_accuracy"] - 13 / 70) > 1e-10:
        raise ValueError("Historical six-class evaluation is inconsistent")
    assert PAPER["binary_training"] + PAPER["binary_test"] == PAPER["images_with_benign_malignant_groups"]
    assert PAPER["six_class_training"] + PAPER["six_class_test"] == PAPER["images_described_with_tirads"]
    rows = [
        {"Comparison": "Reported archive inventory", "Published QuCNet study": "478 ultrasound images", "Recovered research archive": "480 JPG files"},
        {"Comparison": "Case and annotation records", "Published QuCNet study": "399 medical cases described", "Recovered research archive": "390 XML records (different unit)"},
        {"Comparison": "Binary outcome availability", "Published QuCNet study": "460 images grouped benign/malignant", "Recovered research archive": "No verified pathology mapping; TI-RADS-derived proxy"},
        {"Comparison": "Six-class TI-RADS label coverage", "Published QuCNet study": "478 images described as classified", "Recovered research archive": "349 labeled JPG; 131 lacking TI-RADS"},
        {"Comparison": "Binary evaluation", "Published QuCNet study": "127 test images; 97.63% reported accuracy", "Recovered research archive": "70-image validation-selected holdout; 54.29%"},
        {"Comparison": "Six-class evaluation", "Published QuCNet study": "98 test images; 93.87% reported accuracy", "Recovered research archive": "Different 70-image historical evaluation; 18.57%"},
    ]
    return {"paper": PAPER, "recovered": inventory, "rows": rows}


def render_dataset_correspondence(st, dataset: dict, research: dict, source_commit: str) -> None:
    """Render a side-by-side evidence table without comparing unmatched accuracies in one plot."""
    evidence = verified_comparison(dataset, research, source_commit)
    inv = evidence["recovered"]
    st.subheader("Dataset correspondence and label availability")
    st.caption(
        "Both studies refer to DDTI ultrasound data. A shared dataset name does not "
        "establish an identical archive version, case roster, label provenance or test set."
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**Published QuCNet study · reported dataset**")
        st.metric("DDTI image records", f"{PAPER['images']:,}")
        st.caption("399 medical cases described; 460 images assigned benign/malignant groups.")
    with right:
        st.markdown("**Independent reconstruction · recovered archive**")
        st.metric("Recovered JPG files", f"{inv['raw_jpg_images']:,}")
        st.caption(f"{inv['xml_records']} XML records; {inv['images_with_tirads']} images with recoverable TI-RADS labels.")
    st.dataframe(evidence["rows"], hide_index=True, width="stretch")
    st.markdown("**Why this matters for interpreting the results**")
    st.write(
        "The published binary task describes benign/malignant groups, whereas the "
        "reconstructed binary task uses TI-RADS-derived suspicion proxies because an "
        "equivalent pathology-confirmed label mapping was not independently recoverable. "
        "The original paper describes separate held-out tests of 127 and 98 images; "
        "the historical binary reconstruction evaluated a 70-image set also used in "
        "validation/checkpoint selection. These are not the same classification targets "
        "or evaluation cohorts."
    )
    st.info(
        "Both sources concern thyroid ultrasound images—not different 'pathology images'. "
        "The missing element is verifiable pathology-linked outcome labeling, rather than "
        "a different image modality. The paper's 399 reported cases and 390 recovered XML "
        "records cannot be subtracted to claim a missing-case count."
    )
    st.warning(
        "Different label provenance, annotation completeness, sample selection, "
        "preprocessing and checkpoint protocols are evidence-backed comparability limits. "
        "Their contribution to the measured accuracy gap has not been causally established. "
        "Neither 54.29% nor 18.57% is a direct numerical replication of the paper's test."
    )
    st.markdown(
        f"[Original IEEE Access article]({PAPER_DOI}) · "
        f"[Archived research inventory and protocol notes]({SOURCE_REPO}/blob/{source_commit}/README.md) · "
        f"[Original-paper method-fidelity audit]({SOURCE_REPO}/blob/{source_commit}/Research_Extension/ORIGINAL_PAPER_FULLTEXT_FIDELITY_AUDIT.md)"
    )
