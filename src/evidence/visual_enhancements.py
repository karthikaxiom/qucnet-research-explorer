"""Optional, source-linked presentation of historical aggregate-only evidence.

This module never reads source medical images, checkpoints or research-repo files.
All diagrams added after initial publication are file-hash checked at display time.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ORIGINAL_REPO = "https://github.com/karthikaxiom/QCNN_Thyroid_Paper_Reproduction"

# All entries below point to aggregate visual evidence. No ROI preview / raw frames.
ADDITIONAL_CHARTS = (
    ("Historical evaluations", "fig_01_experiment_accuracy_comparison.png", "Recorded experimental accuracies; separate protocols and denominators", "09_Output_images/README/fig_01_experiment_accuracy_comparison.png"),
    ("Dataset & ROI audit", "fig_02_audited_dataset_split.png", "Recorded case-ID / ROI separation; not proof of distinct patients", "09_Output_images/README/fig_02_audited_dataset_split.png"),
    ("Phase 14 ROI model", "fig_03_phase14_prediction_collapse.png", "Single-class TI-RADS 3 prediction collapse, not binary QuCNet", "09_Output_images/README/fig_03_phase14_prediction_collapse.png"),
    ("Phase 14 ROI model", "phase14_roi_confusion_matrix.png", "Strict 71-ROI historical partition; 4/71 correct", "07_Results/phase14_roi_categorical_baseline/roi_confusion_matrix.png"),
    ("Phase 14 ROI model", "phase14_case_confusion_matrix.png", "Strict 44-case-ID historical partition; 2/44 correct", "07_Results/phase14_roi_categorical_baseline/case_confusion_matrix.png"),
    ("Phase 14 ROI model", "phase14_training_curves.png", "Archived training curves of strict classical ROI CNN", "07_Results/phase14_roi_categorical_baseline/training_curves.png"),
    ("Development extension", "fig_04_extension_fixed_maps_color.svg", "Fixed 64-D maps, 203 development case IDs", "09_Output_images/README/fig_04_extension_fixed_maps_color.svg"),
    ("Development extension", "fig_05_extension_preprocessing_color.svg", "Development-only preprocessing sensitivity", "09_Output_images/README/fig_05_extension_preprocessing_color.svg"),
    ("Development extension", "fig_06_extension_padding_logloss_color.svg", "Padding area vs exploratory log-loss; not a proven causal relationship", "09_Output_images/README/fig_06_extension_padding_logloss_color.svg"),
    ("Development extension", "fig_07_extension_fusion_ci_color.svg", "Exploratory image-versus-fusion paired comparisons", "09_Output_images/README/fig_07_extension_fusion_ci_color.svg"),
    ("Development extension", "fig_09_sixclass_ovr_roc_color.svg", "Six-class one-vs-rest ROC on development OOF cases, not binary diagnosis", "09_Output_images/README/fig_09_sixclass_ovr_roc_color.svg"),
    ("Development extension", "fig_10_classwise_sensitivity_color.svg", "Classwise recall highlights TI-RADS 3 failures", "09_Output_images/README/fig_10_classwise_sensitivity_color.svg"),
    ("Architecture distinctions", "fig_11_model_pipeline_comparison_color.svg", "Distinct Phase 9, Phase 14 and later extension model paths", "09_Output_images/README/fig_11_model_pipeline_comparison_color.svg"),
    ("Development extension", "fig_12_multiclass_calibration_color.svg", "Descriptive multiclass probability diagnostics; no clinical calibration claim", "09_Output_images/README/fig_12_multiclass_calibration_color.svg"),
)

# Explicit status: only a subset of full research experiments is visualized live.
EVIDENCE_REGISTER = (
    ("Phases 0–4", "Dataset, XML and preprocessing", "Aggregate audit", "Dataset & Integrity", "07_Results/exp_data_001/audit.json", "480 JPG / 390 XML; image/label-level facts; not patient validation"),
    ("Phases 5–7; 16", "Binary QuCNet and classical CNN", "Interactive + console + archived matrix", "Results Lab", "07_Results/phase16_binary_qucnet_evaluation/binary_evaluation_metrics.json", "54.29% quantum; 57.14% classical; n=70 historical validation-selected holdout"),
    ("Phases 8–10", "Historical six-class comparison", "Interactive + saved figures", "Results Lab", "07_Results/phase9_paper_tirads_class_weighted/test_metrics.json", "18.57% quantum, 24.29% classical; n=70 historical images"),
    ("Phases 11–13", "Recorded-case audit, split and ROI preparation", "Aggregate audit", "Dataset & Integrity", "07_Results/phase13_roi_dataset/roi_dataset_distribution.json", "440 ROI geometries / 291 recorded case IDs; patient uniqueness unknown"),
    ("Phase 14", "Strict classical ROI CNN", "Interpretation + saved figures", "Reliability Analysis", "07_Results/phase14_roi_categorical_baseline/evaluation_metrics.json", "4/71 ROIs and 2/44 case IDs; TI-RADS 3-only prediction"),
    ("Phases 15–18", "Consolidation, binary evaluation and errors", "Selected aggregates, original links", "Results / Evidence", "07_Results/phase17_paper_master_tables/master_results_table.png", "No new model trained; individual error galleries withheld"),
    ("EXT-DATA / DUPLICATE", "Exact duplicates, dataset integrity", "Aggregate audit", "Dataset & Integrity", "07_Results/exp_duplicate_001/local_sha256/exact_raw_image_duplicates.json", "6 exact-file groups, 15 JPG records, 9 redundant; not patient linkage"),
    ("EXT-MODELS", "Eight matched fixed-feature classifiers", "Interactive", "Results Lab", "07_Results/exp_models_001/metrics.json", "Fivefold case-ID grouped OOF; 203 development cases"),
    ("EXT-STABILITY", "Five-seed LR and RF stability", "Interactive", "Results Lab", "07_Results/exp_stability_001/metrics.json", "Same 203 cases repeated across fold assignments"),
    ("EXT-FUSION / PREPROC", "Fusion and preprocessing sensitivity", "Interactive", "Results / Reliability", "07_Results/exp_fusion_001/metrics.json", "Exploratory, development-only; not new independent test results"),
    ("EXT-BN", "BatchNorm running-statistic diagnostic", "Interactive", "Reliability Analysis", "07_Results/exp_bn_001/metrics.json", "Historical validation 2/58 → 11/58; trained weights unchanged"),
    ("EXT-CACHE / PADDING", "Quantum cache-key audit, padding investigation", "Source-linked figures / notes", "Evidence review", "07_Results/exp_cache_001/quantum_cache_key_audit.json", "Cache mismatches are not measured accuracy errors"),
    ("EXT-CALIB / EVAL", "Multiclass probability calibration / sensitivity audits", "Source-linked figures / notes", "Evidence review", "07_Results/exp_calib_001/calibration_metrics.json", "Exploratory probability quality; no clinical calibration validation"),
    ("External cohort gate", "Six-class cohort compatibility", "Source-only review", "Evidence review", "Research_Extension/EXTERNAL_VALIDATION_FEASIBILITY_GATE.md", "No independently compatible six-class validation cohort established"),
)


def verify_visual(root: Path, filename: str, *, manifest_name: str) -> tuple[bool, str]:
    """Use a verified per-file SHA256 or suppress the image completely."""
    figures = Path(root) / "assets" / "figures"
    img = figures / filename
    manifest = figures / manifest_name
    if not img.is_file() or not manifest.is_file():
        return False, "File/manifest not imported"
    try:
        record = json.loads(manifest.read_text(encoding="utf-8"))
        matches = [item for item in record["figures"] if item["file"] == filename]
        if len(matches) != 1:
            return False, "Missing or ambiguous provenance entry"
        if hashlib.sha256(img.read_bytes()).hexdigest() != matches[0]["sha256"]:
            return False, "Image SHA-256 mismatch"
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False, "Invalid manifest"
    return True, "Verified"


def pinned_link(commit: str, relative: str) -> str:
    if not (len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)):
        raise ValueError("Source commit must be a lower-case, 40-character SHA")
    if relative.startswith("/") or ".." in Path(relative).parts or "\\" in relative:
        raise ValueError("Invalid source evidence path")
    return f"{ORIGINAL_REPO}/blob/{commit}/{relative}"


def render_binary_provenance(st: Any, root: Path, commit: str) -> None:
    st.divider()
    st.subheader("Historical evaluation: original console and comparison")
    st.caption("Archived Phase 6 VS Code evaluation output, cropped for readability. The displayed metrics agree with the later saved binary summary after rounding; the screenshot does not establish a unique execution identifier.")
    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        ok, reason = verify_visual(root, "phase6_console_evidence.png", manifest_name="console_evidence_manifest.json")
        if ok:
            st.image(str(Path(root) / "assets" / "figures" / "phase6_console_evidence.png"), width="stretch")
            st.caption("Saved rounded console output: accuracy 0.5429; precision 0.5614; recall 0.8205; F1 0.6667. Screenshot source was cropped for readability.")
        else:
            st.warning(f"Console screenshot unavailable or failed integrity check: {reason}")
        st.link_button("Original unedited Phase 6 screenshot", pinned_link(commit, "09_Output_images/phase6_evaluation_result.png"))
    with col_right:
        st.markdown("**How to interpret the result**")
        st.write("Archived Phase 16 counts are **TN 6, FP 25, FN 7, TP 32**. Correct classifications = **38/70 = 54.29%**. The **56.14%** figure is positive-class precision, **not accuracy**.")
        st.write("In the recorded historical binary comparison, the **classical CNN achieved 57.14%** and the **quantum-feature QuCNet achieved 54.29%**. Both are prior research findings; this application displays their saved evidence without retraining either model.")
        st.write("The **55.71% (39/70)** majority-reference accuracy is a calculation from this holdout's class totals, not a separately trained model.")
        st.caption("The 70-image holdout informed checkpoint selection and is not an untouched test; proxy TI-RADS labels are not pathology-confirmed outcomes. Screenshot and later metric archive agree after rounding but exact historical run identity is not independently proven.")
        st.link_button("Historical binary audit note", pinned_link(commit, "07_Results/phase7_reproducibility_audit.txt"))


def render_phase14_figures(st: Any, root: Path, commit: str) -> None:
    st.markdown("#### Archived Phase 14 figures")
    st.caption("Separate strict *classical* six-class ROI model; 71 ROIs and 44 recorded case IDs. These images are not evidence of binary QuCNet performance.")
    wanted = [
        ("Prediction collapse", "fig_03_phase14_prediction_collapse.png"),
        ("ROI confusion matrix", "phase14_roi_confusion_matrix.png"),
        ("Case-ID confusion matrix", "phase14_case_confusion_matrix.png"),
    ]
    shown = 0
    for left in range(0, len(wanted), 2):
        cols = st.columns(2)
        for col, (label, filename) in zip(cols, wanted[left:left+2]):
            with col:
                ok, _ = verify_visual(root, filename, manifest_name="extra_figure_manifest.json")
                if ok:
                    st.markdown("**" + label + "**")
                    st.image(str(Path(root) / "assets" / "figures" / filename), width="stretch")
                    shown += 1
    if not shown:
        st.caption("Additional archived charts are not imported yet; the original results remain available from the linked research source.")
    st.link_button("Original Phase 14 evaluation figures", pinned_link(commit, "07_Results/phase14_roi_categorical_baseline/roi_confusion_matrix.png"))


def render_additional_gallery(st: Any, root: Path, commit: str) -> None:
    st.divider()
    st.subheader("Additional source-pinned research figures")
    st.caption("Source-verified aggregate figures. Each figure retains its original evaluation context; results from different protocols must not be interpreted as one common test set.")
    categories = list(dict.fromkeys(item[0] for item in ADDITIONAL_CHARTS))
    selected = st.selectbox("Explore research-figure category", categories, key="extra_visual_filter")
    choices = [item for item in ADDITIONAL_CHARTS if item[0] == selected]
    shown = 0
    for offset in range(0, len(choices), 2):
        cols = st.columns(2)
        for col, (_, filename, caveat, original) in zip(cols, choices[offset:offset+2]):
            with col:
                st.markdown("**" + filename.replace("_", " ").rsplit(".", 1)[0].capitalize() + "**")
                ok, why = verify_visual(root, filename, manifest_name="extra_figure_manifest.json")
                if ok:
                    st.image(str(Path(root) / "assets" / "figures" / filename), width="stretch")
                    shown += 1
                else:
                    st.caption("Not imported / provenance check failed: " + why)
                st.caption(caveat)
                st.link_button("Open frozen research source", pinned_link(commit, original), key="extra_visual_src_" + filename)
    if not shown:
        st.info("Source-only view: copy the screened, pinned charts from the local frozen checkout using the optional read-only import utility, then review the images before publishing.")


def render_evidence_register(st: Any, commit: str) -> None:
    st.divider()
    st.subheader("Research record and evidence coverage")
    st.caption("This register separates completed research activities from the evidence available for inspection in this application. Historical experiments are not rerun here.")
    rows = []
    for period, experiment, coverage, module, path, note in EVIDENCE_REGISTER:
        rows.append({"Research phase": period, "Experiment / audit": experiment, "App coverage": coverage,
                     "Shown in": module, "Protocol / scientific boundary": note, "Original evidence": pinned_link(commit, path)})
    st.dataframe(rows, width="stretch", hide_index=True, column_config={
        "Original evidence": st.column_config.LinkColumn("Original evidence", display_text="Open pinned file")
    })
    st.info("Source-linked records document prior work even when they do not have a separate interactive view. The archived research did not establish pathology-confirmed diagnosis or independent six-class external validation.")
