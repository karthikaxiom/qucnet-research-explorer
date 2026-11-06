"""Read-only, source-backed presentation narrative for QuCNet Research Explorer.

Only approved aggregate facts and exact research-source paths are represented.
No patient-specific samples, model inference, training or network access.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Mapping, Any


@dataclass(frozen=True)
class PanelStep:
    title: str
    section: str
    shown: str
    interpretation: str
    finding: str
    significance: str
    speaker_notes: str
    question: str
    answer: str
    source_paths: tuple[str, ...]
    module_path: str


# Only relative, already inspected, file-backed source paths.
STEPS: tuple[PanelStep, ...] = (
    PanelStep(
        "Why this research?", "Research question",
        "A published QuCNet approach, followed by an independent local reconstruction and audit.",
        "The original authors reported 97.63% binary accuracy on 127 images and 93.87% six-class accuracy on 98 images. Those are the authors' reported results, not our measurements.",
        "The research investigated whether the publicly recoverable data and described method supported comparable results.",
        "An independent reconstruction is useful even when it does not recover the published headline scores.",
        "Start with the research question. State the two published percentages and their denominators, then immediately explain that our labels, samples and protocol are not identical. Do not imply we proved the authors wrong.",
        "Did your experiment disprove the original paper?",
        "No. The original and recovered tasks, inventories, labels and evaluation protocols are not directly equivalent. We document limitations of our reconstruction, not a numerical refutation.",
        ("Research_Extension/ORIGINAL_PAPER_FULLTEXT_FIDELITY_AUDIT.md", "README.md"),
        "app.py",
    ),
    PanelStep(
        "Recovering the dataset", "Dataset integrity",
        "Aggregate source composition and TI-RADS label coverage.",
        "480 JPG images and 390 XML records were recovered; 349 image records had a nonempty TI-RADS label and 131 did not.",
        "Data completeness and label provenance had to be audited before interpreting model scores.",
        "The binary endpoint was derived from TI-RADS suspicion groups, not pathology-confirmed malignancy.",
        "Explain image count versus XML/case metadata count. Point out the missing labels and do not call the historical proxy a confirmed cancer diagnosis.",
        "Do the binary labels prove benign or malignant pathology?",
        "No. They are low-/high-suspicion proxies reconstructed from TI-RADS annotations; pathology confirmation is not established.",
        ("07_Results/exp_data_001/audit.json", "07_Results/phase7_reproducibility_audit.txt"),
        "pages/2_Dataset_and_Integrity.py",
    ),
    PanelStep(
        "How QuCNet features are formed", "Architecture",
        "Phase 6: 128×128 grayscale → 2×2 patches → fixed four-qubit simulated circuit → 64×64×4 channels → dense binary classifier.",
        "PennyLane default.qubit uses four RX encodings, two fixed seeded RY/CNOT-ring layers and four Pauli-Z outputs per patch.",
        "The circuit is real simulated quantum feature mapping, but its parameters are fixed and it does not run on quantum hardware.",
        "A synthetic patch demo exposes the actual mathematical transformation without medical-image uploads or retraining.",
        "Move through one or two architecture stages. Then change one synthetic pixel and show how the expectation values update. Call these circuit outputs, not classification predictions.",
        "If it is quantum, why don't you show a live image diagnosis?",
        "The mini-project's purpose is architecture and evidence auditing. The archived model is not clinically validated; we demonstrate the fixed circuit on synthetic input instead of presenting a diagnosis service.",
        ("05_Source_Code/utilities/quantum_utils.py", "05_Source_Code/phase6_qucnet_model.py", "05_Source_Code/phase5_quantum_filter.py"),
        "pages/1_Interactive_Architecture_Lab.py",
    ),
    PanelStep(
        "What happened in the binary experiment?", "Historical results",
        "Phase 16 binary QuCNet confusion matrix and linked metric cards.",
        "On a validation-selected historical 70-image holdout, accuracy was 38/70 = 54.29%, while precision was 32/57 = 56.14%; recall 82.05%, specificity 19.35%.",
        "The model detected 32/39 high-suspicion proxy cases but produced 25 false positives among 31 low-suspicion proxy cases.",
        "Accuracy and sensitivity alone can conceal severe class-specific errors; the historical holdout was used for checkpoint selection.",
        "Point to TN=6, FP=25, FN=7, TP=32. Emphasize that 56.14% is PRECISION, not accuracy. Explain that this is not an untouched clinical test.",
        "Is your accuracy 56.14% or 54.29%?",
        "54.29% is the recorded accuracy (38/70). 56.14% is positive-class precision (32/57). They measure different quantities.",
        ("07_Results/phase16_binary_qucnet_evaluation/binary_evaluation_metrics.json", "05_Source_Code/phase6_train.py"),
        "pages/3_Experimental_Results_Lab.py",
    ),
    PanelStep(
        "Six-class comparison", "Historical results",
        "Phase 9 paper-structured quantum-feature CNN versus Phase 10 classical grayscale CNN on the historical 70-image six-class comparison.",
        "Quantum accuracy 13/70 = 18.57%; classical accuracy 17/70 = 24.29%; majority reference 19/70 = 27.14%.",
        "The quantum reconstruction did not outperform the matched classical comparator in this historical comparison.",
        "The comparison motivates accuracy-gap analysis rather than a claim of quantum superiority.",
        "Open the matched six-class bars and compare macro-F1 or balanced accuracy. Do not juxtapose these as if they shared the Phase 14 ROI test or the authors' original test set.",
        "Why is your six-class score so much lower than the paper's?",
        "The exact original samples, target provenance, normalization and some implementation details were not recoverable. Different methods and evaluations prevent a direct causal explanation of the difference.",
        ("07_Results/phase9_paper_tirads_class_weighted/test_metrics.json", "07_Results/phase10_classical_tirads_class_weighted/test_metrics.json"),
        "pages/3_Experimental_Results_Lab.py",
    ),
    PanelStep(
        "Stricter dataset splitting", "Method integrity",
        "Recorded-case-ID-separated ROI dataset: 440 distinct ROI geometries from 291 recorded case IDs.",
        "There are 311 train ROIs/203 case IDs, 58 validation ROIs/44 case IDs, and 71 historical test ROIs/44 case IDs; recorded case overlap is zero.",
        "The stricter partition addresses a limitation in earlier image-level splitting, which shared some recorded case IDs across subsets.",
        "Case-ID separation improves the audit but does NOT prove distinct patients across identifiers.",
        "Distinguish images, ROI crops, case IDs and patients. Explicitly state we cannot certify patient independence.",
        "Does zero case-ID overlap mean your test is patient-independent?",
        "No. We verified that recorded case IDs do not overlap. Different IDs were not independently verified as different patients.",
        ("07_Results/phase12_case_level_split/split_distribution.json", "07_Results/phase13_roi_dataset/roi_dataset_distribution.json", "07_Results/phase11_case_annotation_audit/previous_split_leakage_report.txt"),
        "pages/2_Dataset_and_Integrity.py",
    ),
    PanelStep(
        "An important negative result", "Reliability",
        "Phase 14 strict classical ROI CNN predicted TI-RADS 3 for every 71 test ROIs and every 44 recorded test cases.",
        "Only 4/71 ROI predictions (5.63%) and 2/44 case predictions (4.55%) were correct.",
        "This is complete single-class prediction collapse in the Phase 14 classical ROI model, not a quantum-model accuracy.",
        "A classifier can appear to run correctly while failing to distinguish categories.",
        "This is a strong audit result. Don't excuse it or attribute a proven single cause. Show that the code ran, but its predictions were not useful as a category discriminator.",
        "Did BatchNorm or padding definitely cause the collapse?",
        "No. The diagnostic changed outputs, but the studies do not isolate a unique causal mechanism for the original collapse.",
        ("07_Results/phase14_roi_categorical_baseline/evaluation_metrics.json", "07_Results/exp_bn_001/metrics.json"),
        "pages/4_Reliability_and_Accuracy_Gap.py",
    ),
    PanelStep(
        "Matched classifier families", "Development-only controls",
        "Eight fixed-feature/classifier configurations: classical 64-D versus simulated quantum 64-D features under LR, RBF-SVM, random forest and 7-NN.",
        "The seed-42 fixed quantum random forest scored 63/203 = 31.03%, versus 54/203 = 26.60% for classical random forest; development majority was 57/203 = 28.08%.",
        "The quantum RF had the highest observed accuracy at that seed but still missed all 10 TI-RADS 3 cases.",
        "This is exploratory fivefold case-grouped out-of-fold analysis on training cases—not the original QuCNet CNN or an external clinical test.",
        "When asked whether quantum won, say one seed and classifier had a higher observed score, but classwise failure and statistical uncertainty prevent superiority claims.",
        "Did the fixed quantum random forest prove quantum advantage?",
        "No. The fixed circuit was a simulated feature map; the advantage at one fold assignment was exploratory and did not establish robust statistical or computational superiority.",
        ("07_Results/exp_models_001/metrics.json", "07_Results/exp_qmap_001/metrics.json"),
        "pages/3_Experimental_Results_Lab.py",
    ),
    PanelStep(
        "What changed under diagnostic checks?", "Reliability",
        "Repeated case-fold stability, BatchNorm running-statistics recalibration, preprocessing choices and feature fusion.",
        "Across five fold seeds, RF mean accuracies were 27.49% classical and 29.36% fixed quantum on the same 203 cases. BatchNorm validation accuracy changed from 2/58=3.45% to 11/58=18.97%, below 17/58=29.31% majority.",
        "Classifier ranking was fold-sensitive and minority-class failures persisted; recalibration altered predictions but did not rescue reliable performance.",
        "These studies improve understanding, not clinical classification accuracy; five seeds are correlated repeats of the same cases.",
        "Show the stability chart or BatchNorm tab. Mention fusion adds one correct case (46 vs 45/203) with McNemar p=1.00; a measured change isn't always meaningful.",
        "If preprocessing changes the score, why not choose the highest?",
        "These are exploratory development comparisons with trade-offs and uncertainty, not an independently validated hyperparameter search. For example direct resizing distorts lesion aspect ratio.",
        ("07_Results/exp_stability_001/metrics.json", "07_Results/exp_bn_001/metrics.json", "07_Results/exp_fusion_001/metrics.json", "07_Results/exp_preproc_001/metrics.json"),
        "pages/4_Reliability_and_Accuracy_Gap.py",
    ),
    PanelStep(
        "What the mini-project contributes", "Software contribution",
        "QuCNet Research Explorer: source-backed architecture diagrams, synthetic circuit calculator, dataset integrity analysis, archived result exploration and guided evidence walkthrough.",
        "The app runs on saved aggregate evidence without TensorFlow, PennyLane, raw medical files, checkpoints or model training at runtime.",
        "The contribution is an interactive reproducibility and evaluation-audit workbench, not a new AI classifier or medical diagnostic system.",
        "It makes a complex, imperfect research record understandable, traceable and reviewable in a live demonstration.",
        "Conclude with software architecture, source provenance, test suite, separate repo and limitations. Offer to open one exact research source for a panel member.",
        "What is new here if your models were trained earlier?",
        "The new work is the verified evidence-integration pipeline, interactive architecture/circuit education, comparison and reliability modules, source traceability, tests and guided presenter workflow. The app itself does not retrain or improve QuCNet.",
        ("Research_Extension/FINAL_RESEARCH_VERIFICATION_MATRIX.md", "Research_Extension/MASTER_VERIFIED_RESULTS.csv"),
        "app.py",
    ),
)


def validate_evidence(hist: Mapping[str, Any], ext: Mapping[str, Any], dataset: Mapping[str, Any], provenance: Mapping[str, Any]) -> None:
    """Refuse to present when the local snapshots contradict core archived figures."""
    sha = provenance.get("source_commit")
    if not isinstance(sha, str) or len(sha) != 40 or any(x not in '0123456789abcdef' for x in sha):
        raise ValueError("Invalid source commit")
    if ext.get("source_commit") != sha or dataset.get("source_commit") != sha:
        raise ValueError("Evidence snapshots do not share the same research commit")
    b = hist["binary_qucnet"]
    cm = b["confusion_matrix"]
    counts = (cm["true_negative"], cm["false_positive"], cm["false_negative"], cm["true_positive"])
    if counts != (6, 25, 7, 32):
        raise ValueError("Binary confusion matrix differs from source")
    if b["n_images"] != sum(counts) or not isclose(b["accuracy"], 38/70, abs_tol=1e-10):
        raise ValueError("Historical binary metric inconsistency")
    if not isclose(b["precision"], 32/57, abs_tol=1e-10):
        raise ValueError("Historical precision inconsistency")
    s = hist["six_class_historical"]
    if s["n_images"] != 70 or not isclose(s["quantum_accuracy"], 13/70, abs_tol=1e-10):
        raise ValueError("Six-class quantum metric inconsistency")
    if not isclose(s["classical_accuracy"], 17/70, abs_tol=1e-10):
        raise ValueError("Six-class classical metric inconsistency")
    roi = hist["strict_roi_cnn"]
    if roi["n_rois"] != 71 or roi["n_case_ids"] != 44:
        raise ValueError("ROI denominator inconsistency")
    if not isclose(roi["roi_accuracy"], 4/71, abs_tol=1e-10):
        raise ValueError("ROI accuracy inconsistency")
    if not isclose(roi["case_accuracy"], 2/44, abs_tol=1e-10):
        raise ValueError("ROI case accuracy inconsistency")
    ds = dataset["inventory"]
    if (ds["raw_jpg_images"], ds["xml_records"], ds["images_with_tirads"], ds["images_missing_tirads"]) != (480, 390, 349, 131):
        raise ValueError("Dataset inventory inconsistency")
    strict = dataset["strict_roi"]
    if (strict["roi_geometries"], strict["recorded_case_ids"], strict["case_id_overlap_between_splits"]) != (440, 291, 0):
        raise ValueError("Case split inconsistency")
    if strict["patient_identity_independence_verified"] is not False:
        raise ValueError("Patient-independence caveat must be retained")
    models = ext["classifier_comparison"]
    if models["case_count"] != 203 or models["roi_count"] != 311:
        raise ValueError("Development denominator inconsistency")
    model = models["models"]["fixed_quantum_64__random_forest"]
    if not isclose(model["accuracy"], 63/203, abs_tol=1e-10):
        raise ValueError("Classifier metric inconsistency")
    if ext["stability"]["seeds"] != [42,101,202,303,404]:
        raise ValueError("Stability-seed inconsistency")
    if not isclose(ext["stability"]["models"]["fixed_quantum_64__random_forest"]["accuracy"]["mean"], 0.2935960591133005, abs_tol=1e-10):
        raise ValueError("Stability mean inconsistency")
    if not isclose(ext["batchnorm"]["conditions"]["recalibrated"]["accuracy"], 11/58, abs_tol=1e-10):
        raise ValueError("BatchNorm numerator/denominator inconsistency")


def speaking_script() -> str:
    """A printable summary based only on the reviewed aggregate evidence."""
    lines = [
        "QUCNET RESEARCH EXPLORER — PANEL DEMONSTRATION NOTES",
        "Independent architecture, reproducibility and evaluation-audit workbench",
        "All outcomes are historical saved evidence, except explicitly synthetic demonstrations.",
        "",
    ]
    for index, slide in enumerate(STEPS, 1):
        lines += [
            f"{index:02d}. {slide.title.upper()} — {slide.section}",
            "ON SCREEN: " + slide.shown,
            "INTERPRETATION: " + slide.interpretation,
            "FINDING: " + slide.finding,
            "WHY IT MATTERS: " + slide.significance,
            "PRESENTATION NOTES: " + slide.speaker_notes,
            "EXAMINER QUESTION: " + slide.question,
            "RESPONSE: " + slide.answer,
            "SOURCE FILES: " + "; ".join(slide.source_paths),
            "",
        ]
    lines += [
        "LIMITATIONS: No pathology-confirmed clinical evaluation; no verified patient identity independence; no independent six-class external cohort; no quantum superiority claim.",
        "The frozen research source is read-only. Mini-project software is a separate artifact.",
    ]
    return "\n".join(lines) + "\n"
