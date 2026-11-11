"""Read-only, source-backed presentation narrative for QuCNet Research Explorer.

Only approved aggregate facts and exact research-source paths are represented.
No patient-specific samples, model inference, training or network access.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Mapping, Any


@dataclass(frozen=True)
class ResearchStage:
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
STEPS: tuple[ResearchStage, ...] = (
    ResearchStage(
        "Research objective", "Research question",
        "A published QuCNet approach, followed by an independent local reconstruction and audit.",
        "The original authors reported 97.63% binary accuracy on 127 images and 93.87% six-class accuracy on 98 images. These are results reported by the authors, not measurements from this reconstruction.",
        "The research investigated whether the publicly recoverable data and described method supported comparable results.",
        "An independent reconstruction is useful even when it does not recover the published headline scores.",
        "I examined whether the published method could be reconstructed using the recoverable data and stated implementation details. The original paper's sample sets and labels were not available as an identical evaluation protocol.",
        "Do these results contradict the original paper's reported accuracy?",
        "No. The tasks, available records, labels and evaluation protocols differ. This study documents the reconstruction and its limitations; it is not a direct numerical refutation of the publication.",
        ("Research_Extension/ORIGINAL_PAPER_FULLTEXT_FIDELITY_AUDIT.md", "README.md"),
        "app.py",
    ),
    ResearchStage(
        "Recovering the dataset", "Dataset integrity",
        "Aggregate source composition and TI-RADS label coverage.",
        "480 JPG images and 390 XML records were recovered; 349 image records had a nonempty TI-RADS label and 131 did not.",
        "Data completeness and label provenance had to be audited before interpreting model scores.",
        "The binary endpoint was derived from TI-RADS suspicion groups, not pathology-confirmed malignancy.",
        "I first checked the recoverable images, XML records and TI-RADS labels. The missing-label count matters because the original binary outcomes cannot be interpreted as pathology-confirmed cancer diagnoses.",
        "Do the binary labels prove benign or malignant pathology?",
        "No. They are low-/high-suspicion proxies reconstructed from TI-RADS annotations; pathology confirmation is not established.",
        ("07_Results/exp_data_001/audit.json", "07_Results/phase7_reproducibility_audit.txt"),
        "pages/2_Dataset_and_Integrity.py",
    ),
    ResearchStage(
        "How QuCNet features are formed", "Architecture",
        "Phase 6: 128×128 grayscale → 2×2 patches → fixed four-qubit simulated circuit → 64×64×4 channels → dense binary classifier.",
        "PennyLane default.qubit uses four RX encodings, two fixed seeded RY/CNOT-ring layers and four Pauli-Z outputs per patch.",
        "The circuit is real simulated quantum feature mapping, but its parameters are fixed and it does not run on quantum hardware.",
        "A synthetic patch demo exposes the actual mathematical transformation without medical-image uploads or retraining.",
        "I reconstructed the fixed, simulated four-qubit feature transformation described by the available source. In this workbench it is illustrated with synthetic values; the outputs are circuit expectations, not medical classifications.",
        "Why does the architecture view use synthetic inputs rather than live ultrasound inference?",
        "The application is designed for architecture and evaluation auditing. The archived model has no demonstrated clinical validity, so the circuit is illustrated on synthetic inputs rather than used for diagnostic predictions.",
        ("05_Source_Code/utilities/quantum_utils.py", "05_Source_Code/phase6_qucnet_model.py", "05_Source_Code/phase5_quantum_filter.py"),
        "pages/1_Interactive_Architecture_Lab.py",
    ),
    ResearchStage(
        "Binary QuCNet evaluation", "Historical results",
        "Phase 16 binary QuCNet confusion matrix and linked metric cards.",
        "On a validation-selected historical 70-image holdout, accuracy was 38/70 = 54.29%, while precision was 32/57 = 56.14%; recall 82.05%, specificity 19.35%.",
        "The model detected 32/39 high-suspicion proxy cases but produced 25 false positives among 31 low-suspicion proxy cases.",
        "Accuracy and sensitivity alone can conceal severe class-specific errors; the historical holdout was used for checkpoint selection.",
        "My recorded binary evaluation showed 38 correct classifications among 70 images: TN=6, FP=25, FN=7 and TP=32. Accuracy was 54.29%; the separate 56.14% figure was precision. This holdout was used during checkpoint selection.",
        "How do the reported accuracy and precision differ?",
        "54.29% is the recorded accuracy (38/70). 56.14% is positive-class precision (32/57). They measure different quantities.",
        ("07_Results/phase16_binary_qucnet_evaluation/binary_evaluation_metrics.json", "05_Source_Code/phase6_train.py"),
        "pages/3_Experimental_Results_Lab.py",
    ),
    ResearchStage(
        "Six-class comparison", "Historical results",
        "Phase 9 paper-structured quantum-feature CNN versus Phase 10 classical grayscale CNN on the historical 70-image six-class comparison.",
        "Quantum accuracy 13/70 = 18.57%; classical accuracy 17/70 = 24.29%; majority reference 19/70 = 27.14%.",
        "The quantum reconstruction did not outperform the matched classical comparator in this historical comparison.",
        "The comparison motivates accuracy-gap analysis rather than a claim of quantum superiority.",
        "I compared the six-class quantum-feature and grayscale models on the recorded historical image-level setting. Their results belong to this comparison, not to the separate Phase 14 ROI evaluation or the paper's original samples.",
        "What limits the six-class comparison with the published result?",
        "The exact original samples, target provenance, normalization and some implementation details were not recoverable. Different methods and evaluations prevent a direct causal explanation of the difference.",
        ("07_Results/phase9_paper_tirads_class_weighted/test_metrics.json", "07_Results/phase10_classical_tirads_class_weighted/test_metrics.json"),
        "pages/3_Experimental_Results_Lab.py",
    ),
    ResearchStage(
        "Stricter dataset splitting", "Method integrity",
        "Recorded-case-ID-separated ROI dataset: 440 distinct ROI geometries from 291 recorded case IDs.",
        "There are 311 train ROIs/203 case IDs, 58 validation ROIs/44 case IDs, and 71 historical test ROIs/44 case IDs; recorded case overlap is zero.",
        "The stricter partition addresses a limitation in earlier image-level splitting, which shared some recorded case IDs across subsets.",
        "Case-ID separation improves the audit but does NOT prove distinct patients across identifiers.",
        "I separated the ROI data by recorded case ID and checked for identifier overlap. That supports case-ID grouping, but the available information does not establish that different identifiers are different patients.",
        "Does zero recorded case-ID overlap establish patient independence?",
        "No. The recorded case IDs were separated, but the correspondence between IDs and distinct patient identities was not independently verified.",
        ("07_Results/phase12_case_level_split/split_distribution.json", "07_Results/phase13_roi_dataset/roi_dataset_distribution.json", "07_Results/phase11_case_annotation_audit/previous_split_leakage_report.txt"),
        "pages/2_Dataset_and_Integrity.py",
    ),
    ResearchStage(
        "Phase 14: single-class prediction collapse", "Reliability",
        "Phase 14 strict classical ROI CNN predicted TI-RADS 3 for every 71 test ROIs and every 44 recorded test cases.",
        "Only 4/71 ROI predictions (5.63%) and 2/44 case predictions (4.55%) were correct.",
        "This is complete single-class prediction collapse in the Phase 14 classical ROI model, not a quantum-model accuracy.",
        "A classifier can appear to run correctly while failing to distinguish categories.",
        "The strict Phase 14 CNN completed evaluation, yet predicted TI-RADS 3 for every test ROI and case. I retained the 5.63% ROI and 4.55% case-level findings rather than treating completed execution as evidence of reliable classification.",
        "Did BatchNorm or padding definitely cause the collapse?",
        "No. The diagnostic changed outputs, but the studies do not isolate a unique causal mechanism for the original collapse.",
        ("07_Results/phase14_roi_categorical_baseline/evaluation_metrics.json", "07_Results/exp_bn_001/metrics.json"),
        "pages/4_Reliability_and_Accuracy_Gap.py",
    ),
    ResearchStage(
        "Matched classifier families", "Development-only controls",
        "Eight fixed-feature/classifier configurations: classical 64-D versus simulated quantum 64-D features under LR, RBF-SVM, random forest and 7-NN.",
        "The seed-42 fixed quantum random forest scored 63/203 = 31.03%, versus 54/203 = 26.60% for classical random forest; development majority was 57/203 = 28.08%.",
        "The quantum RF had the highest observed accuracy at that seed but still missed all 10 TI-RADS 3 cases.",
        "This is exploratory fivefold case-grouped out-of-fold analysis on training cases—not the original QuCNet CNN or an external clinical test.",
        "A fixed-quantum random forest scored higher in one development-fold comparison, but it missed all 10 TI-RADS 3 cases. That observation, alongside statistical uncertainty, does not establish quantum superiority.",
        "Did the fixed quantum random forest prove quantum advantage?",
        "No. The fixed circuit was a simulated feature map; the advantage at one fold assignment was exploratory and did not establish robust statistical or computational superiority.",
        ("07_Results/exp_models_001/metrics.json", "07_Results/exp_qmap_001/metrics.json"),
        "pages/3_Experimental_Results_Lab.py",
    ),
    ResearchStage(
        "Sensitivity to method and fold choices", "Reliability",
        "Repeated case-fold stability, BatchNorm running-statistics recalibration, preprocessing choices and feature fusion.",
        "Across five fold seeds, RF mean accuracies were 27.49% classical and 29.36% fixed quantum on the same 203 cases. BatchNorm validation accuracy changed from 2/58=3.45% to 11/58=18.97%, below 17/58=29.31% majority.",
        "Classifier ranking was fold-sensitive and minority-class failures persisted; recalibration altered predictions but did not rescue reliable performance.",
        "These studies improve understanding, not clinical classification accuracy; five seeds are correlated repeats of the same cases.",
        "I examined the saved five-seed, BatchNorm, preprocessing and fusion analyses as separate diagnostics. Fusion changed the correct-case count from 45 to 46 of 203, with McNemar p=1.00, so a meaningful improvement was not established.",
        "If preprocessing changes the score, why not choose the highest?",
        "These are exploratory development comparisons with trade-offs and uncertainty, not an independently validated hyperparameter search. For example direct resizing distorts lesion aspect ratio.",
        ("07_Results/exp_stability_001/metrics.json", "07_Results/exp_bn_001/metrics.json", "07_Results/exp_fusion_001/metrics.json", "07_Results/exp_preproc_001/metrics.json"),
        "pages/4_Reliability_and_Accuracy_Gap.py",
    ),
    ResearchStage(
        "Software contribution: QuCNet Research Explorer", "Software contribution",
        "QuCNet Research Explorer integrates source-backed architecture explanations, a synthetic circuit calculator, dataset integrity summaries, archived experimental comparisons and a research-evidence review.",
        "The app runs on saved aggregate evidence without TensorFlow, PennyLane, raw medical files, checkpoints or model training at runtime.",
        "The contribution is an interactive reproducibility and evaluation-audit workbench, not a new AI classifier or medical diagnostic system.",
        "It makes a complex, imperfect research record understandable, traceable and reviewable in a live demonstration.",
        "I developed a separate Streamlit workbench for examining the completed research record. Its contribution is evidence integration, interaction and traceability; the previously trained models and their recorded results remain unchanged.",
        "What does the separate Streamlit application contribute?",
        "The software contribution is the aggregate-evidence pipeline, interactive architecture and circuit analysis, experimental-comparison views, integrity checks and source traceability. It does not retrain QuCNet or improve historical classifier accuracy.",
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
        "QUCNET RESEARCH EXPLORER — RESEARCH EVIDENCE SUMMARY",
        "Independent architecture, reproducibility and evaluation-audit workbench",
        "Experimental outcomes come from archived research; synthetic calculations are identified separately.",
        "",
    ]
    for index, slide in enumerate(STEPS, 1):
        lines += [
            f"{index:02d}. {slide.title.upper()} — {slide.section}",
            "ON SCREEN: " + slide.shown,
            "INTERPRETATION: " + slide.interpretation,
            "FINDING: " + slide.finding,
            "WHY IT MATTERS: " + slide.significance,
            "RESEARCHER COMMENTARY: " + slide.speaker_notes,
            "RESEARCH QUESTION: " + slide.question,
            "RESPONSE: " + slide.answer,
            "SOURCE FILES: " + "; ".join(slide.source_paths),
            "",
        ]
    lines += [
        "LIMITATIONS: No pathology-confirmed clinical evaluation; no verified patient identity independence; no independent six-class external cohort; no quantum superiority claim.",
        "The frozen research source is read-only. Mini-project software is a separate artifact.",
    ]
    return "\n".join(lines) + "\n"
