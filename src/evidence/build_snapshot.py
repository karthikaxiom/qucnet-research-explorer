"""Build a privacy-safe research evidence snapshot.

Reads the frozen research repository without modifying it.
Exports only verified aggregate results and provenance.
"""

import hashlib
import json
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
RESEARCH = Path(r"D:\QCNN_Thyroid_Paper_Reproduction")

sources = {}


def read_json(relative_path):
    path = RESEARCH / relative_path

    if not path.is_file():
        raise FileNotFoundError(path)

    raw = path.read_bytes()
    sources[relative_path] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def verify(actual, expected, label):
    if abs(float(actual) - float(expected)) > 1e-8:
        raise ValueError(
            f"{label}: expected {expected}, found {actual}"
        )


if not (PROJECT / ".gitignore").is_file():
    raise RuntimeError("Not running inside the mini-project.")

binary = read_json(
    "07_Results/phase16_binary_qucnet_evaluation/"
    "binary_evaluation_metrics.json"
)

quantum6 = read_json(
    "07_Results/phase9_paper_tirads_class_weighted/"
    "test_metrics.json"
)

classical6 = read_json(
    "07_Results/phase10_classical_tirads_class_weighted/"
    "test_metrics.json"
)

split = read_json(
    "07_Results/phase12_case_level_split/"
    "split_distribution.json"
)

roi_dataset = read_json(
    "07_Results/phase13_roi_dataset/"
    "roi_dataset_distribution.json"
)

roi_eval = read_json(
    "07_Results/phase14_roi_categorical_baseline/"
    "evaluation_metrics.json"
)

cm = binary["confusion_matrix"]

verify(binary["accuracy"], 38 / 70, "Binary accuracy")
verify(
    binary["precision_positive_malignant"],
    32 / 57,
    "Binary precision"
)
verify(quantum6["accuracy"], 13 / 70, "Six-class quantum")
verify(classical6["accuracy"], 17 / 70, "Six-class classical")
verify(roi_eval["roi_level"]["accuracy"], 4 / 71, "ROI accuracy")
verify(roi_eval["case_level"]["accuracy"], 2 / 44, "Case accuracy")

if (
    cm["true_negative"],
    cm["false_positive"],
    cm["false_negative"],
    cm["true_positive"],
) != (6, 25, 7, 32):
    raise ValueError("Binary confusion matrix mismatch")

audit = split["audit"]

if audit["total_cases"] != 291 or audit["total_rois"] != 440:
    raise ValueError("Dataset audit count mismatch")

if any(audit[k] != 0 for k in (
    "train_validation_overlap_count",
    "train_test_overlap_count",
    "validation_test_overlap_count"
)):
    raise ValueError("Unexpected recorded case-ID overlap")

roi_splits = roi_dataset["processed_dataset"]["split_audit"]

for name, rois, cases in (
    ("train", 311, 203),
    ("validation", 58, 44),
    ("test", 71, 44),
):
    record = roi_splits[name]
    if record["sample_count"] != rois:
        raise ValueError(f"{name} ROI count mismatch")
    if record["unique_case_count"] != cases:
        raise ValueError(f"{name} case count mismatch")

commit = subprocess.check_output(
    ["git", "-C", str(RESEARCH), "rev-parse", "HEAD"],
    text=True
).strip()

status = subprocess.check_output(
    ["git", "-C", str(RESEARCH), "status", "--porcelain"],
    text=True
).strip()

if status:
    raise RuntimeError("Research checkout is not clean.")

snapshot = {
    "schema_version": 1,
    "binary_qucnet": {
        "evaluation": "validation-selected historical holdout",
        "target": "TI-RADS-derived binary suspicion proxy",
        "n_images": binary["test_samples"],
        "accuracy": binary["accuracy"],
        "precision": binary["precision_positive_malignant"],
        "recall": binary["sensitivity_recall_malignant"],
        "specificity": binary["specificity_benign"],
        "f1": binary["f1_score_malignant"],
        "roc_auc": binary["roc_auc"],
        "average_precision": binary["average_precision"],
        "confusion_matrix": cm,
    },
    "six_class_historical": {
        "n_images": 70,
        "quantum_accuracy": quantum6["accuracy"],
        "quantum_balanced_accuracy": quantum6["balanced_accuracy"],
        "quantum_macro_f1": quantum6["macro_f1"],
        "classical_accuracy": classical6["accuracy"],
        "classical_balanced_accuracy": classical6["balanced_accuracy"],
        "classical_macro_f1": classical6["macro_f1"],
        "majority_accuracy": quantum6["majority_baseline_accuracy"],
    },
    "strict_roi_cnn": {
        "n_rois": roi_eval["test_roi_count"],
        "n_case_ids": roi_eval["test_case_count"],
        "roi_accuracy": roi_eval["roi_level"]["accuracy"],
        "case_accuracy": roi_eval["case_level"]["accuracy"],
        "roi_macro_f1": roi_eval["roi_level"]["macro_f1"],
        "case_macro_f1": roi_eval["case_level"]["macro_f1"],
        "observed_prediction": "TI-RADS 3 for every test ROI and case",
    },
    "case_split": {
        "total_case_ids": audit["total_cases"],
        "total_rois": audit["total_rois"],
        "training_cases": audit["train_cases"],
        "validation_cases": audit["validation_cases"],
        "historical_test_cases": audit["test_cases"],
        "training_rois": 311,
        "validation_rois": 58,
        "historical_test_rois": 71,
        "recorded_case_id_overlap": 0,
        "patient_independence_verified": False,
    },
}

provenance = {
    "schema_version": 1,
    "research_repository": (
        "https://github.com/karthikaxiom/"
        "QCNN_Thyroid_Paper_Reproduction"
    ),
    "source_commit": commit,
    "source_branch": "research/bn-recalibration-audit",
    "source_file_sha256": sources,
    "data_policy": (
        "Aggregate evidence only. No raw medical images, XML, "
        "case IDs, predictions, probabilities or checkpoints."
    ),
    "retraining_performed_by_app": False,
}

outputs = {
    PROJECT / "data" / "research_snapshot.json": snapshot,
    PROJECT / "provenance.json": provenance,
}

for path in outputs:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")

for path, content in outputs.items():
    path.write_text(
        json.dumps(content, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )

print("\nSTEP 5B PASS")
print("Research commit:", commit)
print("Verified source JSON files:", len(sources))
print("Binary accuracy:", round(binary["accuracy"] * 100, 2), "%")
print("Six-class quantum:", round(quantum6["accuracy"] * 100, 2), "%")
print("Strict ROI accuracy:", round(
    roi_eval["roi_level"]["accuracy"] * 100, 2
), "%")
print("Research repository unchanged.")
print("Created: data/research_snapshot.json")
print("Created: provenance.json")
