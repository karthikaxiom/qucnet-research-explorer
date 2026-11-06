"""Create a privacy-safe dataset/integrity snapshot from the frozen research checkout.

Run only from the separate mini-project. Reads research source files; writes
one new aggregate JSON under the mini-project. No patient-level data is exported.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = Path(r"D:\QCNN_Thyroid_Paper_Reproduction")
REPO_URL = "https://github.com/karthikaxiom/QCNN_Thyroid_Paper_Reproduction"
LABELS = ("2", "3", "4a", "4b", "4c", "5")

SOURCES = {
    "inventory": "07_Results/exp_data_001/audit.json",
    "roi": "07_Results/phase13_roi_dataset/roi_dataset_distribution.json",
    "split": "07_Results/phase12_case_level_split/split_distribution.json",
    "duplicates": "07_Results/exp_duplicate_001/local_sha256/exact_raw_image_duplicates.json",
    "labels": "07_Results/phase7_reproducibility_audit.txt",
}


def parse_tirads_distribution(text: str) -> dict[str, int]:
    """Read only aggregate TI-RADS counts from the archived audit text."""
    match = re.search(r"TI-RADS distribution:\s*\n(?P<rows>(?:[^\n]*\n)*)", text)
    if match is None:
        raise ValueError("TI-RADS distribution section missing")
    counts: dict[str, int] = {}
    for line in match.group("rows").splitlines():
        record = re.fullmatch(r"\s*(NaN|4a|4b|4c|5|2|3)\s+(\d+)\s*", line)
        if record:
            label, count = record.groups()
            if label in counts:
                raise ValueError(f"Duplicate TI-RADS count for {label}")
            counts[label] = int(count)
    if set(counts) != set(LABELS) | {"NaN"}:
        raise ValueError(f"Unexpected TI-RADS labels: {sorted(counts)}")
    return counts


def validated_summary(data: dict) -> dict:
    """Reduce read-only source files to safe, cross-checked counts."""
    inventory = data["inventory"]
    roi = data["roi"]["processed_dataset"]["split_audit"]
    split = data["split"]["audit"]
    duplicates = data["duplicates"]["counts"]
    labels = parse_tirads_distribution(data["labels"])

    if (inventory["master_rows"], inventory["labeled_images"], inventory["unlabeled_images"], inventory["case_ids"]) != (480, 349, 131, 390):
        raise ValueError("Raw dataset inventory mismatch")
    if sum(labels.values()) != 480 or labels["NaN"] != 131:
        raise ValueError("TI-RADS total or missingness mismatch")
    if sum(labels[x] for x in LABELS) != 349:
        raise ValueError("TI-RADS labeled count mismatch")
    if (split["total_cases"], split["total_rois"]) != (291, 440):
        raise ValueError("Strict ROI/case totals mismatch")
    if any(split[k] != 0 for k in (
        "train_validation_overlap_count", "train_test_overlap_count", "validation_test_overlap_count"
    )):
        raise ValueError("Recorded case IDs overlap between splits")
    if split["duplicated_roi_assignments"] != 0:
        raise ValueError("Duplicate ROI split assignments")
    if (duplicates["raw_image_files"], duplicates["distinct_file_sha256"],
        duplicates["duplicate_groups"], duplicates["files_in_duplicate_groups"],
        duplicates["redundant_files"], duplicates["cross_assigned_split_duplicate_groups"]) != (480, 471, 6, 15, 9, 0):
        raise ValueError("Raw-image duplicate audit mismatch")
    if duplicates["raw_image_files"] - duplicates["distinct_file_sha256"] != duplicates["redundant_files"]:
        raise ValueError("Raw duplicate arithmetic mismatch")

    splits: dict[str, dict] = {}
    for name, expected_rois, expected_cases in (("train", 311, 203), ("validation", 58, 44), ("test", 71, 44)):
        item = roi[name]
        if (item["sample_count"], item["unique_case_count"], item["unique_roi_count"]) != (expected_rois, expected_cases, expected_rois):
            raise ValueError(f"{name} count mismatch")
        distribution = {key: int(item["class_distribution"][key]) for key in LABELS}
        if sum(distribution.values()) != expected_rois:
            raise ValueError(f"{name} class counts don't add to ROI total")
        splits[name] = {
            "rois": expected_rois,
            "case_ids": expected_cases,
            "tirads_roi_distribution": distribution,
        }
    if sum(x["rois"] for x in splits.values()) != split["total_rois"]:
        raise ValueError("Strict ROI count does not sum")
    if sum(x["case_ids"] for x in splits.values()) != split["total_cases"]:
        raise ValueError("Case ID count does not sum")

    return {
        "schema_version": 1,
        "inventory": {
            "raw_jpg_images": inventory["master_rows"],
            "xml_records": inventory["case_ids"],
            "images_with_tirads": inventory["labeled_images"],
            "images_missing_tirads": inventory["unlabeled_images"],
            "tirads_image_distribution": {label: labels[label] for label in LABELS},
        },
        "strict_roi": {
            "roi_geometries": split["total_rois"],
            "recorded_case_ids": split["total_cases"],
            "splits": splits,
            "case_id_overlap_between_splits": 0,
            "patient_identity_independence_verified": False,
        },
        "raw_image_duplicates": {
            key: int(duplicates[key]) for key in (
                "raw_image_files", "distinct_file_sha256", "duplicate_groups",
                "files_in_duplicate_groups", "redundant_files",
                "cross_case_duplicate_groups", "cross_assigned_split_duplicate_groups"
            )
        },
        "caveats": [
            "TI-RADS grades are not pathology-confirmed malignancy diagnoses.",
            "Recorded case-ID separation does not prove unique patient identities.",
            "Raw-file duplicate screening is distinct from ROI-geometry deduplication.",
            "No raw images, XML records, identifiable case IDs or predictions are included.",
            "Source medical-image redistribution permission and privacy clearance are unresolved.",
        ],
    }


def build_snapshot(project: Path = ROOT, research: Path = RESEARCH) -> Path:
    project = project.resolve()
    research = research.resolve()
    if project == research or research in project.parents or project in research.parents:
        raise RuntimeError("Mini-project and research source must be separate directories")
    if not (project / "data" / "research_snapshot.json").is_file():
        raise RuntimeError("Run the verified initial snapshot builder first")

    provenance_path = project / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8-sig"))
    commit = subprocess.check_output(["git", "-C", str(research), "rev-parse", "HEAD"], text=True).strip()
    if commit != provenance["source_commit"]:
        raise RuntimeError("Frozen source commit differs from recorded provenance")
    status = subprocess.check_output(["git", "-C", str(research), "status", "--porcelain"], text=True).strip()
    if status:
        raise RuntimeError("Research checkout has local changes; refusing to import")

    source_hashes: dict[str, str] = {}
    source_data: dict[str, object] = {}
    for name, relpath in SOURCES.items():
        content = (research / relpath).read_bytes()
        source_hashes[relpath] = hashlib.sha256(content).hexdigest()
        source_data[name] = content.decode("utf-8-sig") if relpath.endswith(".txt") else json.loads(content)

    snapshot = validated_summary(source_data)
    snapshot["source_commit"] = commit
    snapshot["research_repository"] = REPO_URL
    snapshot["source_file_sha256"] = source_hashes

    destination = project / "data" / "dataset_integrity_snapshot.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    destination.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination


if __name__ == "__main__":
    path = build_snapshot()
    obj = json.loads(path.read_text(encoding="utf-8"))
    print("\nSTEP 9 IMPORT PASS")
    print("Source commit:", obj["source_commit"])
    print("Source files checked:", len(obj["source_file_sha256"]))
    print("Raw images:", obj["inventory"]["raw_jpg_images"])
    print("Missing TI-RADS labels:", obj["inventory"]["images_missing_tirads"])
    print("Strict ROIs:", obj["strict_roi"]["roi_geometries"])
    print("Recorded case-ID overlaps:", obj["strict_roi"]["case_id_overlap_between_splits"])
    print("Created:", path)
    print("No raw medical records copied or research files changed.")
