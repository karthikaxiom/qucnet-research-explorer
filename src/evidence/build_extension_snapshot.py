"""Import verified aggregate extension evidence, without model training."""

import hashlib
import json
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
RESEARCH = Path(r"D:\QCNN_Thyroid_Paper_Reproduction")
SOURCE_HASHES = {}


def read_source(relative):
    path = RESEARCH / relative
    content = path.read_bytes()
    SOURCE_HASHES[relative] = hashlib.sha256(content).hexdigest()
    return json.loads(content)


def check(value, expected, name):
    if abs(float(value) - float(expected)) > 1e-8:
        raise ValueError(f"{name}: unexpected value {value}")


provenance = json.loads(
    (PROJECT / "provenance.json").read_text(encoding="utf-8")
)

commit = subprocess.check_output(
    ["git", "-C", str(RESEARCH), "rev-parse", "HEAD"],
    text=True
).strip()

if commit != provenance["source_commit"]:
    raise RuntimeError("Research commit differs from initial evidence.")

if subprocess.check_output(
    ["git", "-C", str(RESEARCH), "status", "--porcelain"],
    text=True
).strip():
    raise RuntimeError("Original research working tree is not clean.")

root = "07_Results/"

models = read_source(root + "exp_models_001/metrics.json")
stability = read_source(root + "exp_stability_001/metrics.json")
fusion = read_source(root + "exp_fusion_001/metrics.json")
preproc = read_source(root + "exp_preproc_001/metrics.json")
bn = read_source(root + "exp_bn_001/metrics.json")

if models["training_cases"] != 203 or models["training_rois"] != 311:
    raise ValueError("Classifier denominator mismatch")

if stability["seeds"] != [42, 101, 202, 303, 404]:
    raise ValueError("Unexpected stability seeds")

if models["original_test_accessed"] or bn["test_data_accessed"]:
    raise ValueError("Unexpected historical test access")

classifiers = {}

for name, record in models["models"].items():
    case = record["case"]
    matrix = case["confusion_matrix"]

    if sum(map(sum, matrix)) != 203:
        raise ValueError(f"{name}: case denominator mismatch")

    correct = sum(matrix[i][i] for i in range(6))
    check(case["accuracy"], correct / 203, name)

    classifiers[name] = {
        "accuracy": case["accuracy"],
        "balanced_accuracy": case["balanced_accuracy"],
        "macro_f1": case["macro_f1"],
        "confusion_matrix": matrix,
    }

if len(classifiers) != 8:
    raise ValueError("Expected eight classifier configurations")

check(
    classifiers["fixed_quantum_64__random_forest"]["accuracy"],
    63 / 203,
    "Quantum random forest"
)

check(
    classifiers["classical_64__random_forest"]["accuracy"],
    54 / 203,
    "Classical random forest"
)

# Both random forests failed to recognize TI-RADS 3.
for name in (
    "classical_64__random_forest",
    "fixed_quantum_64__random_forest",
):
    if classifiers[name]["confusion_matrix"][1][1] != 0:
        raise ValueError("TI-RADS 3 finding mismatch")

seed_summary = {}

for name, record in stability["across_seed_summary"].items():
    seed_summary[name] = {
        metric: {
            "mean": record[metric]["mean"],
            "std": record[metric]["std_across_seeds_ddof_1"],
            "per_seed": record[metric]["per_seed"],
        }
        for metric in ("accuracy", "balanced_accuracy", "macro_f1")
    }

check(
    seed_summary["fixed_quantum_64__random_forest"]
    ["accuracy"]["mean"],
    0.2935960591133005,
    "Five-seed quantum RF"
)

def case_scores(data):
    return {
        name: {
            key: float(record["pooled_case"][key])
            for key in ("accuracy", "balanced_accuracy", "macro_f1")
        }
        for name, record in data["models"].items()
    }


fusion_scores = case_scores(fusion)
preproc_scores = case_scores(preproc)

check(fusion_scores["image"]["accuracy"], 45 / 203, "Image")
check(fusion_scores["geometry"]["accuracy"], 39 / 203, "Geometry")
check(fusion_scores["fusion"]["accuracy"], 46 / 203, "Fusion")
check(
    preproc_scores["direct_stretch"]["accuracy"],
    57 / 203,
    "Direct resize"
)

bn_validation = {
    item["condition"]: {
        "accuracy": item["accuracy"],
        "majority_reference": item["majority_baseline_accuracy"],
        "samples": item["samples"],
    }
    for item in bn["evaluations"]
    if item["partition"] == "validation"
}

check(bn_validation["original"]["accuracy"], 2 / 58, "Original BN")
check(
    bn_validation["recalibrated"]["accuracy"],
    11 / 58,
    "Recalibrated BN"
)

snapshot = {
    "schema_version": 1,
    "source_commit": commit,
    "evidence_scope": (
        "Exploratory development case-fold analyses and "
        "historical validation-only BatchNorm diagnostic"
    ),
    "classifier_comparison": {
        "case_count": 203,
        "roi_count": 311,
        "models": classifiers,
    },
    "stability": {
        "case_count": 203,
        "seeds": stability["seeds"],
        "models": seed_summary,
    },
    "fusion": {
        "case_count": 203,
        "models": fusion_scores,
    },
    "preprocessing": {
        "case_count": 203,
        "models": preproc_scores,
    },
    "batchnorm": {
        "partition": "historical_validation",
        "learned_weights_updated": bn["learned_weights_updated"],
        "conditions": bn_validation,
    },
    "source_file_sha256": SOURCE_HASHES,
}

output = PROJECT / "data" / "extension_snapshot.json"

if output.exists():
    raise FileExistsError("Extension snapshot already exists")

output.write_text(
    json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8"
)

print("\nSTEP 5C PASS")
print("Source commit:", commit)
print("Extension source files verified:", len(SOURCE_HASHES))
print("Classifier configurations:", len(classifiers))
print("Stability seeds:", stability["seeds"])
print("Quantum RF accuracy: 31.03% (63/203)")
print("Fusion accuracy: 22.66% (46/203)")
print("BatchNorm validation: 3.45% -> 18.97%")
print("Created: data/extension_snapshot.json")
print("No model training performed.")
