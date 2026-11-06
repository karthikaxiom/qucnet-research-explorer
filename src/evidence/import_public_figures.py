"""One-time, source-pinned, privacy-restricted import of derived research charts.

Never walks, reads or copies DDTI raw image/XML, case predictions or checkpoints.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

PROJECT = Path(__file__).resolve().parents[2]
RESEARCH = Path(r"D:\QCNN_Thyroid_Paper_Reproduction")
OUT = PROJECT / "assets" / "figures"

# Only verified, aggregate figure assets. Never add raw image previews here.
APPROVED = {
    "binary_confusion_matrix.png": "07_Results/phase16_binary_qucnet_evaluation/binary_confusion_matrix.png",
    "binary_roc_curve.png": "07_Results/phase16_binary_qucnet_evaluation/binary_roc_curve.png",
    "binary_precision_recall_curve.png": "07_Results/phase16_binary_qucnet_evaluation/binary_precision_recall_curve.png",
    "phase9_test_confusion_matrix.png": "07_Results/phase9_paper_tirads_class_weighted/test_confusion_matrix.png",
    "phase10_test_confusion_matrix.png": "07_Results/phase10_classical_tirads_class_weighted/test_confusion_matrix.png",
    "phase9_training_accuracy_plot.png": "07_Results/phase9_paper_tirads_class_weighted/training_accuracy_plot.png",
    "phase10_training_accuracy_plot.png": "07_Results/phase10_classical_tirads_class_weighted/training_accuracy_plot.png",
    "fig_08_matched_classifier_families_color.svg": "09_Output_images/README/fig_08_matched_classifier_families_color.svg",
}


def main():
    provenance = json.loads((PROJECT / "provenance.json").read_text(encoding="utf-8"))
    commit = subprocess.check_output(["git", "-C", str(RESEARCH), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(RESEARCH), "status", "--porcelain"], text=True).strip()
    if commit != provenance["source_commit"] or status:
        raise RuntimeError("Frozen research commit mismatch or working tree not clean; refusing import")
    if (OUT / "figure_manifest.json").exists():
        raise FileExistsError("Figure manifest already exists. Refusing to overwrite imported evidence")
    prepared = []
    for filename, relative in APPROVED.items():
        original = RESEARCH / relative
        dest = OUT / filename
        if not original.is_file() or dest.exists():
            raise FileNotFoundError(f"Missing source or destination already exists: {relative} / {dest}")
        contents = original.read_bytes()
        if len(contents) > 12_000_000:
            raise ValueError(f"Unusually large figure, manual review required: {relative}")
        if filename.endswith(".png"):
            if not contents.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError(f"Invalid PNG figure: {relative}")
        else:
            root = ET.fromstring(contents)
            if not root.tag.endswith("svg"):
                raise ValueError(f"Expected SVG chart: {relative}")
            if any(el.tag.rsplit("}",1)[-1].lower() in ("image", "foreignobject", "script") for el in root.iter()):
                raise ValueError(f"SVG contains embedded images/scripts: {relative}")
        prepared.append((original, dest, relative, hashlib.sha256(contents).hexdigest()))
    OUT.mkdir(parents=True, exist_ok=True)
    for original, dest, relative, sha in prepared:
        shutil.copyfile(original, dest)
        if hashlib.sha256(dest.read_bytes()).hexdigest()!=sha:
            raise RuntimeError(f"Figure copy hash mismatch: {relative}")
    manifest = {
        "source_commit":commit,
        "scope":"Whitelisted derived research charts only; no raw medical imagery or individual predictions",
        "figures":[{"file":d.name, "source_path":p,"sha256":sha} for _,d,p,sha in prepared]
    }
    (OUT/"figure_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print("STEP 10 FIGURES PASS")
    print("Whitelisted aggregate charts imported:",len(prepared))
    print("Source commit:",commit)
    print("Created: assets/figures/figure_manifest.json")
    print("Original research repository unchanged.")


if __name__ == "__main__":
    main()
