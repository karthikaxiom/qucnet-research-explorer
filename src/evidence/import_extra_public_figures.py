"""Read-only, commit-pinned whitelist importer for additional *aggregate* charts.

Use ONLY in the separate mini-project. Nothing is written to the frozen research checkout.
Default is dry-run; --apply writes to the mini-project after all checks pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

from src.evidence.visual_enhancements import ADDITIONAL_CHARTS

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DEFAULT = Path(r"D:\QCNN_Thyroid_Paper_Reproduction")
TARGET = ROOT / "assets" / "figures"


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT).strip()


def verify_png(data: bytes) -> None:
    if not data.startswith(b"\x89PNG\r\n\x1a\n") or len(data) < 50:
        raise ValueError("Missing or invalid PNG signature")


def verify_svg(data: bytes) -> None:
    # The chosen files are Plotly/Matplotlib vector charts, not embedded medical images.
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise ValueError("XML declaration with external entity/DTD is forbidden")
    root = ET.fromstring(data)
    if root.tag.split("}")[-1].lower() != "svg":
        raise ValueError("Not a vector SVG chart")
    forbidden = {"image", "script", "foreignobject", "iframe", "use"}
    for el in root.iter():
        if el.tag.split("}")[-1].lower() in forbidden:
            raise ValueError("Embedded image/script is forbidden")
        for key, val in el.attrib.items():
            attr = key.split("}")[-1].lower()
            if attr.startswith("on") or ((attr in {"href", "src"}) and val):
                raise ValueError("External/active SVG reference is forbidden")


def plan(source: Path) -> tuple[str, list[dict]]:
    expected = json.loads((ROOT / "provenance.json").read_text(encoding="utf-8"))["source_commit"]
    if git(source, "rev-parse", "HEAD") != expected:
        raise ValueError("Research checkout does not match frozen source commit")
    if git(source, "status", "--porcelain"):
        raise ValueError("Research checkout is not clean; refusing to copy")
    manifest = TARGET / "extra_figure_manifest.json"
    if manifest.exists():
        raise FileExistsError("Additional figure manifest exists; will not overwrite")
    entries = []
    for _, filename, caption, relative in ADDITIONAL_CHARTS:
        dest = TARGET / filename
        src = source / relative
        if not src.is_file() or dest.exists():
            raise FileNotFoundError(f"Missing source or already present in mini-project: {relative}")
        payload = src.read_bytes()
        if not (100 <= len(payload) <= 12_000_000):
            raise ValueError(f"Suspicious size: {relative}")
        (verify_svg if filename.endswith(".svg") else verify_png)(payload)
        # Git blob verification pins bytes to the archived commit, independently of SHA256 manifest.
        stored = git(source, "rev-parse", "HEAD:" + relative)
        actual = subprocess.check_output(["git", "hash-object", str(src)], text=True).strip()
        if actual != stored:
            raise ValueError(f"File differs from frozen Git object: {relative}")
        entries.append({"file": filename, "source_path": relative,
                        "sha256": hashlib.sha256(payload).hexdigest(), "caption": caption})
    return expected, entries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="copy approved charts into the separate mini-project")
    ap.add_argument("--source", type=Path, default=SOURCE_DEFAULT, help="frozen read-only research checkout")
    args = ap.parse_args()
    source = args.source.resolve()
    commit, rows = plan(source)
    print("SOURCE COMMIT:", commit)
    print("Aggregate-chart files checked:", len(rows))
    for item in rows:
        print("  VERIFIED:", item["source_path"], item["sha256"][:12])
    if not args.apply:
        print("DRY RUN COMPLETE — no files copied. Rerun with --apply after reviewing the whitelist.")
        return
    TARGET.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        for item in rows:
            src = source / item["source_path"]
            dest = TARGET / item["file"]
            shutil.copyfile(src, dest)
            created.append(dest)
            if hashlib.sha256(dest.read_bytes()).hexdigest() != item["sha256"]:
                raise RuntimeError(f"Copy integrity failed: {dest}")
        manifest = {"source_commit": commit, "scope": "Whitelisted aggregate charts only; visual publication review still required", "figures": rows}
        target = TARGET / "extra_figure_manifest.json"
        with target.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    print("EXTRA FIGURE IMPORT PASS — review the added charts before staging or pushing.")
    print("FROZEN SOURCE READ ONLY; no training, raw frames, predictions or checkpoints copied.")


if __name__ == "__main__":
    main()
