"""Numerical checks for published-paper and archived-reconstruction comparison."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.evidence.dataset_correspondence import PAPER, SOURCE_COMMIT, verified_comparison

ROOT = Path(__file__).resolve().parents[1]


class PaperDatasetCorrespondenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = json.loads((ROOT / "data/dataset_integrity_snapshot.json").read_text(encoding="utf-8-sig"))
        cls.research = json.loads((ROOT / "data/research_snapshot.json").read_text(encoding="utf-8-sig"))

    def test_published_paper_train_test_accounting(self):
        self.assertEqual(PAPER["binary_training"] + PAPER["binary_test"], 460)
        self.assertEqual(PAPER["six_class_training"] + PAPER["six_class_test"], 478)

    def test_aggregate_evidence_matches_source_snapshots(self):
        result = verified_comparison(self.dataset, self.research, SOURCE_COMMIT)
        self.assertEqual(len(result["rows"]), 6)
        self.assertEqual(result["recovered"]["images_with_tirads"], 349)
        self.assertEqual(result["recovered"]["images_missing_tirads"], 131)

    def test_rejects_mismatched_source_commit(self):
        with self.assertRaises(ValueError):
            verified_comparison(self.dataset, self.research, "0" * 40)

    def test_does_not_equate_xml_records_with_medical_cases(self):
        table = verified_comparison(self.dataset, self.research, SOURCE_COMMIT)["rows"]
        self.assertIn("different unit", table[1]["Recovered research archive"])
        self.assertIn("No verified pathology mapping", table[2]["Recovered research archive"])


if __name__ == "__main__":
    unittest.main()
