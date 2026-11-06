"""Safety and source-fidelity tests for the presenter narrative."""
import unittest
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evidence.presenter_flow import STEPS, speaking_script, validate_evidence


class PresenterContentTests(unittest.TestCase):
    def test_ten_distinct_steps(self):
        self.assertEqual(len(STEPS), 10)
        self.assertEqual(len({step.title for step in STEPS}), 10)

    def test_source_references_are_relative_and_safe(self):
        for step in STEPS:
            self.assertTrue(step.source_paths)
            for p in step.source_paths:
                self.assertFalse(p.startswith("/"))
                self.assertNotIn("..", Path(p).parts)
                self.assertNotIn("02_Dataset_Raw/", p)
                self.assertNotIn("03_Dataset_Processed/", p)
                self.assertNotIn("06_Models/", p)

    def test_speaking_notes_nonempty_and_include_nonclaims(self):
        script = speaking_script()
        self.assertIn("54.29%", script)
        self.assertIn("56.14%", script)
        self.assertIn("TI-RADS", script)
        self.assertIn("not pathology-confirmed", script)
        self.assertIn("No pathology-confirmed clinical evaluation", script)
        self.assertNotIn("patient-level prediction", script)
        self.assertGreater(len(script), 2000)

    def test_every_stage_has_panel_question_answer(self):
        for step in STEPS:
            self.assertGreater(len(step.question), 15)
            self.assertGreater(len(step.answer), 50)
            self.assertGreater(len(step.speaker_notes), 50)

    def test_strict_evidence_accepts_verified_snapshot_schema(self):
        h,e,d,p = self.sample_snapshots()
        validate_evidence(h,e,d,p)

    def test_rejects_mismatched_research_commit(self):
        h,e,d,p = self.sample_snapshots()
        d["source_commit"] = "f" * 40
        with self.assertRaises(ValueError):
            validate_evidence(h,e,d,p)

    def test_rejects_confusion_accuracy_mismatch(self):
        h,e,d,p = self.sample_snapshots()
        h["binary_qucnet"]["accuracy"] = 0.5614
        with self.assertRaises(ValueError):
            validate_evidence(h,e,d,p)

    def test_rejects_unproven_patient_independence(self):
        h,e,d,p = self.sample_snapshots()
        d["strict_roi"]["patient_identity_independence_verified"] = True
        with self.assertRaises(ValueError):
            validate_evidence(h,e,d,p)

    @staticmethod
    def sample_snapshots():
        sha = "e848013b9ff943d816a8c90840255233a5ae7b8d"
        hist = {
            "binary_qucnet": {"n_images":70, "accuracy":38/70, "precision":32/57,
                "confusion_matrix":{"true_negative":6,"false_positive":25,"false_negative":7,"true_positive":32}},
            "six_class_historical":{"n_images":70,"quantum_accuracy":13/70,"classical_accuracy":17/70},
            "strict_roi_cnn":{"n_rois":71,"n_case_ids":44,"roi_accuracy":4/71,"case_accuracy":2/44},
        }
        ext = {
            "source_commit":sha,
            "classifier_comparison":{"case_count":203,"roi_count":311,"models":{
                "fixed_quantum_64__random_forest":{"accuracy":63/203}}},
            "stability":{"seeds":[42,101,202,303,404],"models":{
                "fixed_quantum_64__random_forest":{"accuracy":{"mean":0.2935960591133005}}}},
            "batchnorm":{"conditions":{"recalibrated":{"accuracy":11/58}}},
        }
        dataset = {"source_commit":sha,"inventory":{"raw_jpg_images":480,
            "xml_records":390,"images_with_tirads":349,"images_missing_tirads":131},
            "strict_roi":{"roi_geometries":440,"recorded_case_ids":291,
            "case_id_overlap_between_splits":0,"patient_identity_independence_verified":False}}
        prov = {"source_commit":sha}
        return deepcopy(hist),deepcopy(ext),deepcopy(dataset),deepcopy(prov)


if __name__ == "__main__":
    unittest.main()
