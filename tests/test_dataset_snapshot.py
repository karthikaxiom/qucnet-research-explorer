"""Dependency-free tests for privacy-safe dataset snapshot reductions."""
import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "evidence" / "build_dataset_snapshot.py"
spec = importlib.util.spec_from_file_location("build_dataset_snapshot", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DatasetSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.inventory = {"master_rows": 480, "case_ids": 390, "labeled_images": 349, "unlabeled_images": 131}
        self.counts = {"2": 42, "3": 19, "4a": 96, "4b": 79, "4c": 68, "5": 45}
        self.distribs = {
            "train": {"2": 33, "3": 14, "4a": 91, "4b": 77, "4c": 62, "5": 34},
            "validation": {"2": 7, "3": 2, "4a": 17, "4b": 14, "4c": 9, "5": 9},
            "test": {"2": 9, "3": 4, "4a": 18, "4b": 16, "4c": 15, "5": 9},
        }
        self.data = {
            "inventory": self.inventory,
            "labels": "TI-RADS distribution:\ntirads\nNaN    131\n4a 96\n4b 79\n4c 68\n5 45\n2 42\n3 19\nName: count, dtype: int64\n",
            "split": {"audit": {
                "total_cases": 291, "total_rois": 440,
                "train_validation_overlap_count": 0, "train_test_overlap_count": 0,
                "validation_test_overlap_count": 0, "duplicated_roi_assignments": 0,
            }},
            "roi": {"processed_dataset": {"split_audit": {
                key: {"sample_count": sum(vals.values()), "unique_roi_count": sum(vals.values()),
                      "unique_case_count": {"train": 203, "validation": 44, "test": 44}[key],
                      "class_distribution": vals}
                for key, vals in self.distribs.items()
            }}},
            "duplicates": {"counts": {"raw_image_files": 480, "distinct_file_sha256": 471,
                "duplicate_groups": 6, "files_in_duplicate_groups": 15, "redundant_files": 9,
                "cross_case_duplicate_groups": 2, "cross_assigned_split_duplicate_groups": 0}},
        }

    def test_parse_distribution(self):
        counts = module.parse_tirads_distribution(self.data["labels"])
        self.assertEqual(counts["4a"], 96)
        self.assertEqual(sum(counts.values()), 480)

    def test_snapshot_counts(self):
        summary = module.validated_summary(self.data)
        self.assertEqual(summary["strict_roi"]["roi_geometries"], 440)
        self.assertEqual(summary["strict_roi"]["splits"]["test"]["rois"], 71)
        self.assertFalse(summary["strict_roi"]["patient_identity_independence_verified"])

    def test_invalid_overlap_rejected(self):
        self.data["split"]["audit"]["train_test_overlap_count"] = 1
        with self.assertRaises(ValueError):
            module.validated_summary(self.data)

    def test_missing_label_mismatch_rejected(self):
        self.data["inventory"]["unlabeled_images"] = 130
        with self.assertRaises(ValueError):
            module.validated_summary(self.data)

    def test_wrong_roi_class_total_rejected(self):
        self.distribs["test"]["3"] = 5
        with self.assertRaises(ValueError):
            module.validated_summary(self.data)

    def test_duplicate_count_mismatch_rejected(self):
        self.data["duplicates"]["counts"]["distinct_file_sha256"] = 470
        with self.assertRaises(ValueError):
            module.validated_summary(self.data)

    def test_no_sensitive_individual_records(self):
        text = str(module.validated_summary(self.data)).lower()
        for forbidden in ("patient_name", "case_ids_roi_train", "image_path", "predictions.csv"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
