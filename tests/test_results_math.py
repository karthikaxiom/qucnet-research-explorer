"""Numerical safety checks for the Results Laboratory."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evidence.results_math import BinaryConfusion, multiclass_summary


class ResultsMathTests(unittest.TestCase):
    def test_phase16_metrics(self):
        v = BinaryConfusion(tn=6, fp=25, fn=7, tp=32)
        m = v.metrics()
        self.assertEqual(v.n, 70)
        self.assertAlmostEqual(m["accuracy"], 38/70)
        self.assertAlmostEqual(m["precision"], 32/57)
        self.assertAlmostEqual(m["recall"], 32/39)
        self.assertAlmostEqual(m["specificity"], 6/31)
        self.assertAlmostEqual(m["f1"], 2/3)

    def test_what_if_keeps_historical_class_totals(self):
        fp, fn = 10, 4
        v = BinaryConfusion(tn=31-fp, fp=fp, fn=fn, tp=39-fn)
        self.assertEqual(v.n, 70)
        self.assertEqual(v.tn+v.fp, 31)
        self.assertEqual(v.fn+v.tp, 39)
        self.assertGreater(v.metrics()["accuracy"], 38/70)

    def test_undefined_precision_is_not_zero(self):
        v = BinaryConfusion(tn=31, fp=0, fn=39, tp=0)
        self.assertIsNone(v.metrics()["precision"])
        self.assertIsNone(v.metrics()["f1"])

    def test_invalid_binary_counts(self):
        for counts in ((0, 0, 0, 0), (-1, 0, 0, 1), (1.0, 0, 0, 1)):
            with self.assertRaises(ValueError):
                BinaryConfusion(*counts)

    def test_multiclass_accuracy_and_recall(self):
        m = [[3, 1, 0], [1, 0, 1], [0, 1, 2]]
        v = multiclass_summary(m)
        self.assertEqual(v["n"], 9)
        self.assertEqual(v["correct"], 5)
        self.assertAlmostEqual(v["accuracy"], 5/9)
        self.assertEqual(v["class_recall"], [3/4, 0, 2/3])

    def test_multiclass_rejects_wrong_shape(self):
        for matrix in ([], [[1,2]], [[-1]], [[0]]):
            with self.assertRaises(ValueError):
                multiclass_summary(matrix)


if __name__ == "__main__":
    unittest.main()
