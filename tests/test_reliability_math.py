"""Automated verification for aggregate-only reliability mathematics."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evidence.reliability_math import classwise_from_confusion, development_scores, validate_bn_conditions


class ReliabilityMathTests(unittest.TestCase):
    def test_archived_quantum_rf_matrix(self):
        matrix = [
            [3,1,14,4,4,0], [1,0,1,5,3,0], [5,0,25,15,9,3],
            [3,0,18,22,5,3], [2,0,19,6,9,2], [0,0,7,5,5,4],
        ]
        result = classwise_from_confusion(matrix)
        self.assertEqual(result['n'], 203)
        self.assertEqual(result['correct'], 63)
        self.assertAlmostEqual(result['accuracy'], 63/203)
        self.assertEqual(result['support'][1], 10)
        self.assertEqual(result['recall'][1], 0)

    def test_missing_support_is_not_zero_recall(self):
        matrix = [[0]*6 for _ in range(6)]
        matrix[0][0] = 2
        result = classwise_from_confusion(matrix)
        self.assertEqual(result['recall'][0], 1)
        self.assertIsNone(result['recall'][1])

    def test_invalid_matrix_shape(self):
        for value in ([], [[1]], [[0]*5 for _ in range(6)]):
            with self.assertRaises(ValueError):
                classwise_from_confusion(value)

    def test_invalid_matrix_values(self):
        matrix = [[0]*6 for _ in range(6)]
        matrix[0][0] = -1
        with self.assertRaises(ValueError):
            classwise_from_confusion(matrix)
        matrix[0][0] = 1.2
        with self.assertRaises(ValueError):
            classwise_from_confusion(matrix)

    def test_development_comparisons_do_not_mix_protocols(self):
        source = {'fusion': {'case_count': 203, 'models': {
            'image': {'accuracy':45/203, 'balanced_accuracy':0.2, 'macro_f1':0.19},
            'fusion': {'accuracy':46/203, 'balanced_accuracy':0.21, 'macro_f1':0.20},
        }}}
        result = development_scores(source, 'fusion')
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0]['accuracy'], 45/203)
        with self.assertRaises(ValueError):
            development_scores(source, 'phase14')
        source['fusion']['case_count'] = 44
        with self.assertRaises(ValueError):
            development_scores(source, 'fusion')

    def test_invalid_development_score_rejected(self):
        source = {'preprocessing': {'case_count':203, 'models': {
            'direct_stretch': {'accuracy':1.4, 'balanced_accuracy':0.5, 'macro_f1':0.3}}}}
        with self.assertRaises(ValueError):
            development_scores(source, 'preprocessing')

    def test_batchnorm_validation_only(self):
        source = {'partition':'historical_validation', 'learned_weights_updated':False,
                  'conditions':{'original': {'accuracy':2/58, 'samples':58, 'majority_reference':17/58},
                                'recalibrated': {'accuracy':11/58, 'samples':58, 'majority_reference':17/58}}}
        result = validate_bn_conditions(source)
        self.assertEqual(result['n'],58)
        self.assertAlmostEqual(result['recalibrated'],11/58)
        source['partition'] = 'test'
        with self.assertRaises(ValueError):
            validate_bn_conditions(source)

    def test_batchnorm_cannot_be_described_as_retraining(self):
        source = {'partition':'historical_validation', 'learned_weights_updated':True,
                  'conditions':{'original': {'accuracy':2/58, 'samples':58, 'majority_reference':17/58},
                                'recalibrated': {'accuracy':11/58, 'samples':58, 'majority_reference':17/58}}}
        with self.assertRaises(ValueError):
            validate_bn_conditions(source)


if __name__ == '__main__':
    unittest.main()
