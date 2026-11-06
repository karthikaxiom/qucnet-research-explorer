"""Numerical checks for educational fixed four-qubit patch simulator."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.quantum.simulator import _cnot, _single_wire, fixed_weights, simulate_patch


class QuantumSimulatorTests(unittest.TestCase):
    def test_weights_repeat_and_shape(self):
        a = fixed_weights()
        self.assertEqual(a.shape, (2, 4))
        self.assertTrue(np.array_equal(a, fixed_weights()))
        self.assertAlmostEqual(a[0,0], 2.353304971691044, places=12)

    def test_range_and_stability(self):
        for patch in ([0,0,0,0], [1,1,1,1], [0.2,0.4,0.6,0.8]):
            out = simulate_patch(patch)
            self.assertEqual(out.shape, (4,))
            self.assertTrue(np.isfinite(out).all())
            self.assertTrue((np.abs(out) <= 1 + 1e-10).all())
            self.assertTrue(np.array_equal(out, simulate_patch(patch)))

    def test_zero_rotations_identity_on_zero_patch(self):
        result = simulate_patch([0,0,0,0], weights=np.zeros((2,4)))
        np.testing.assert_allclose(result, [1,1,1,1], rtol=0, atol=1e-12)

    def test_invalid_inputs_rejected(self):
        for data in ([0,0,0], [0,0,float('nan'),0], [-0.2,0,0,0], [1.1,0,0,0]):
            with self.assertRaises(ValueError):
                simulate_patch(data)
        with self.assertRaises(ValueError):
            simulate_patch([0.1,0.2,0.3,0.4], weights=np.ones((3,4)))

    def test_cnot_swaps_correct_basis_states(self):
        state = np.zeros(16, complex)
        state[8] = 1 # |1000>: control wire 0=1
        after = _cnot(state,0,1)
        self.assertAlmostEqual(abs(after[12]), 1.0)  # |1100>
        self.assertAlmostEqual(np.sum(abs(after)**2), 1.0)

    def test_single_wire_x_gate_on_each_wire(self):
        xgate = np.array([[0,1],[1,0]],complex)
        for wire in range(4):
            state = np.zeros(16,complex)
            state[0] = 1
            out = _single_wire(state,xgate,wire)
            self.assertAlmostEqual(abs(out[1 << (3-wire)]),1.0)


if __name__ == '__main__':
    unittest.main()
