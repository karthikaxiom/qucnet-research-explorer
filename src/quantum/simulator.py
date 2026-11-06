"""Reference-gate-sequence simulator for educational 2x2 synthetic patches.

Matches the *operations and fixed-seed weights* in the frozen research source:
05_Source_Code/utilities/quantum_utils.py

This module intentionally does not import PennyLane, TensorFlow or medical data.
It does not classify images, train models, or claim quantum advantage.
"""
from __future__ import annotations

import numpy as np

N_QUBITS = 4
N_LAYERS = 2
SEED = 42


def fixed_weights() -> np.ndarray:
    """Return the archived circuit's seeded RY angles (2 layers x 4 wires)."""
    return np.random.RandomState(SEED).uniform(0, 2 * np.pi, (N_LAYERS, N_QUBITS))


def _rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def _ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _single_wire(state: np.ndarray, gate: np.ndarray, wire: int) -> np.ndarray:
    if wire not in range(N_QUBITS):
        raise ValueError("Wire index must be 0..3")
    tensor = state.reshape((2,) * N_QUBITS)
    front = np.moveaxis(tensor, wire, 0)
    transformed = (gate @ front.reshape(2, -1)).reshape(front.shape)
    return np.moveaxis(transformed, 0, wire).reshape(-1)


def _cnot(state: np.ndarray, control: int, target: int) -> np.ndarray:
    if control == target or control not in range(4) or target not in range(4):
        raise ValueError("Invalid CNOT wire pair")
    transformed = np.empty_like(state)
    control_bit = 1 << (N_QUBITS - 1 - control)
    target_bit = 1 << (N_QUBITS - 1 - target)
    for basis_index, amplitude in enumerate(state):
        new_index = basis_index ^ target_bit if basis_index & control_bit else basis_index
        transformed[new_index] = amplitude
    return transformed


def simulate_patch(pixels: list[float] | np.ndarray, *, weights: np.ndarray | None = None) -> np.ndarray:
    """Return four Pauli-Z expectations for four normalized synthetic values.

    Pixel intensity is used *as RX rotation angle in radians*, matching the
    archived code; there is no additional factor of pi. The resulting values
    are expectations in [-1, 1], not class probabilities.
    """
    patch = np.asarray(pixels, dtype=float)
    if patch.shape != (4,) or not np.isfinite(patch).all():
        raise ValueError("Expected exactly four finite synthetic pixel values")
    if (patch < 0).any() or (patch > 1).any():
        raise ValueError("Synthetic normalized pixels must lie within [0, 1]")

    angles = fixed_weights() if weights is None else np.asarray(weights, dtype=float)
    if angles.shape != (N_LAYERS, N_QUBITS) or not np.isfinite(angles).all():
        raise ValueError("Expected a finite (2, 4) fixed-rotation matrix")

    state = np.zeros(2**N_QUBITS, dtype=np.complex128)
    state[0] = 1.0

    for wire in range(N_QUBITS):
        state = _single_wire(state, _rx(float(patch[wire])), wire)

    for layer in range(N_LAYERS):
        for wire in range(N_QUBITS):
            state = _single_wire(state, _ry(float(angles[layer, wire])), wire)
        for control, target in ((0, 1), (1, 2), (2, 3), (3, 0)):
            state = _cnot(state, control, target)

    prob = np.abs(state) ** 2
    if not np.isclose(prob.sum(), 1.0, atol=1e-10):
        raise ArithmeticError("Simulated quantum state lost normalization")
    values = []
    for wire in range(N_QUBITS):
        mask = 1 << (N_QUBITS - 1 - wire)
        signs = np.where(np.arange(16) & mask, -1.0, 1.0)
        values.append(float(np.dot(prob, signs)))
    return np.asarray(values, dtype=float)
