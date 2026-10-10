import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from tailrisk.models import LossModel, make_grid
from tailrisk.quantum import (
    RecordedIQAE,
    direct_oracle,
    quantum_var,
    run_iqae,
    run_mlae,
    tail_oracle,
    verify_oracle,
)


@pytest.fixture
def grid():
    return make_grid(LossModel("gaussian", [0, 0], [0.05, 0.06], 0.4), bits=2)


@pytest.mark.parametrize("arithmetic", [False, True])
@pytest.mark.parametrize("threshold", [-1.0, 0.0625, 0.08, 1.0])
def test_tail_oracle_matches_exact_probability_and_cleans_workspace(grid, arithmetic, threshold):
    oracle = tail_oracle(grid, threshold, arithmetic)
    check = verify_oracle(oracle)
    assert check["objective_probability"] == pytest.approx(grid.tail(threshold), abs=1e-10)
    assert abs(check["workspace_nonzero_probability"]) < 1e-10


def test_nonuniform_payoff_rotation_and_endianness(grid):
    values = np.linspace(0, 1, len(grid.probabilities)) ** 2
    oracle = direct_oracle(grid, values)
    assert verify_oracle(oracle)["objective_probability"] == pytest.approx(
        grid.probabilities @ values, abs=1e-10
    )


def test_arithmetic_unequal_weights():
    grid = make_grid(LossModel("gaussian", [0, 0], [0.05, 0.06], 0.4), bits=1, weight_units=(2, 1))
    oracle = tail_oracle(grid, 0.04, arithmetic=True)
    assert verify_oracle(oracle)["objective_probability"] == pytest.approx(
        grid.tail(0.04), abs=1e-10
    )


@pytest.mark.parametrize("arithmetic", [False, True])
def test_grover_amplification_matches_rotation_law(grid, arithmetic):
    oracle = tail_oracle(grid, 0.08, arithmetic=arithmetic)
    theta = np.arcsin(np.sqrt(grid.tail(0.08)))
    for k in [0, 1, 2, 4]:
        circuit = RecordedIQAE(0.03, 0.05).construct_circuit(oracle.problem, k)
        measured = Statevector.from_instruction(circuit).probabilities([oracle.objective])[1]
        assert measured == pytest.approx(np.sin((2 * k + 1) * theta) ** 2, abs=1e-9)


def test_iqae_finite_shots_and_accounting(grid):
    result = run_iqae(tail_oracle(grid, 0.08), epsilon=0.02, seed=17)
    lo, hi = result["confidence_interval"]
    assert lo <= grid.tail(0.08) <= hi
    assert (hi - lo) / 2 <= 0.02
    assert result["queries"] == 2 * result["grover_calls"] + result["shots"]
    assert sum(row["shots"] for row in result["circuits"]) == result["shots"]


def test_mlae_fixed_schedule_counts_k_zero(grid):
    result = run_mlae(tail_oracle(grid, 0.08), schedule=(0, 1, 2), shots=64, seed=17)
    assert result["queries"] == 64 * (1 + 3 + 5)
    assert result["shots"] == 192
    assert result["circuits"][0]["a_calls"] == 64
    assert abs(result["estimate"] - grid.tail(0.08)) < 0.025


def test_var_does_not_claim_resolution_when_cdf_interval_overlaps(monkeypatch, grid):
    def uncertain(*args, **kwargs):
        return {"confidence_interval": [0.0, 1.0], "queries": 10}

    monkeypatch.setattr("tailrisk.quantum.run_iqae", uncertain)
    result = quantum_var(grid, confidence=0.95)
    assert not result["resolved"]
    assert result["bracket"][0] <= grid.risk(0.95)[0] <= result["bracket"][1]
    assert result["queries"] == 30
