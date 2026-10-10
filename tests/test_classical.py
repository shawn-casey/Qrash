import numpy as np
import pytest

from tailrisk.classical import discrete_transform, estimate, sobol_indices, tilted_proposal
from tailrisk.models import LossModel, make_grid


@pytest.fixture
def grid():
    return make_grid(LossModel("gaussian", [0, 0], [0.05, 0.06], 0.4))


def test_inverse_transform_boundaries():
    np.testing.assert_array_equal(
        discrete_transform([0.2, 0.3, 0.5], [0, 0.199, 0.2, 0.5, 0.999]), [0, 0, 1, 2, 2]
    )


def test_importance_weights_recover_exact_target(grid):
    proposal, _ = tilted_proposal(grid, 0.1)
    values = grid.tail_values(0.1)
    positive = grid.probabilities > 0
    assert np.all(proposal[positive] > 0)
    weighted = proposal[positive] * grid.probabilities[positive] / proposal[positive]
    assert weighted @ values[positive] == pytest.approx(grid.tail(0.1), abs=1e-14)


def test_conditional_sobol_recovers_joint_distribution(grid):
    indices = sobol_indices(grid, 65536, seed=15)
    frequency = np.bincount(indices, minlength=len(grid.probabilities)) / len(indices)
    assert np.max(np.abs(frequency - grid.probabilities)) < 0.0003


@pytest.mark.parametrize("method", ["MC", "Sobol", "IS"])
def test_classical_estimates_converge(grid, method):
    result = estimate(grid, 0.1, method, 65536, 17)
    assert abs(result["estimate"] - grid.tail(0.1)) < 0.003
    assert result["queries"] == 65536


def test_sobol_rejects_non_power_of_two(grid):
    with pytest.raises(ValueError):
        sobol_indices(grid, 100, 17)
