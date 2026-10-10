import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from tailrisk.data import holding_period_losses, parse_daily
from tailrisk.models import LossModel, make_grid


@pytest.fixture
def model():
    return LossModel("gaussian", [0.0, 0.0], [0.05, 0.06], 0.4)


def test_daily_parser_selects_value_weighted_table_and_handles_missing():
    text = """A source header
  Average Value Weighted Returns -- Daily
,BusEq,Enrgy
20000103,1.0,2.0
20000104,-99.99,-1.0

  Average Equal Weighted Returns -- Daily
,BusEq,Enrgy
20000103,55,55
"""
    frame = parse_daily(text)
    assert len(frame) == 2
    assert frame.iloc[0, 0] == 0.01
    assert np.isnan(frame.iloc[1, 0])
    assert frame.iloc[1, 1] == -0.01


def test_compounding_and_missing_windows():
    daily = pd.DataFrame({"a": [0.1, 0.1, np.nan, 0.2, 0.2]})
    loss = holding_period_losses(daily, days=2)
    assert list(loss.index) == [1, 4]
    np.testing.assert_allclose(loss["a"], [-0.21, -0.44])


def test_gaussian_tail_is_analytic(model):
    sigma = np.sqrt((0.05**2 + 0.06**2 + 2 * 0.4 * 0.05 * 0.06) / 4)
    assert model.tail(0.1) == pytest.approx(norm.sf(0.1 / sigma))


def test_independent_grid_matches_product_of_marginal_cell_masses():
    model = LossModel("gaussian", [0.01, -0.02], [0.05, 0.08], 0.0)
    grid = make_grid(model, bits=2)
    edges = np.linspace(-0.25, 0.25, 5)
    masses = [np.diff(model.marginal(i).cdf(edges)) for i in range(2)]
    expected = np.outer(*masses)
    assert grid.retained_mass == pytest.approx(expected.sum(), abs=1e-10)
    np.testing.assert_allclose(
        grid.probabilities, (expected / expected.sum()).ravel(order="F"), atol=1e-10
    )


def test_loss_encoding_and_boundary_predicate(model):
    grid = make_grid(model, bits=3, weight_units=(2, 1))
    m = len(grid.nodes)
    for j in range(m):
        for i in range(m):
            assert grid.losses[i + m * j] == pytest.approx((2 * grid.nodes[i] + grid.nodes[j]) / 3)
    for threshold in np.unique(grid.losses):
        np.testing.assert_array_equal(grid.tail_values(threshold), grid.losses >= threshold)
    assert grid.tail(-1) == pytest.approx(1)
    assert grid.tail(1) == 0


def test_expected_shortfall_accounts_for_mass_at_quantile(model):
    grid = make_grid(model, bits=1)
    grid.probabilities = np.array([0.9, 0.06, 0.03, 0.01])
    grid.losses = np.array([0.0, 1.0, 2.0, 3.0])
    quantile, es = grid.risk(0.95)
    assert quantile == 1.0
    assert es == pytest.approx((0.01 * 1 + 0.03 * 2 + 0.01 * 3) / 0.05)
    assert es != pytest.approx((grid.probabilities[1:] @ grid.losses[1:]) / 0.1)


def test_grid_converges_in_tail(model):
    grid = make_grid(model, bits=7)
    assert abs(grid.tail(0.1) - model.tail(0.1, bound=0.25)) < 0.001


def test_student_grid_is_normalized():
    model = LossModel("student", [0.0, 0.0], [0.04, 0.05], 0.4, [4.0, 5.0])
    grid = make_grid(model)
    assert grid.probabilities.sum() == pytest.approx(1)
    assert 0.98 < grid.retained_mass < 1
    assert 0 < model.tail(0.1) < 0.1
