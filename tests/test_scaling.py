"""Audit the saved scaling evidence independently of the plotting code."""

import json

import numpy as np
import pandas as pd

from tailrisk.experiments import ROOT


def test_matched_budget_has_equal_n_and_reuses_original_quantum_trials():
    frame = pd.read_csv(ROOT / "results/scaling/budget_estimates.csv")
    original = pd.read_csv(ROOT / "results/benchmark/estimates.csv")
    original = original[(original.scenario == "student_0.100") & (original.method == "MLAE")]
    for _, group in frame.groupby("level"):
        assert group.queries.nunique() == 1
        assert group.groupby("method").size().to_dict() == {"IS": 30, "MC": 30, "MLAE": 30}
    quantum = frame[frame.method == "MLAE"].sort_values(["level", "replicate"])
    original = original.sort_values(["level", "replicate"])
    # CSV decimal round-trips can change the least significant floating-point bit.
    np.testing.assert_allclose(quantum.estimate, original.estimate, rtol=1e-14, atol=0)
    np.testing.assert_array_equal(quantum.seed, original.seed)


def test_scaling_summary_matches_raw_errors_and_mc_formula():
    for prefix in ["budget", "scenario"]:
        frame = pd.read_csv(ROOT / f"results/scaling/{prefix}_estimates.csv")
        summary = pd.read_csv(ROOT / f"results/scaling/{prefix}_summary.csv")
        for row in summary.itertuples():
            selected = frame[
                (frame.scenario == row.scenario)
                & (frame.method == row.method)
                & (frame.level == row.level)
            ]
            assert np.isclose(row.rmse, np.sqrt(np.mean((selected.estimate - selected.truth) ** 2)))
            assert row.rmse_low <= row.rmse <= row.rmse_high
            if prefix == "budget":
                assert np.isclose(
                    row.mc_theoretical_rmse, np.sqrt(row.truth * (1 - row.truth) / row.queries)
                )


def test_scenario_study_reconciles_actual_circuit_queries():
    records = json.loads((ROOT / "results/scaling/scenario_circuits.json").read_text())
    assert len(records) == 90
    for row in records:
        assert row["queries"] == 576
        assert row["queries"] == sum(c["shots"] * (2 * c["power"] + 1) for c in row["circuits"])
    resources = pd.read_csv(ROOT / "results/scaling/scenario_resources.csv")
    assert resources.S.tolist() == [16, 64, 256]
    np.testing.assert_allclose(resources.truth, resources.objective_probability, atol=1e-10)
