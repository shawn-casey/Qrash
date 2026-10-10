"""Audit the weighted quantum experiment against its raw trials and circuit ledger."""

import json

import numpy as np
import pandas as pd

from tailrisk.experiments import ROOT


def test_importance_trial_scaling_and_actual_query_budgets():
    folder = ROOT / "results/importance"
    frame = pd.read_csv(folder / "estimates.csv")
    summary = pd.read_csv(folder / "summary.csv")
    manifest = json.loads((folder / "manifest.json").read_text())
    ledger = json.loads((folder / "circuits.json").read_text())
    assert len(frame) == 6 * 6 * 30
    assert len(ledger) == 180
    for record in ledger:
        selected = frame[
            (frame.method == "IS-MLAE")
            & (frame.level == record["level"])
            & (frame.replicate == record["replicate"])
        ].iloc[0]
        assert selected.queries == sum(
            c["shots"] * (2 * c["power"] + 1) for c in record["circuits"]
        )
        assert selected.seed == record["seed"]
    quantum = frame[frame.method == "IS-MLAE"]
    np.testing.assert_allclose(
        quantum.estimate, quantum.amplitude_estimate * manifest["importance"]["scale"]
    )
    for _, group in frame.groupby("level"):
        matched = group[group.method.isin(["MC", "IS", "MLAE", "IS-MLAE"])]
        assert matched.queries.nunique() == 1
        for method, trials in group.groupby("method"):
            assert len(trials) == 30
            if method in {"Sobol", "IS-Sobol"}:
                assert trials.queries.iloc[0] >= matched.queries.iloc[0]
    for row in summary.itertuples():
        trials = frame[(frame.method == row.method) & (frame.level == row.level)]
        assert np.isclose(row.rmse, np.sqrt(np.mean((trials.estimate - trials.truth) ** 2)))
