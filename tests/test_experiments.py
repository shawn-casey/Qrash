import numpy as np
import pandas as pd

from tailrisk.experiments import summarize
from tailrisk.figures import accuracy_table


def test_summary_uses_rmse_not_mean_absolute_error():
    frame = pd.DataFrame(
        {
            "scenario": ["case"] * 3,
            "method": ["MC"] * 3,
            "level": [0] * 3,
            "queries": [64] * 3,
            "error": [-0.1, 0.0, 0.2],
            "truth": [0.2] * 3,
            "seconds": [0.1, 0.2, 0.3],
        }
    )
    summary = summarize(frame)
    assert np.isclose(summary.iloc[0]["rmse"], np.sqrt(0.05 / 3))
    assert summary.iloc[0]["replicates"] == 3


def test_accuracy_table_requires_upper_error_band_to_reach_target():
    summary = pd.DataFrame(
        {
            "scenario": ["case"] * 3,
            "method": ["MC", "MC", "Sobol"],
            "queries": [64, 256, 64],
            "rmse_high": [0.02, 0.005, 0.03],
        }
    )
    table = accuracy_table(summary, 0.01).set_index("method")
    assert table.loc["MC", "first_observed_budget"] == 256
    assert table.loc["Sobol", "status"] == "Not reached in tested budgets"
