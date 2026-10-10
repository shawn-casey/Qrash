"""Single-run timing breakdown: python -m scripts.profile_pipeline."""

from time import perf_counter

from tailrisk.experiments import ROOT, environment, load_model, write_json
from tailrisk.models import make_grid
from tailrisk.quantum import run_iqae, tail_oracle


def timed(function, *args):
    start = perf_counter()
    result = function(*args)
    return result, perf_counter() - start


def main():
    records = []
    for family in ["gaussian", "student"]:
        model, loading = timed(load_model, family)
        grid, grid_time = timed(make_grid, model)
        truth, enumeration = timed(grid.tail, 0.1)
        continuous, reference = timed(model.tail, 0.1)
        oracle, construction = timed(tail_oracle, grid, 0.1)
        result = run_iqae(oracle, epsilon=0.01, seed=17)
        records.append(
            {
                "family": family,
                "threshold": 0.1,
                "model_loading_seconds": loading,
                "grid_construction_seconds": grid_time,
                "exact_grid_tail_seconds": enumeration,
                "continuous_reference_seconds": reference,
                "oracle_construction_seconds": construction,
                "iqae_seconds": result["seconds"],
                "exact_grid_tail": truth,
                "continuous_tail": continuous,
                "iqae_queries": result["queries"],
            }
        )
    write_json(
        ROOT / "results/timing.json",
        {
            "environment": environment(),
            "description": "Single cold run per model; descriptive timings, not a hardware prediction. "
            "Data download and model fitting excluded.",
            "records": records,
        },
    )


if __name__ == "__main__":
    main()
