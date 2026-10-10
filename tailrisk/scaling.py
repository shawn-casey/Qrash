"""Separate estimator budget N from scenario count S in measured comparisons."""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from tailrisk.classical import estimate
from tailrisk.experiments import ROOT, environment, load_model, summarize, write_json
from tailrisk.models import make_grid


def matched_budget(output=ROOT / "results/scaling", seed=20261011):
    """Reuse recorded MLAE trials and measure MC/IS at exactly the same N."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "results/benchmark/estimates.csv"
    old = pd.read_csv(source)
    quantum = old[(old.scenario == "student_0.100") & (old.method == "MLAE")].copy()
    grid = make_grid(load_model("student"), bits=3)
    truth = grid.tail(0.10)
    rows = quantum.to_dict("records")
    for level, group in quantum.groupby("level"):
        budget = int(group.queries.iloc[0])
        for method_index, method in enumerate(["MC", "IS"]):
            for rep in range(len(group)):
                record = estimate(
                    grid, 0.10, method, budget, seed + level * 1000 + method_index * 100 + rep
                )
                rows.append(
                    {
                        **record,
                        "scenario": "student_0.100",
                        "level": level,
                        "replicate": rep,
                        "truth": truth,
                        "error": record["estimate"] - truth,
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "budget_estimates.csv", index=False)
    summary = summarize(frame, seed)
    summary["mc_theoretical_rmse"] = np.sqrt(truth * (1 - truth) / summary.queries)
    summary.to_csv(output / "budget_summary.csv", index=False)
    slopes = {
        method: float(np.polyfit(np.log(group.queries), np.log(group.rmse), 1)[0])
        for method, group in summary.groupby("method")
    }
    write_json(
        output / "budget_manifest.json",
        {
            "environment": environment(),
            "seed": seed,
            "quantum_source": "results/benchmark/estimates.csv: student_0.100 MLAE rows",
            "quantum_reused_not_rerun": True,
            "N": "MC/IS samples or MLAE A/A-dagger calls, matched numerically, not equal time",
            "S": 64,
            "threshold": 0.10,
            "truth": truth,
            "replicates": 30,
            "empirical_log_log_slopes": slopes,
            "slope_interpretation": "Descriptive fit to six measured points; not an asymptotic proof",
        },
    )
    return summary


def scenario_scaling(output=ROOT / "results/scaling", bits=(2, 3, 4), replicates=30, seed=20261012):
    """Hold query budget fixed while refining the same two-asset model's grid."""
    from tailrisk.quantum import resources, run_mlae, tail_oracle, verify_oracle

    if replicates < 2:
        raise ValueError("At least two independent replicates are required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    schedule, shots = (0, 1, 2), 64
    budget = shots * sum(2 * k + 1 for k in schedule)
    rows, resources_rows, ledger = [], [], []
    write_json(
        output / "scenario_manifest.json",
        {
            "environment": environment(),
            "seed": seed,
            "bits": list(bits),
            "replicates": replicates,
            "N": budget,
            "schedule": list(schedule),
            "shots": shots,
            "S": "Number of joint grid scenarios = 2**(2 * bits_per_asset)",
            "model": "Student-t marginals and Gaussian copula; 50/50 weights; 10% loss threshold",
            "timing": "Estimator wall time includes first-use compilation, cached for later replicates; grid and oracle construction reported separately",
            "comparison": "Each grid has its own exact truth. Refinement changes discretization, not the number of assets.",
        },
    )
    for b in bits:
        start = perf_counter()
        grid = make_grid(load_model("student"), bits=b)
        grid_seconds = perf_counter() - start
        start = perf_counter()
        oracle = tail_oracle(grid, 0.10)
        build_seconds = perf_counter() - start
        verified = verify_oracle(oracle)
        truth = grid.tail(0.10)
        if abs(verified["objective_probability"] - truth) > 1e-10:
            raise RuntimeError("Oracle probability differs from the exact grid truth")
        exact_times = []
        for _ in range(100):
            start = perf_counter()
            grid.tail(0.10)
            exact_times.append(perf_counter() - start)
        s = len(grid.probabilities)
        resources_rows.append(
            {
                "S": s,
                "bits": b,
                "truth": truth,
                "grid_seconds": grid_seconds,
                "oracle_build_seconds": build_seconds,
                "exact_median_seconds": float(np.median(exact_times)),
                **resources(oracle),
                **verified,
            }
        )
        print(f"Scenario scaling: S={s}, N={budget}, {replicates} replicates", flush=True)
        for rep in range(replicates):
            records = [
                estimate(grid, 0.10, method, budget, seed + b * 10000 + i * 1000 + rep)
                for i, method in enumerate(["MC", "IS"])
            ]
            records.append(run_mlae(oracle, schedule, shots, seed + b * 10000 + 2000 + rep))
            for record in records:
                circuits = record.pop("circuits", None)
                row = {
                    **record,
                    "scenario": f"S={s}",
                    "S": s,
                    "level": 0,
                    "replicate": rep,
                    "truth": truth,
                    "error": record["estimate"] - truth,
                }
                rows.append(row)
                if circuits is not None:
                    ledger.append(
                        {
                            "S": s,
                            "replicate": rep,
                            "queries": record["queries"],
                            "seed": record["seed"],
                            "circuits": circuits,
                        }
                    )
        pd.DataFrame(rows).to_csv(output / "scenario_estimates.csv", index=False)
        pd.DataFrame(resources_rows).to_csv(output / "scenario_resources.csv", index=False)
        write_json(output / "scenario_circuits.json", ledger)
    result = summarize(pd.DataFrame(rows), seed)
    result["S"] = result.scenario.str.split("=").str[-1].astype(int)
    result.to_csv(output / "scenario_summary.csv", index=False)
    return result
