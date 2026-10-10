"""Importance-weighted amplitude estimation and its classical controls."""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from tailrisk.classical import estimate, tilted_proposal
from tailrisk.experiments import ROOT, environment, load_model, summarize, write_json
from tailrisk.models import make_grid


def benchmark_importance(output=ROOT / "results/importance", seed=20261015):
    from tailrisk.quantum import importance_oracle, resources, run_mlae, verify_oracle

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    grid = make_grid(load_model("student"), bits=3)
    truth = grid.tail(0.10)
    start = perf_counter()
    oracle, metadata = importance_oracle(grid, 0.10)
    setup_seconds = perf_counter() - start
    verified = verify_oracle(oracle)
    if abs(verified["objective_probability"] * metadata["scale"] - truth) > 1e-10:
        raise RuntimeError("Importance circuit does not recover the original tail probability")
    proposal, _ = tilted_proposal(grid, 0.10)
    weighted = grid.probabilities * grid.tail_values(0.10) / proposal
    variance = float(proposal @ weighted**2 - truth**2)
    source = ROOT / "results/scaling/budget_estimates.csv"
    rows = pd.read_csv(source).to_dict("records")
    ledger = []
    write_json(
        output / "manifest.json",
        {
            "environment": environment(),
            "seed": seed,
            "replicates": 30,
            "levels": 6,
            "shots_per_circuit": 64,
            "schedule": "[0] + [2**i for i in range(level)], level=0,...,5",
            "model": "Student copula, 64 scenarios, 50/50 weights, 10% threshold",
            "truth": truth,
            "importance": metadata,
            "setup_seconds": setup_seconds,
            "verification": verified,
            "resources": resources(oracle),
            "classical_is_variance": variance,
            "reused": "MC, IS, MLAE from results/scaling/budget_estimates.csv",
            "new": "IS-MLAE, Sobol, IS-Sobol: 30 new independent trials per budget",
            "budget": "MC/IS use exactly quantum A/A-dagger calls; Sobol methods get the next power of two, favoring classical",
            "scope": "Ideal finite-shot Aer simulation; table-dependent preparation and payoff bounds; no runtime advantage",
            "development_pilots": [
                "One direct MLAE pilot at 33216 calls, seed 20261013; not a replicated benchmark",
                "One IS-MLAE pilot at 4352 calls, seed 20261014; benchmark uses fresh seeds",
            ],
        },
    )
    for level in range(6):
        schedule = [0] + [2**i for i in range(level)]
        budget = 64 * sum(2 * k + 1 for k in schedule)
        sobol_budget = 1 << (budget - 1).bit_length()
        print(f"Importance study: N={budget}, Sobol N={sobol_budget}", flush=True)
        for rep in range(30):
            records = [
                estimate(grid, 0.10, method, sobol_budget, seed + level * 1000 + i * 100 + rep)
                for i, method in enumerate(["Sobol", "IS-Sobol"])
            ]
            quantum = run_mlae(oracle, schedule, 64, seed + level * 1000 + 200 + rep)
            circuits = quantum.pop("circuits")
            quantum["amplitude_estimate"] = quantum["estimate"]
            quantum["estimate"] *= metadata["scale"]
            quantum["method"] = "IS-MLAE"
            quantum["payoff_scale"] = metadata["scale"]
            records.append(quantum)
            ledger.append(
                {
                    "level": level,
                    "replicate": rep,
                    "seed": quantum["seed"],
                    "queries": quantum["queries"],
                    "circuits": circuits,
                }
            )
            for record in records:
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
        pd.DataFrame(rows).to_csv(output / "estimates.csv", index=False)
        write_json(output / "circuits.json", ledger)
    summary = summarize(pd.DataFrame(rows), seed)
    summary["exact_is_rmse"] = np.sqrt(variance / summary.queries)
    summary.to_csv(output / "summary.csv", index=False)
    print(
        summary[["method", "level", "queries", "rmse", "rmse_low", "rmse_high"]].to_string(
            index=False
        )
    )
    return summary
