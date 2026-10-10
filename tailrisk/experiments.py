"""Reproducible experiment entry points. Results always include the actual circuit ledger."""

import hashlib
import importlib.metadata
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tailrisk import classical
from tailrisk.data import fetch_daily, holding_period_losses
from tailrisk.models import LossModel, fit_model, make_grid

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="ascii")


def environment():
    packages = ["qiskit", "qiskit-algorithms", "qiskit-aer", "numpy", "scipy", "pandas"]
    source_files = sorted((ROOT / "tailrisk").glob("*.py"))
    source_hash = hashlib.sha256()
    for file in source_files:
        source_hash.update(file.name.encode() + file.read_bytes())
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p) for p in packages},
        "core_source_sha256": source_hash.hexdigest(),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }


def prepare_data(output=ROOT / "data/fitted_models.json"):
    daily, provenance = fetch_daily(ROOT / "data/raw")
    selected = daily.loc["2000-01-01":"2025-12-31", ["BusEq", "Enrgy"]]
    losses = holding_period_losses(selected)
    models = {
        family: fit_model(losses.to_numpy(), family).to_dict() for family in ["gaussian", "student"]
    }
    payload = {
        "provenance": provenance,
        "industries": ["BusEq", "Enrgy"],
        "industry_labels": ["Business equipment", "Energy"],
        "start": str(selected.index.min().date()),
        "end": str(selected.index.max().date()),
        "daily_observations": len(selected),
        "ten_day_windows": len(losses),
        "holding_period_trading_days": 10,
        "calibration": "Full-period descriptive calibration; not an out-of-sample forecast",
        "overlapping_windows": True,
        "models": models,
    }
    write_json(output, payload)
    return payload


def load_model(family="student"):
    path = ROOT / "data/fitted_models.json"
    if not path.exists():
        raise FileNotFoundError("Run: python main.py prepare-data")
    payload = json.loads(path.read_text())
    return LossModel(**payload["models"][family])


def grid_metadata(grid):
    return {
        "bits_per_asset": grid.bits,
        "scenarios": len(grid.probabilities),
        "weight_units": list(grid.weight_units),
        "bound": grid.bound,
        "retained_mass": grid.retained_mass,
        "quadrature_error": grid.quadrature_error,
        "probabilities": grid.probabilities.tolist(),
        "losses": grid.losses.tolist(),
    }


def demo(family="student", threshold=0.10, bits=3, epsilon=0.01, seed=17, weight_units=(1, 1)):
    from tailrisk.quantum import resources, run_iqae, tail_oracle, verify_oracle

    model = load_model(family)
    grid = make_grid(model, bits=bits, weight_units=tuple(weight_units))
    oracle = tail_oracle(grid, threshold)
    exact = grid.tail(threshold)
    check = verify_oracle(oracle)
    if abs(check["objective_probability"] - exact) > 1e-9:
        raise RuntimeError("Oracle failed probability verification")
    estimate = run_iqae(oracle, epsilon=epsilon, seed=seed)
    budget = 2 ** int(np.floor(np.log2(estimate["queries"])))
    baseline = [
        classical.estimate(grid, threshold, method, budget, seed)
        for method in ["MC", "Sobol", "IS"]
    ]
    return {
        "environment": environment(),
        "family": family,
        "model": model.to_dict(),
        "grid": grid_metadata(grid),
        "threshold": threshold,
        "exact_grid_tail": exact,
        "continuous_tail": model.tail(threshold, grid.weights),
        "bounded_continuous_tail": model.tail(threshold, grid.weights, bound=grid.bound),
        "risk": {
            str(c): dict(zip(["var", "expected_shortfall"], grid.risk(c)))
            for c in [0.95, 0.99, 0.999]
        },
        "circuit_check": check,
        "resources": resources(oracle),
        "quantum": estimate,
        "classical": baseline,
    }


def benchmark(
    output,
    replicates=30,
    families=("gaussian", "student"),
    thresholds=(0.05, 0.10),
    bits=3,
    seed=20261010,
    levels=6,
):
    from tailrisk.quantum import resources, run_mlae, tail_oracle, verify_oracle

    if replicates < 2 or not 1 <= levels <= 7:
        raise ValueError("Use at least two replicates and 1-7 budget levels")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "replicates": replicates,
        "families": list(families),
        "thresholds": list(thresholds),
        "bits": bits,
        "seed": seed,
        "levels": levels,
        "mlae_shots_per_circuit": 64,
        "description": "Independent finite-shot circuit runs; fixed Grover schedules",
    }
    write_json(output / "manifest.json", {"config": config, "environment": environment()})
    rows, ledger, scenarios = [], [], []
    for f, family in enumerate(families):
        model = load_model(family)
        grid = make_grid(model, bits=bits)
        for t_index, threshold in enumerate(thresholds):
            oracle = tail_oracle(grid, threshold)
            truth = grid.tail(threshold)
            verification = verify_oracle(oracle)
            if not np.isclose(verification["objective_probability"], truth, atol=1e-10):
                raise RuntimeError("Circuit and grid probabilities disagree")
            scenario = f"{family}_{threshold:.3f}"
            scenarios.append(
                {
                    "scenario": scenario,
                    "model": model.to_dict(),
                    "grid": grid_metadata(grid),
                    "threshold": threshold,
                    "truth": truth,
                    "continuous_tail": model.tail(threshold),
                    "bounded_continuous_tail": model.tail(threshold, bound=grid.bound),
                    "resources": resources(oracle),
                    "verification": verification,
                }
            )
            for level in range(levels):
                schedule = [0] + [2**i for i in range(level)]
                quantum_budget = 64 * sum(2 * k + 1 for k in schedule)
                classical_budget = 2 ** int(np.floor(np.log2(quantum_budget)))
                print(
                    f"{scenario}: level {level + 1}/{levels}, "
                    f"QAE {quantum_budget} calls, classical {classical_budget} samples",
                    flush=True,
                )
                for rep in range(replicates):
                    run_seed = seed + f * 100000 + t_index * 10000 + level * 100 + rep
                    records = [
                        classical.estimate(
                            grid, threshold, method, classical_budget, run_seed + offset * 1000000
                        )
                        for offset, method in enumerate(["MC", "Sobol", "IS"])
                    ]
                    records.append(
                        run_mlae(oracle, tuple(schedule), shots=64, seed=run_seed + 3000000)
                    )
                    for record in records:
                        circuits = record.pop("circuits", None)
                        common = {
                            "scenario": scenario,
                            "family": family,
                            "threshold": threshold,
                            "truth": truth,
                            "level": level,
                            "replicate": rep,
                            "error": record["estimate"] - truth,
                            "absolute_error": abs(record["estimate"] - truth),
                        }
                        rows.append({**record, **common})
                        if circuits is not None:
                            ledger.append(
                                {
                                    **common,
                                    "seed": record["seed"],
                                    "queries": record["queries"],
                                    "circuits": circuits,
                                }
                            )
                # Save completed levels so a long run can be inspected without losing results.
                pd.DataFrame(rows).to_csv(output / "estimates.csv", index=False)
                write_json(output / "quantum_circuits.json", ledger)
                write_json(output / "scenarios.json", scenarios)
    summary = summarize(pd.DataFrame(rows), seed)
    summary.to_csv(output / "summary.csv", index=False)
    return summary


def summarize(frame, seed=17):
    rng = np.random.default_rng(seed)
    rows = []
    for keys, group in frame.groupby(["scenario", "method", "level"], sort=False):
        errors = group["error"].to_numpy()
        resampled = rng.choice(errors, size=(2000, len(errors)), replace=True)
        bootstrap_rmse = np.sqrt(np.mean(resampled**2, axis=1))
        rows.append(
            {
                "scenario": keys[0],
                "method": keys[1],
                "level": keys[2],
                "queries": int(group["queries"].iloc[0]),
                "replicates": len(group),
                "truth": float(group["truth"].iloc[0]),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "rmse_low": float(np.quantile(bootstrap_rmse, 0.025)),
                "rmse_high": float(np.quantile(bootstrap_rmse, 0.975)),
                "bias": float(np.mean(errors)),
                "median_seconds": float(group["seconds"].median()),
                "median_setup_seconds": float(group["setup_seconds"].median())
                if keys[1] == "IS"
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def studies(output=ROOT / "results/studies", seed=17):
    from tailrisk.quantum import (
        quantum_excess,
        quantum_var,
        resources,
        run_iqae,
        run_mlae,
        tail_oracle,
        verify_oracle,
    )

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = load_model("student")
    grid = make_grid(model)
    threshold = 0.10
    records = {"environment": environment(), "threshold": threshold}
    print("Checking oracle implementations", flush=True)
    records["oracles"] = {}
    for kind, arithmetic in [("direct", False), ("arithmetic", True)]:
        oracle = tail_oracle(grid, threshold, arithmetic)
        records["oracles"][kind] = {**verify_oracle(oracle), **resources(oracle)}
    print("Checking grid convergence", flush=True)
    continuous = model.tail(threshold)
    bounded = model.tail(threshold, bound=grid.bound)
    records["discretization"] = []
    for bits in range(2, 8):
        refined = make_grid(model, bits=bits)
        records["discretization"].append(
            {
                "bits": bits,
                "scenarios": len(refined.probabilities),
                "grid_tail": refined.tail(threshold),
                "continuous_tail": continuous,
                "bounded_continuous_tail": bounded,
                "retained_mass": refined.retained_mass,
                "quantization_error": refined.tail(threshold) - bounded,
                "truncation_effect": bounded - continuous,
            }
        )
    write_json(output / "studies.json", records)
    print("Running IQAE accuracy targets", flush=True)
    oracle = tail_oracle(grid, threshold)
    records["iqae"] = [
        run_iqae(oracle, epsilon=eps, seed=seed + i) for i, eps in enumerate([0.03, 0.015, 0.0075])
    ]
    print("Running interval-aware VaR search", flush=True)
    records["var"] = quantum_var(grid, confidence=0.95, seed=seed)
    quantile, exact_es = grid.risk(0.95)
    records["exact_var"] = quantile
    records["exact_es"] = exact_es
    if records["var"]["resolved"]:
        q = records["var"]["bracket"][0]
        records["es_quantile_source"] = "resolved quantum VaR"
    else:
        q = quantile
        records["es_quantile_source"] = "exact grid VaR; quantum search unresolved"
    print("Running positive-excess amplitude estimation", flush=True)
    records["es"] = quantum_excess(grid, q, epsilon=0.003, seed=seed)
    write_json(output / "studies.json", records)
    # Keep density-matrix noise circuits small and publish all replicates.
    print("Running depolarizing-noise study on a 2-bit-per-asset grid", flush=True)
    small_grid = make_grid(model, bits=2)
    noisy_oracle = tail_oracle(small_grid, threshold)
    noise_rows = []
    for noise in [0.0, 0.0001, 0.001, 0.005]:
        for schedule in [(0,), (0, 1), (0, 1, 2), (0, 1, 2, 4)]:
            for rep in range(10):
                result = run_mlae(
                    noisy_oracle,
                    schedule,
                    shots=128,
                    seed=seed + 100 * len(schedule) + rep,
                    noise=noise,
                )
                result.update(
                    {
                        "truth": small_grid.tail(threshold),
                        "replicate": rep,
                        "error": result["estimate"] - small_grid.tail(threshold),
                    }
                )
                noise_rows.append(result)
    records["noise"] = {
        "grid": grid_metadata(small_grid),
        "model": "Depolarizing channel after each transpiled CX only",
        "records": noise_rows,
    }
    write_json(output / "studies.json", records)
    return records
