"""Figures generated from measured experiment records."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from tailrisk.experiments import ROOT

COLORS = {"MC": "#63758c", "Sobol": "#16806a", "IS": "#dd8833", "MLAE": "#7754ba"}


def accuracy_table(summary, target):
    rows = []
    for (scenario, method), group in summary.groupby(["scenario", "method"]):
        reached = group[group["rmse_high"] <= target].sort_values("queries")
        rows.append(
            {
                "scenario": scenario,
                "method": method,
                "target_rmse": target,
                "first_observed_budget": int(reached.iloc[0]["queries"]) if len(reached) else None,
                "status": "Reached" if len(reached) else "Not reached in tested budgets",
            }
        )
    return pd.DataFrame(rows)


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.labelcolor": "#253449",
            "text.color": "#253449",
            "savefig.bbox": "tight",
        }
    )


def distribution_figure(grid, threshold):
    style()
    support, inverse = np.unique(grid.losses, return_inverse=True)
    probabilities = np.bincount(inverse, weights=grid.probabilities)
    fig, ax = plt.subplots(figsize=(9, 3.8), layout="constrained")
    width = np.min(np.diff(support)) * 0.8
    ax.bar(
        support,
        probabilities,
        width=width,
        color=["#dd8833" if loss >= threshold else "#496ca6" for loss in support],
    )
    ax.axvline(threshold, color="#b75820", linestyle="--", label="Loss threshold")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(
        xlabel="10-trading-day portfolio loss",
        ylabel="Probability mass",
        title="Where does the tail begin?",
    )
    ax.legend(frameon=False)
    return fig


def benchmark_figure(summary, scenario=None):
    style()
    scenarios = [scenario] if scenario else summary["scenario"].unique()
    columns = 1 if len(scenarios) == 1 else 2
    rows = int(np.ceil(len(scenarios) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(7 * columns, 4.7 * rows), squeeze=False, layout="constrained"
    )
    for ax, name in zip(axes.flat, scenarios):
        data = summary[summary["scenario"] == name]
        for method, color in COLORS.items():
            group = data[data["method"] == method].sort_values("queries")
            ax.plot(group["queries"], group["rmse"], "o-", label=method, color=color)
            ax.fill_between(
                group["queries"], group["rmse_low"], group["rmse_high"], color=color, alpha=0.12
            )
        family, threshold = name.rsplit("_", 1)
        truth = float(data["truth"].iloc[0])
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="Samples (classical) / A calls (MLAE)",
            ylabel="Tail-probability RMSE",
            title=f"{family.title()} | loss >= {float(threshold):.0%} | grid tail {truth:.2%}",
        )
        ax.grid(alpha=0.18, which="both")
        ax.legend(frameon=False, ncols=2)
    for ax in list(axes.flat)[len(scenarios) :]:
        ax.set_visible(False)
    return fig


def noise_figure(study):
    style()
    data = pd.DataFrame(study["noise"]["records"])
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for noise, group in data.groupby("noise"):
        aggregate = group.groupby("queries")["error"].apply(lambda x: np.sqrt(np.mean(x**2)))
        ax.plot(aggregate.index, aggregate.values, "o-", label=f"CX error {noise:g}")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="A / A-dagger calls",
        ylabel="MLAE RMSE",
        title="Depth stops helping when gate noise dominates",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.18, which="both")
    return fig


def grid_figure(study):
    style()
    data = pd.DataFrame(study["discretization"])
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot(data["bits"], data["grid_tail"], "o-", color="#7754ba", label="Exact grid tail")
    ax.axhline(
        data["bounded_continuous_tail"].iloc[0],
        color="#16806a",
        linestyle="--",
        label="Bounded continuous model",
    )
    ax.axhline(
        data["continuous_tail"].iloc[0],
        color="#63758c",
        linestyle=":",
        label="Full continuous model",
    )
    ax.set(
        xlabel="Qubits per asset (grid resolution)",
        ylabel="Tail probability",
        title=f"Grid error is separate from estimation error | loss >= {study['threshold']:.0%}",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)
    return fig


def render_all():
    output = ROOT / "results/figures"
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(ROOT / "results/benchmark/summary.csv")
    pd.concat(
        [accuracy_table(summary, target) for target in [0.01, 0.005, 0.002, 0.001]],
        ignore_index=True,
    ).to_csv(ROOT / "results/benchmark/accuracy_cost.csv", index=False)
    fig = benchmark_figure(summary)
    fig.savefig(output / "benchmark.png", dpi=180)
    plt.close(fig)
    study_path = ROOT / "results/studies/studies.json"
    if study_path.exists():
        study = json.loads(study_path.read_text())
        for name, function in [("grid_convergence", grid_figure), ("noise", noise_figure)]:
            if name == "noise" and "noise" not in study:
                continue
            fig = function(study)
            fig.savefig(output / f"{name}.png", dpi=180)
            plt.close(fig)
    print(f"Figures saved to {output}")
