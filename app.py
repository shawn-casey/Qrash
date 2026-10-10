"""Local presentation app. Start with: streamlit run app.py"""

import json
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from tailrisk.experiments import ROOT, load_model
from tailrisk.figures import (
    accuracy_table,
    benchmark_figure,
    budget_scaling_figure,
    distribution_figure,
    grid_figure,
    noise_figure,
    scenario_scaling_figure,
)
from tailrisk.models import make_grid

st.set_page_config(page_title="TailRisk Lab", page_icon="Q", layout="wide")


@st.cache_data
def model_grid(family, bits, weight_units):
    model = load_model(family)
    return model, make_grid(model, bits=bits, weight_units=weight_units)


@st.cache_data
def continuous_references(family, threshold, weights, bound):
    model = load_model(family)
    return model.tail(threshold, weights), model.tail(threshold, weights, bound)


def show_figure(fig):
    st.pyplot(fig, width="stretch")
    plt.close(fig)


st.caption("GT QISKIT FALL FEST 2026  /  QUANTUM IMPACT")
st.title("TailRisk Lab")
st.markdown("**How much computation does a rare loss deserve?**")
st.write(
    "Estimate portfolio tail risk with quantum amplitude estimation, then put it beside "
    "Monte Carlo, scrambled Sobol, and importance sampling."
)

with st.sidebar:
    st.header("Portfolio")
    family = st.selectbox(
        "Return model",
        ["student", "gaussian"],
        format_func=lambda x: (
            "Student-t + Gaussian copula" if x == "student" else "Gaussian (analytic control)"
        ),
    )
    weights_label = st.selectbox("Business equipment / Energy", ["50 / 50", "67 / 33", "33 / 67"])
    weight_units = {"50 / 50": (1, 1), "67 / 33": (2, 1), "33 / 67": (1, 2)}[weights_label]
    threshold = st.slider("10-day loss threshold (%)", 2, 18, 10, 1) / 100
    bits = st.select_slider("Qubits per asset", options=[2, 3, 4], value=3)
    confidence = st.selectbox(
        "Risk confidence", [0.95, 0.99, 0.999], format_func=lambda c: f"{c:.1%}"
    )
    st.divider()
    st.caption(
        "Kenneth French 12 industry portfolios. Value-weighted daily returns, "
        "2000-2025. Ten-trading-day holding period; fixed starting weights."
    )
    st.caption("Descriptive model calibration, not an out-of-sample forecast.")

model, grid = model_grid(family, bits, weight_units)
truth = grid.tail(threshold)
quantile, expected_shortfall = grid.risk(confidence)
portfolio, benchmark_tab, circuit_tab, method_tab = st.tabs(
    ["Portfolio & live estimate", "Estimator benchmark", "Circuits & limits", "How it works"]
)

with portfolio:
    metrics = st.columns(4)
    metrics[0].metric("Exact grid tail probability", f"{truth:.3%}")
    metrics[1].metric(f"Grid VaR at {confidence:.1%}", f"{quantile:.2%}")
    metrics[2].metric("Grid expected shortfall", f"{expected_shortfall:.2%}")
    metrics[3].metric("Grid scenarios", len(grid.probabilities))
    show_figure(distribution_figure(grid, threshold))
    st.caption(
        "Orange bars satisfy the loss predicate. These are exact answers for the displayed "
        "finite model; the quantum circuit estimates the same orange probability mass."
    )
    if confidence == 0.999:
        st.warning(
            "99.9% risk on this small grid is exploratory. Inspect grid convergence and "
            "truncated probability before interpreting this tail."
        )
    st.subheader("Run a real Qiskit circuit")
    controls = st.columns([2, 1, 1])
    epsilon = controls[0].selectbox(
        "Target absolute error",
        [0.02, 0.01, 0.005],
        index=1,
        format_func=lambda x: f"{x:.1%} probability",
    )
    seed = controls[1].number_input("Random seed", min_value=0, max_value=1000000, value=17)
    controls[2].write("")
    run = controls[2].button("Run IQAE", type="primary", width="stretch")
    key = (family, bits, weight_units, threshold, epsilon, seed)
    if run:
        st.session_state.pop("live_result", None)
        with st.spinner(
            "Preparing the state, checking the oracle, and sampling amplified circuits..."
        ):
            settings = {
                "family": family,
                "bits": bits,
                "weight_units": weight_units,
                "threshold": threshold,
                "epsilon": epsilon,
                "seed": int(seed),
            }
            try:
                process = subprocess.run(
                    [sys.executable, "-m", "tailrisk.live"],
                    input=json.dumps(settings),
                    text=True,
                    capture_output=True,
                    cwd=ROOT,
                    check=True,
                    timeout=120,
                )
                output = json.loads(process.stdout)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                st.error(
                    "The circuit run did not complete. Try fewer qubits or a looser error target."
                )
                with st.expander("Run details"):
                    st.code(str(error))
                st.stop()
            st.session_state["live_result"] = (
                key,
                output["quantum"],
                output["classical"],
                output["resources"],
            )
    saved = st.session_state.get("live_result")
    if saved and saved[0] == key:
        _, result, baselines, cost = saved
        columns = st.columns(4)
        columns[0].metric("IQAE estimate", f"{result['estimate']:.3%}")
        columns[1].metric("A / A-dagger calls", f"{result['queries']:,}")
        columns[2].metric("Total circuit shots", f"{result['shots']:,}")
        columns[3].metric("Largest Grover power", result["max_power"])
        low, high = result["confidence_interval"]
        st.info(
            f"95% confidence interval: {low:.3%} to {high:.3%}. "
            f"Exact grid probability: {truth:.3%}."
        )
        table = [
            {
                "Method": r["method"],
                "Estimate": f"{r['estimate']:.4%}",
                "Absolute error": f"{abs(r['estimate'] - truth):.4%}",
                "Samples / A calls": r["queries"],
            }
            for r in [*baselines, result]
        ]
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption(
            "This is one random run, not a ranking. Classical budgets are rounded down to "
            "a power of two for Sobol. Use the replicated benchmark for comparisons."
        )
        with st.expander("Circuit execution ledger"):
            st.dataframe(result["circuits"], hide_index=True)
            st.json(cost)
    else:
        st.caption(
            "Select Run IQAE to calculate an estimate. No saved result is presented as a live run."
        )
    with st.expander("Continuous model, truncation, and grid error"):
        full, bounded = continuous_references(family, threshold, tuple(grid.weights), grid.bound)
        st.dataframe(
            [
                {"Reference": "Full continuous model", "Tail probability": full},
                {
                    "Reference": "Continuous model conditioned on grid bounds",
                    "Tail probability": bounded,
                },
                {"Reference": "Exact discretized model", "Tail probability": truth},
            ],
            hide_index=True,
        )
        st.write(f"Probability outside the two-asset domain: {1 - grid.retained_mass:.4%}.")
        st.write(
            "Gaussian tails have an analytic answer. Student-copula tails use one-dimensional "
            "conditional quadrature. Both are strong classical references for this small problem."
        )

with benchmark_tab:
    st.subheader("Classical vs quantum as N grows")
    scaling = ROOT / "results/scaling"
    if (scaling / "budget_summary.csv").exists():
        matched = pd.read_csv(scaling / "budget_summary.csv")
        st.write(
            "N is the computation budget: classical samples or quantum A/A-dagger calls. "
            "These counts match exactly here, but a quantum call costs more than a sample. "
            "S is the number of scenarios, held at 64 for this comparison."
        )
        show_figure(budget_scaling_figure(matched))
        st.caption(
            "Student model, 50/50 weights, 10% loss threshold. 30 independent trials per point. "
            "Shaded bands: bootstrap 95% RMSE intervals. Lower is better. "
            "Quantum rows reuse the recorded MLAE experiment; MC and IS use the exact same N."
        )
        endpoint = matched[matched.queries == matched.queries.max()].set_index("method")
        cols = st.columns(3)
        cols[0].metric("N for both methods", f"{int(endpoint.queries.iloc[0]):,}")
        cols[1].metric("Monte Carlo RMSE", f"{100 * endpoint.loc['MC', 'rmse']:.3f} pp")
        cols[2].metric("Quantum MLAE RMSE", f"{100 * endpoint.loc['MLAE', 'rmse']:.3f} pp")
        st.write(
            "Importance sampling remains a stronger classical baseline in this experiment. "
            "Query savings do not establish a wall-clock speedup."
        )
        with st.expander("Matched-N table and accuracy target"):
            st.dataframe(
                matched[["method", "queries", "rmse", "rmse_low", "rmse_high"]], hide_index=True
            )
            st.dataframe(accuracy_table(matched, 0.002), hide_index=True)
        if (scaling / "scenario_summary.csv").exists():
            with st.expander("What grows when there are more scenarios?"):
                show_figure(
                    scenario_scaling_figure(
                        pd.read_csv(scaling / "scenario_summary.csv"),
                        pd.read_csv(scaling / "scenario_resources.csv"),
                    )
                )
                st.caption(
                    "S = 16, 64, 256 scenarios, with N = 576 fixed. This refines a two-asset grid; "
                    "it does not add assets. Each grid has its own exact reference probability. "
                    "Times are simulator/CPU measurements, not quantum hardware times."
                )
    st.subheader("Broader comparison with strong classical baselines")
    summary_path = ROOT / "results/benchmark/summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        scenario = st.selectbox("Recorded experiment", summary["scenario"].unique())
        selected = summary[summary["scenario"] == scenario]
        show_figure(benchmark_figure(summary, scenario))
        st.caption(
            f"Saved experiment: {int(selected['replicates'].iloc[0])} independent replicates "
            "per point. Shading: bootstrap 95% intervals for measured RMSE. "
            "All experiments here use 50/50 weights and 3 qubits per asset."
        )
        target = st.select_slider(
            "Target RMSE (absolute probability)",
            options=[0.001, 0.002, 0.005, 0.01],
            value=0.002,
            format_func=lambda x: f"{x:.1%}",
        )
        st.dataframe(
            accuracy_table(selected, target)[["method", "first_observed_budget", "status"]],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "First measured budget whose upper RMSE uncertainty band meets the target. "
            "No extrapolation beyond the tested budgets; this is empirical performance."
        )
        st.write(
            "MLAE uses fixed Grover schedules and finite shots from Aer. Classical methods use "
            "power-of-two sample budgets; x-coordinates show each method's actual cost. "
            "A circuit call and a classical sample are different resources, not equal runtimes."
        )
        st.dataframe(
            selected[["method", "queries", "rmse", "rmse_low", "rmse_high", "median_seconds"]],
            hide_index=True,
            width="stretch",
        )
        st.download_button(
            "Download all raw estimates",
            (ROOT / "results/benchmark/estimates.csv").read_bytes(),
            "tailrisk_estimates.csv",
            "text/csv",
        )
    else:
        st.info("Run python main.py benchmark to generate the recorded experiment.")

with circuit_tab:
    st.subheader("The cost behind an oracle call")
    st.code(
        "Prepare joint loss state -> mark loss >= threshold -> amplitude estimation", language=None
    )
    study_path = ROOT / "results/studies/studies.json"
    if study_path.exists():
        study = json.loads(study_path.read_text())
        st.write("Two implementations of the same 64-scenario Student-copula threshold predicate:")
        st.dataframe(
            [{"oracle": name, **record} for name, record in study["oracles"].items()],
            hide_index=True,
        )
        st.caption(
            "Counts include state preparation. Basis: rz, sx, x, cx; optimization level 1. "
            "Arithmetic computes the weighted loss, compares it, and uncomputes workspace."
        )
        show_figure(grid_figure(study))
        if "noise" in study:
            show_figure(noise_figure(study))
            st.caption(
                "Ten independent runs per point; 2 qubits per asset. Depolarizing noise is applied "
                "after each CX. This is a sensitivity experiment, not a hardware calibration."
            )
        if "var" in study:
            with st.expander("Quantum VaR and expected-shortfall experiment"):
                st.write(f"95% VaR search resolved: {study['var']['resolved']}")
                st.write(f"Returned bracket: {study['var']['bracket']}")
                st.write(f"Search A calls: {study['var']['queries']:,}")
                st.write(f"Expected-shortfall quantile source: {study['es_quantile_source']}")
                st.write(
                    f"Quantum ES: {study['es']['estimate']:.3%}; exact grid ES: {study['exact_es']:.3%}"
                )
                st.write(f"Conditional ES interval: {study['es']['confidence_interval']}")
    else:
        st.info("Run python main.py studies to generate the circuit and noise experiments.")
    st.write(
        "Loading an arbitrary probability table and marking its states both scale with the "
        "table size. The small grid is classically trivial. This experiment measures estimator "
        "behavior; it does not establish a practical speedup."
    )

with method_tab:
    st.subheader("One financial question, four estimators")
    st.markdown("""
1. Compound daily industry returns into overlapping 10-trading-day returns.
2. Fit Gaussian and Student-t-marginal/Gaussian-copula loss models.
3. Integrate probability mass in grid cells and encode their square roots in a quantum state.
4. Flip an objective qubit exactly when the weighted portfolio loss crosses the threshold.
5. Estimate its probability using IQAE or fixed-budget MLAE.
6. Compare against Monte Carlo, scrambled Sobol, importance sampling, and exact classical references.
""")
    st.latex(r"P(L \geq x) = \sum_j p_j\,\mathbf{1}\{L_j \geq x\}")
    st.latex(r"N_A = \sum_i s_i(2k_i+1)")
    st.write(
        "Each shot at Grover power k uses 2k+1 calls to A or its inverse. "
        "The k=0 circuit still costs one state preparation per shot."
    )
    st.latex(r"\mathrm{ES}_c = q_c + \frac{\mathbb{E}[(L-q_c)_+]}{1-c}")
    st.write(
        "The excess-loss formula handles probability mass exactly at discrete VaR. "
        "The payoff circuit uses controlled rotations for each scenario; their exponential "
        "construction cost is included in the resource discussion."
    )
    st.markdown(
        "**What is original here:** the public-data pipeline, identical-grid comparisons, "
        "instrumented query ledger, oracle comparison, and error/noise analysis. "
        "QAE and its finance applications are established prior work."
    )
    st.markdown(
        "[Quantum risk analysis](https://www.nature.com/articles/s41534-019-0130-6) | "
        "[Qiskit Algorithms](https://qiskit-community.github.io/qiskit-algorithms/) | "
        "[Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)"
    )
