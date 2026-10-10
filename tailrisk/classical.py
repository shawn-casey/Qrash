"""Classical estimators of the same finite probability distribution used by QAE."""

from time import perf_counter

import numpy as np
from scipy.special import logsumexp
from scipy.stats import qmc

from tailrisk.models import LossGrid


def discrete_transform(probabilities, uniforms):
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, uniforms, side="right")


def sobol_indices(grid: LossGrid, count: int, seed: int):
    if count < 1 or count & (count - 1):
        raise ValueError("Scrambled Sobol requires a power-of-two sample count")
    # A two-dimensional conditional inverse CDF, preserving asset register order.
    m = len(grid.nodes)
    joint = grid.probabilities.reshape((m, m), order="F")
    marginal = joint.sum(axis=1)
    uniforms = qmc.Sobol(2, scramble=True, rng=np.random.default_rng(seed)).random_base2(
        int(np.log2(count))
    )
    first = discrete_transform(marginal, uniforms[:, 0])
    second = np.empty(count, dtype=int)
    for i in np.unique(first):
        selected = first == i
        second[selected] = discrete_transform(joint[i] / marginal[i], uniforms[selected, 1])
    return first + m * second


def tilted_proposal(grid: LossGrid, threshold: float):
    mean = grid.probabilities @ grid.losses
    std = np.sqrt(grid.probabilities @ (grid.losses - mean) ** 2)
    tilt = float(np.clip((threshold - mean) / std, 0.5, 5.0))
    log_probabilities = np.full_like(grid.probabilities, -np.inf)
    np.log(grid.probabilities, out=log_probabilities, where=grid.probabilities > 0)
    tilted = log_probabilities + tilt * (grid.losses - mean) / std
    proposal = np.exp(tilted - logsumexp(tilted))
    # Defensive mixture guarantees support and bounds likelihood ratios by 20.
    return 0.95 * proposal + 0.05 * grid.probabilities, tilt


def estimate(grid: LossGrid, threshold: float, method: str, count: int, seed: int):
    if count < 1:
        raise ValueError("count must be positive")
    start = perf_counter()
    values = grid.tail_values(threshold)
    rng = np.random.default_rng(seed)
    setup_seconds = 0.0
    ess = None
    tilt = None
    if method == "MC":
        indices = rng.choice(len(values), count, p=grid.probabilities)
        observations = values[indices]
    elif method == "Sobol":
        indices = sobol_indices(grid, count, seed)
        observations = values[indices]
    elif method == "IS":
        proposal, tilt = tilted_proposal(grid, threshold)
        setup_seconds = perf_counter() - start
        indices = rng.choice(len(values), count, p=proposal)
        ratios = grid.probabilities[indices] / proposal[indices]
        observations = values[indices] * ratios
        ess = float(ratios.sum() ** 2 / np.dot(ratios, ratios))
    else:
        raise ValueError("method must be MC, Sobol, or IS")
    return {
        "method": method,
        "estimate": float(np.mean(observations)),
        "queries": count,
        "seed": seed,
        "seconds": perf_counter() - start,
        "setup_seconds": setup_seconds,
        "effective_sample_size": ess,
        "tilt": tilt,
    }
