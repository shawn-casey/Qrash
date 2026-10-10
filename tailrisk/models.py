"""Two loss marginals joined by a Gaussian copula, with deterministic quadrature."""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.integrate import quad, quad_vec
from scipy.optimize import minimize
from scipy.stats import norm, t


@dataclass
class LossModel:
    family: str
    location: list[float]
    scale: list[float]
    rho: float
    df: list[float] | None = None

    def __post_init__(self):
        if self.family not in ("gaussian", "student"):
            raise ValueError("family must be gaussian or student")
        if len(self.location) != 2 or len(self.scale) != 2 or min(self.scale) <= 0:
            raise ValueError("Two marginals with positive scales are required")
        if not -1 < self.rho < 1:
            raise ValueError("Correlation must be strictly between -1 and 1")
        if self.family == "student" and (self.df is None or min(self.df) <= 2):
            raise ValueError("Student marginals require df > 2")

    def marginal(self, asset: int):
        if self.family == "gaussian":
            return norm(loc=self.location[asset], scale=self.scale[asset])
        return t(df=self.df[asset], loc=self.location[asset], scale=self.scale[asset])

    def to_dict(self):
        return asdict(self)

    def tail(self, threshold: float, weights=(0.5, 0.5), bound=None) -> float:
        """Continuous tail; optionally condition both asset losses on [-bound, bound]."""
        w = np.asarray(weights)
        if w.shape != (2,) or np.any(w <= 0) or not np.isclose(w.sum(), 1):
            raise ValueError("Two strictly positive weights must sum to one")
        if self.family == "gaussian" and bound is None:
            covariance = np.outer(self.scale, self.scale) * np.array([[1, self.rho], [self.rho, 1]])
            return float(norm.sf(threshold, np.dot(w, self.location), np.sqrt(w @ covariance @ w)))
        m0, m1 = self.marginal(0), self.marginal(1)
        conditional_sd = np.sqrt(1 - self.rho**2)
        zlo = -9.0 if bound is None else norm.ppf(m0.cdf(-bound))
        zhi = 9.0 if bound is None else norm.ppf(m0.cdf(bound))
        lower1 = -np.inf if bound is None else norm.ppf(m1.cdf(-bound))
        upper1 = np.inf if bound is None else norm.ppf(m1.cdf(bound))

        def integrand(z, tail):
            loss0 = m0.ppf(norm.cdf(z))
            required = (threshold - w[0] * loss0) / w[1]
            z_required = norm.ppf(m1.cdf(required)) if tail else -np.inf
            low = max(lower1, z_required)
            probability = max(
                0.0,
                norm.cdf((upper1 - self.rho * z) / conditional_sd)
                - norm.cdf((low - self.rho * z) / conditional_sd),
            )
            return norm.pdf(z) * probability

        numerator = quad(lambda z: integrand(z, True), zlo, zhi, epsabs=1e-10, limit=200)[0]
        denominator = (
            1.0
            if bound is None
            else quad(lambda z: integrand(z, False), zlo, zhi, epsabs=1e-10, limit=200)[0]
        )
        return float(numerator / denominator)


def fit_model(losses: np.ndarray, family: str = "gaussian") -> LossModel:
    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 2 or losses.shape[1] != 2 or len(losses) < 30:
        raise ValueError("At least 30 complete two-asset observations are required")
    if not np.all(np.isfinite(losses)):
        raise ValueError("Loss observations must be finite")
    if family == "gaussian":
        return LossModel(
            family,
            losses.mean(0).tolist(),
            losses.std(0, ddof=1).tolist(),
            float(np.corrcoef(losses.T)[0, 1]),
        )
    if family != "student":
        raise ValueError("Unknown family")
    locations, scales, dfs, latent = [], [], [], []
    for values in losses.T:
        fit = minimize(
            lambda p, values=values: -t.logpdf(values, df=p[0], loc=p[1], scale=np.exp(p[2])).sum(),
            [6.0, np.median(values), np.log(np.std(values) * 0.75)],
            bounds=[(2.1, 100.0), (None, None), (-12, 1)],
            method="L-BFGS-B",
        )
        if not fit.success:
            raise RuntimeError(f"Student marginal fit failed: {fit.message}")
        df, loc, log_scale = fit.x
        dfs.append(float(df))
        locations.append(float(loc))
        scales.append(float(np.exp(log_scale)))
        u = t.cdf(values, df=df, loc=loc, scale=np.exp(log_scale))
        latent.append(norm.ppf(np.clip(u, 1e-10, 1 - 1e-10)))
    return LossModel(family, locations, scales, float(np.corrcoef(latent)[0, 1]), dfs)


@dataclass
class LossGrid:
    probabilities: np.ndarray
    losses: np.ndarray
    nodes: np.ndarray
    bits: int
    weight_units: tuple[int, int]
    bound: float
    retained_mass: float
    quadrature_error: float

    @property
    def weights(self):
        return np.array(self.weight_units) / sum(self.weight_units)

    @property
    def integer_losses(self):
        m = len(self.nodes)
        index = np.arange(m * m)
        return self.weight_units[0] * (index % m) + self.weight_units[1] * (index // m)

    @property
    def step(self):
        return (self.nodes[1] - self.nodes[0]) / sum(self.weight_units)

    def cutoff(self, threshold: float) -> int:
        # The same support comparison defines classical and quantum boundary behavior.
        sums = np.unique(self.integer_losses)
        values = self.nodes[0] + self.step * sums
        eligible = sums[values >= threshold]
        return int(eligible[0]) if len(eligible) else int(sums[-1] + 1)

    def tail_values(self, threshold: float) -> np.ndarray:
        return (self.integer_losses >= self.cutoff(threshold)).astype(float)

    def tail(self, threshold: float) -> float:
        return float(self.probabilities @ self.tail_values(threshold))

    def risk(self, confidence: float) -> tuple[float, float]:
        if not 0 < confidence < 1:
            raise ValueError("Confidence must be between zero and one")
        order = np.argsort(self.losses, kind="stable")
        cumulative = np.cumsum(self.probabilities[order])
        index = min(np.searchsorted(cumulative, confidence), len(order) - 1)
        quantile = float(self.losses[order[index]])
        excess = np.maximum(self.losses - quantile, 0)
        expected_shortfall = quantile + self.probabilities @ excess / (1 - confidence)
        return quantile, float(expected_shortfall)


def make_grid(model: LossModel, bits=3, weight_units=(1, 1), bound=0.25) -> LossGrid:
    if not isinstance(bits, int) or not 1 <= bits <= 8:
        raise ValueError("bits must be an integer from 1 to 8")
    if len(weight_units) != 2 or any(not isinstance(w, int) or w <= 0 for w in weight_units):
        raise ValueError("Weights must be two positive integers")
    if bound <= 0:
        raise ValueError("bound must be positive")
    m = 2**bits
    edges = np.linspace(-bound, bound, m + 1)
    nodes = (edges[:-1] + edges[1:]) / 2
    latent_edges = [norm.ppf(model.marginal(i).cdf(edges)) for i in range(2)]
    matrix = np.empty((m, m))
    quadrature_error = 0.0
    conditional_sd = np.sqrt(1 - model.rho**2)
    for i in range(m):

        def integrand(z):
            cdf = norm.cdf((latent_edges[1] - model.rho * z) / conditional_sd)
            return norm.pdf(z) * np.diff(cdf)

        row, error = quad_vec(
            integrand, latent_edges[0][i], latent_edges[0][i + 1], epsabs=1e-11, epsrel=1e-10
        )
        matrix[i] = row
        quadrature_error += error
    if matrix.min() < -1e-12:
        raise ValueError("Negative cell probability from quadrature")
    matrix = np.maximum(matrix, 0)
    retained_mass = float(matrix.sum())
    if not 0 < retained_mass <= 1 + 1e-10:
        raise ValueError("Invalid retained probability mass")
    probabilities = matrix.ravel(order="F") / retained_mass
    index = np.arange(m * m)
    sums = weight_units[0] * (index % m) + weight_units[1] * (index // m)
    losses = nodes[0] + ((nodes[1] - nodes[0]) / sum(weight_units)) * sums
    return LossGrid(
        probabilities,
        losses,
        nodes,
        bits,
        tuple(weight_units),
        bound,
        retained_mass,
        float(quadrature_error),
    )
