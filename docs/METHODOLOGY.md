# Methodology

## Model and estimand

Let R_i be an industry's compounded ten-trading-day simple return, X_i = -R_i its loss, and L = w_0 X_0 + w_1 X_1 the portfolio loss. We use nonnegative weights that sum to one. The event is always `L >= threshold`.

The Gaussian control uses fitted marginal means, sample standard deviations, and Pearson correlation. For this model, portfolio loss is univariate normal; its tail is calculated analytically.

The second model fits Student-t loss marginals and joins them with a Gaussian copula. In latent standard-normal coordinates, Z_1 conditioned on Z_0=z has mean rho*z and standard deviation sqrt(1-rho^2). This makes both rectangular cell integration and continuous tail reference calculations one-dimensional quadrature problems. The full Student-copula tail reference integrates latent z over [-9, 9]; the omitted standard-normal mass is negligible relative to the reported precision. Quadrature remains a numerical reference, not an algebraic exact answer.

## Discretization

Each asset loss is restricted to [-0.25, 0.25], divided into 2^n equal-width cells, and represented by the cell midpoint. Joint cell masses are computed by deterministic adaptive conditional quadrature. They are normalized by the retained joint mass. We record the integration error estimate and retained mass.

Grid probabilities therefore describe the distribution conditioned on both asset losses lying within the bounds. The full model, bounded continuous model, and finite grid have separate answers. For a fixed threshold:

```text
truncation effect = bounded continuous tail - full continuous tail
quantization error = exact grid tail - bounded continuous tail
estimation error = estimator output - exact grid tail
```

At the default three bits per asset there are 64 joint scenarios. "Exact" grid reference means a direct sum of the recorded PMF, up to floating-point and PMF quadrature error. It does not mean an exact financial forecast.

## Classical estimators

- **MC:** independent categorical samples from the grid PMF.
- **Sobol:** a scrambled two-dimensional Sobol sequence with power-of-two sample counts. The first coordinate selects an asset-0 marginal cell; the second selects asset 1 conditional on that cell. It samples exactly the same joint PMF in distribution. Each reported benchmark replicate uses an independent scramble.
- **IS:** exponential tilting of the grid loss distribution, mixed with 5% of the original PMF to maintain support. The tilt is a predeclared function of the threshold, mean, and standard deviation. Samples are weighted by p/q without self-normalization. Setup, proposal normalization, and evaluation time are recorded. The tabulated proposal is built in O(number of scenarios), so this baseline also relies on explicit model access.

Sobol and importance sampling are not sampling-only black boxes. QAE has the additional coherent A/A-dagger access requirement. Our query plot compares estimator behavior under these disclosed capabilities; it is not a lower-bound proof under identical access.

## Quantum state and predicates

StatePreparation loads amplitudes sqrt(p_j) into the joint asset register. It is unitary and can be inverted. Reset-based Initialize is not used.

For asset indices i_0 and i_1, with integer weight units u_0 and u_1, the portfolio loss is an affine function of `u_0*i_0 + u_1*i_1`. The first asset occupies the least significant qubits. We use the same integer cutoff and floating-point support convention in the classical predicate and quantum construction.

Two implementations are included:

1. **Direct marking:** controlled X gates mark the scenario bit strings that satisfy the predicate. This uses 2n+1 qubits but has table-size construction cost.
2. **Arithmetic:** WeightedAdder computes the integer weighted loss, IntegerComparator marks it, and inverse addition uncomputes the sum and carry workspace. Statevector tests check the objective probability and zeroed workspace. At the default setting it uses 18 qubits.

The main replicated benchmark uses the smaller direct oracle. The arithmetic implementation is a validated resource ablation; it is not secretly substituted into the benchmark. Both use the full joint probability state. Neither encodes a classically precomputed tail probability into a single rotation.

## Estimation and cost accounting

For amplitude a, write theta = asin(sqrt(a)). The objective probability after k Grover iterations is sin^2((2k+1)*theta). Tests verify this relationship by simulating the actual circuits. The experiment itself samples finite-shot transpiled Aer circuits; it does not draw binomial samples from the analytic amplification formula.

Each shot executing Q^k A costs 2k+1 calls to A/A-dagger. Total cost is:

```text
A calls = sum_i shots_i * (2*power_i + 1)
Grover calls = sum_i shots_i * power_i
```

The k=0 circuit is included. Instrumented Grover calls are cross-checked against Qiskit's result. Every quantum result includes actual shots, ones counts, powers, gate counts, maximum depth, compilation time, and execution metadata.

**Fixed-budget comparison:** MLAE uses schedules [0], [0,1], [0,1,2], [0,1,2,4], [0,1,2,4,8], [0,1,2,4,8,16], with 64 shots per circuit. The resulting A-call budgets are 64, 256, 576, 1152, 2240, and 4352. Classical budgets are rounded down to the nearest power of two for Sobol. The plotted x-coordinates are each method's actual budget; they are not artificially aligned. A level is a nearby-budget comparison, not equal runtime or exactly equal query cost.

**Accuracy-target comparison:** IQAE adapts Grover powers and stops according to Qiskit's target error logic. We report its actual realized cost and confidence interval. We do not label it as a fixed-budget estimator.

The main benchmark uses 30 independent replicates per point. RMSE is computed against exact grid truth. Bands are percentile bootstrap 95% intervals for RMSE from 2,000 resamples of the replicate errors; these bands describe uncertainty in measured performance. They are not confidence intervals for an individual amplitude. MLAE's asymptotic Fisher intervals are not presented as finite-sample coverage guarantees.

Oracle compilation is cached within a scenario. Runtime includes the actual cost incurred by a given run, and the raw ledger records compilation costs. Median simulator runtime is not a quantum hardware runtime estimate. Basis gates are rz, sx, x, cx, optimization level 1, transpiler seed 17, with no device connectivity constraint.

## VaR and expected shortfall

Risk confidence c and estimator failure probability delta are distinct. VaR_c is the smallest grid loss whose CDF is at least c.

The quantum VaR search bisects the ordered support and estimates CDF predicates. It only discards a half when the estimated interval lies on the appropriate side of c. Each refinement receives part of a union-bound failure budget. If the CDF interval remains ambiguous after the declared precision schedule, it returns an unresolved bracket. All calls across the search are counted.

For q=VaR_c:

```text
ES_c = q + E[max(L-q, 0)] / (1-c)
```

This treats atoms at the quantile correctly. Using E[L | L>=q] would generally give a different discrete answer. The positive excess is divided by its maximum and encoded with uniformly controlled Y rotations. The angles implement 2*asin(sqrt(payoff)) exactly up to floating-point and gate synthesis; there is no small-angle approximation in this implementation. Its arbitrary-table construction cost is exponential in the state-register size.

The ES interval is conditional on the supplied quantile. When the preceding quantum VaR search resolves, its failure budget and the ES failure budget must both be considered: two 95% guarantees imply at least 90% joint coverage by a union bound, not 95% end-to-end. If VaR is unresolved, the saved study explicitly labels use of the exact grid quantile for the excess experiment.

## Noise and resources

The noise study uses a smaller 16-scenario, five-qubit oracle, ten independent runs per point, 128 shots per circuit, and CX depolarizing parameters 0, 0.0001, 0.001, and 0.005. We apply the channel after each transpiled CX and use Aer density-matrix simulation. This model excludes readout errors, decoherence, routing, crosstalk, and hardware calibration. The same ideal-model MLAE estimator is deliberately used to reveal degradation under noise.

Resource tables include state preparation, predicate computation, and uncomputation. We report qubits, gate count, CX count, and circuit depth in the declared basis. These are logical circuit metrics with unrestricted connectivity.

We do not report an invented T-count. Arbitrary rz rotations require approximate Clifford+T synthesis with a global error allocation, and translating logical counts into physical qubits or runtime requires an error-correction architecture. Those steps remain future work. The cited derivative-pricing resource paper motivates that distinction; its numerical resource estimates do not apply directly to this small portfolio example.

## Limits

The demonstration is classically trivial at its current size. Generic probability loading, lookup predicates, explicit classical PMFs, and proposal setup all scale with grid size. The Student model is a descriptive fit and its Gaussian copula does not capture asymptotic tail dependence. The 99.9% grid risk values are exploratory, especially when truncation removes comparable probability mass. There is no claim of practical quantum advantage or validated real-world risk forecasting.
