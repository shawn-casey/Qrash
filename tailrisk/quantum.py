"""Coherent portfolio oracles and instrumented Qiskit amplitude estimation."""

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.circuit.library import IntegerComparator, StatePreparation, UCRYGate, WeightedAdder
from qiskit.primitives import BaseSamplerV2
from qiskit.primitives.containers import SamplerPub
from qiskit.quantum_info import Statevector
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.primitives import SamplerV2
from qiskit_algorithms import (
    EstimationProblem,
    IterativeAmplitudeEstimation,
    MaximumLikelihoodAmplitudeEstimation,
)

from tailrisk.models import LossGrid

BASIS = ["rz", "sx", "x", "cx"]


@dataclass
class Oracle:
    circuit: QuantumCircuit
    objective: int
    state_qubits: int
    workspace: list[int]
    kind: str
    compiled: dict = field(default_factory=dict, repr=False)

    @property
    def problem(self):
        return EstimationProblem(state_preparation=self.circuit, objective_qubits=[self.objective])


def direct_oracle(grid: LossGrid, values: np.ndarray, kind="direct") -> Oracle:
    """Encode a Boolean predicate or bounded payoff without encoding its expectation."""
    values = np.asarray(values, dtype=float)
    if values.shape != grid.probabilities.shape or np.any((values < 0) | (values > 1)):
        raise ValueError("One payoff in [0, 1] is required for each grid state")
    n = 2 * grid.bits
    circuit = QuantumCircuit(n + 1, name="A")
    circuit.append(StatePreparation(np.sqrt(grid.probabilities)), range(n))
    if np.all((values == 0) | (values == 1)):
        marked = np.flatnonzero(values)
        if len(marked) > len(values) / 2:
            circuit.x(n)
            marked = np.flatnonzero(1 - values)
        for index in marked:
            zero_bits = [bit for bit in range(n) if not (int(index) >> bit) & 1]
            for bit in zero_bits:
                circuit.x(bit)
            circuit.mcx(list(range(n)), n)
            for bit in zero_bits:
                circuit.x(bit)
    else:
        angles = (2 * np.arcsin(np.sqrt(values))).tolist()
        circuit.append(UCRYGate(angles), [n, *range(n)])
    return Oracle(circuit, n, n, [], kind)


def tail_oracle(grid: LossGrid, threshold: float, arithmetic=False) -> Oracle:
    if not arithmetic:
        return direct_oracle(grid, grid.tail_values(threshold))
    n = 2 * grid.bits
    weights = [unit * 2**bit for unit in grid.weight_units for bit in range(grid.bits)]
    adder = WeightedAdder(n, weights)
    cutoff = grid.cutoff(threshold)
    comparator = IntegerComparator(adder.num_sum_qubits, cutoff, geq=True)
    objective = adder.num_qubits
    circuit = QuantumCircuit(adder.num_qubits + 1 + comparator.num_ancillas, name="A_arithmetic")
    circuit.append(StatePreparation(np.sqrt(grid.probabilities)), range(n))
    circuit.compose(adder, range(adder.num_qubits), inplace=True)
    compare_registers = [
        *range(n, n + adder.num_sum_qubits),
        objective,
        *range(objective + 1, circuit.num_qubits),
    ]
    circuit.compose(comparator, compare_registers, inplace=True)
    circuit.compose(adder.inverse(), range(adder.num_qubits), inplace=True)
    workspace = [i for i in range(n, circuit.num_qubits) if i != objective]
    return Oracle(circuit, objective, n, workspace, "arithmetic")


def verify_oracle(oracle: Oracle) -> dict:
    state = Statevector.from_instruction(oracle.circuit)
    probability = float(state.probabilities([oracle.objective])[1])
    workspace_error = 0.0
    if oracle.workspace:
        workspace_error = float(1 - state.probabilities(oracle.workspace)[0])
    return {"objective_probability": probability, "workspace_nonzero_probability": workspace_error}


def resource_counts(circuit):
    return {
        "qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "cx": int(circuit.count_ops().get("cx", 0)),
        "gates": sum(circuit.count_ops().values()),
    }


def resources(oracle: Oracle):
    start = perf_counter()
    compiled = transpile(
        oracle.circuit, basis_gates=BASIS, optimization_level=1, seed_transpiler=17
    )
    return {
        **resource_counts(compiled),
        "compilation_seconds": perf_counter() - start,
        "basis": BASIS,
        "optimization_level": 1,
        "seed_transpiler": 17,
    }


def tag_circuit(circuit, objective, power, measurement):
    circuit.metadata = {"grover_power": int(power)}
    if measurement:
        circuit.add_register(ClassicalRegister(1, "meas"))
        circuit.measure(objective, 0)
    return circuit


class RecordedIQAE(IterativeAmplitudeEstimation):
    def construct_circuit(self, estimation_problem, k=0, measurement=False):
        circuit = super().construct_circuit(estimation_problem, k, measurement=False)
        return tag_circuit(circuit, estimation_problem.objective_qubits[0], k, measurement)


class RecordedMLAE(MaximumLikelihoodAmplitudeEstimation):
    def __init__(self, evaluation_schedule, **kwargs):
        self.schedule = list(evaluation_schedule)
        super().__init__(evaluation_schedule=self.schedule, **kwargs)

    def construct_circuits(self, estimation_problem, measurement=False):
        circuits = super().construct_circuits(estimation_problem, measurement=False)
        return [
            tag_circuit(circuit, estimation_problem.objective_qubits[0], k, measurement)
            for circuit, k in zip(circuits, self.schedule)
        ]


class CircuitSampler(BaseSamplerV2):
    """Run actual Aer circuits and record shots, Grover powers, and gate counts."""

    def __init__(self, oracle: Oracle, shots=256, seed=17, noise=0.0):
        if shots < 1 or not 0 <= noise <= 1:
            raise ValueError("Invalid shots or depolarizing error")
        self.oracle = oracle
        self.default_shots = shots
        self.rng = np.random.default_rng(seed)
        self.records = []
        self.noise = noise

    def run(self, pubs, *, shots=None):
        prepared, entries = [], []
        for raw_pub in pubs:
            pub = SamplerPub.coerce(raw_pub, shots=shots or self.default_shots)
            if pub.circuit.num_parameters:
                raise ValueError("This sampler only supports bound circuits")
            power = pub.circuit.metadata["grover_power"]
            start = perf_counter()
            # Each Oracle owns its cache; different probabilities/predicates cannot share it.
            if power not in self.oracle.compiled:
                self.oracle.compiled[power] = transpile(
                    pub.circuit, basis_gates=BASIS, optimization_level=1, seed_transpiler=17
                )
            compiled = self.oracle.compiled[power]
            prepared.append((compiled, None, pub.shots))
            entries.append(
                {
                    "power": power,
                    "requested_shots": pub.shots,
                    "compilation_seconds": perf_counter() - start,
                    **resource_counts(compiled),
                }
            )
        backend_options = {"method": "statevector", "max_parallel_threads": 1}
        if self.noise:
            noise_model = NoiseModel()
            noise_model.add_all_qubit_quantum_error(depolarizing_error(self.noise, 2), ["cx"])
            backend_options["noise_model"] = noise_model
            backend_options["method"] = "density_matrix"
        sampler = SamplerV2(
            default_shots=self.default_shots,
            seed=int(self.rng.integers(2**31)),
            options={"backend_options": backend_options},
        )
        start = perf_counter()
        job = sampler.run(prepared)
        results = job.result()
        elapsed = perf_counter() - start
        for entry, result in zip(entries, results):
            counts = result.data.meas.get_counts()
            actual_shots = int(sum(counts.values()))
            self.records.append(
                {
                    **entry,
                    "shots": actual_shots,
                    "ones": int(counts.get("1", 0)),
                    "a_calls": actual_shots * (2 * entry["power"] + 1),
                    "grover_calls": actual_shots * entry["power"],
                    "batch_execution_seconds": elapsed,
                }
            )
        return job


def run_iqae(oracle: Oracle, epsilon=0.01, delta=0.05, shots=256, seed=17):
    sampler = CircuitSampler(oracle, shots, seed)
    start = perf_counter()
    result = RecordedIQAE(epsilon_target=epsilon, alpha=delta, sampler=sampler).estimate(
        oracle.problem
    )
    return estimation_record(result, sampler, "IQAE", seed, start, epsilon=epsilon, delta=delta)


def run_mlae(oracle: Oracle, schedule=(0, 1, 2, 4), shots=64, seed=17, noise=0.0):
    if not schedule or any(not isinstance(k, int) or k < 0 for k in schedule):
        raise ValueError("Grover powers must be nonnegative integers")
    sampler = CircuitSampler(oracle, shots, seed, noise)
    start = perf_counter()
    result = RecordedMLAE(evaluation_schedule=list(schedule), sampler=sampler).estimate(
        oracle.problem
    )
    return estimation_record(
        result, sampler, "MLAE", seed, start, schedule=list(schedule), noise=noise
    )


def estimation_record(result, sampler, method, seed, start, **extra):
    records = sampler.records
    record = {
        "method": method,
        "estimate": float(result.estimation),
        "seed": seed,
        "queries": sum(row["a_calls"] for row in records),
        "shots": sum(row["shots"] for row in records),
        "grover_calls": sum(row["grover_calls"] for row in records),
        "max_power": max(row["power"] for row in records),
        "max_depth": max(row["depth"] for row in records),
        "cx_executions": sum(row["cx"] * row["shots"] for row in records),
        "seconds": perf_counter() - start,
        "circuits": records,
        **extra,
    }
    if method == "IQAE":
        record["confidence_interval"] = list(map(float, result.confidence_interval))
    # MLAE's Fisher interval is asymptotic. We use empirical replicate errors instead.
    if record["grover_calls"] != result.num_oracle_queries:
        raise RuntimeError("Instrumented Grover count disagrees with Qiskit")
    return record


def quantum_var(
    grid: LossGrid, confidence=0.95, delta=0.05, seed=17, epsilons=(0.02, 0.005, 0.001)
):
    if not 0 < confidence < 1 or not 0 < delta < 1:
        raise ValueError("Confidence and delta must be in (0, 1)")
    support = np.unique(grid.losses)
    low, high = 0, len(support) - 1
    max_steps = int(np.ceil(np.log2(len(support))))
    per_call_delta = delta / (max_steps * len(epsilons))
    calls = []
    while low < high:
        mid = (low + high) // 2
        oracle = direct_oracle(grid, (grid.losses <= support[mid]).astype(float), "cdf")
        classified = False
        for epsilon in epsilons:
            record = run_iqae(oracle, epsilon, per_call_delta, seed=seed + len(calls))
            record["threshold"] = float(support[mid])
            calls.append(record)
            left, right = record["confidence_interval"]
            if left >= confidence:
                high = mid
                classified = True
                break
            if right < confidence:
                low = mid + 1
                classified = True
                break
        if not classified:
            break
    return {
        "resolved": low == high,
        "bracket": [float(support[low]), float(support[high])],
        "confidence": confidence,
        "delta": delta,
        "queries": sum(row["queries"] for row in calls),
        "calls": calls,
    }


def quantum_excess(grid: LossGrid, quantile: float, confidence=0.95, epsilon=0.005, seed=17):
    excess = np.maximum(grid.losses - quantile, 0)
    scale = float(excess.max())
    if scale == 0:
        return {"estimate": quantile, "confidence_interval": [quantile, quantile], "queries": 0}
    oracle = direct_oracle(grid, excess / scale, "excess")
    record = run_iqae(oracle, epsilon=epsilon, seed=seed)
    multiplier = scale / (1 - confidence)
    return {
        **record,
        "amplitude_estimate": record["estimate"],
        "estimate": quantile + multiplier * record["estimate"],
        "confidence_interval": [quantile + multiplier * p for p in record["confidence_interval"]],
        "quantile_supplied": quantile,
        "confidence": confidence,
        "excess_scale": scale,
    }
