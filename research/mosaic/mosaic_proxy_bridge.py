"""Finite-sample target-law recovery from proxy sensitive labels.

The target audit observes a fine token, task label, and proxy source label for
many records.  A known or separately calibrated source-to-proxy confusion
matrix defines a polytope of compatible true joint laws.  This module computes
an exact conditional L1 envelope over that polytope, avoiding unstable plug-in
matrix inversion and failing closed when a source can have zero mass.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np
from scipy.optimize import linprog
from scipy.stats import beta

from mosaic_envelope import weissman_l1_radius


LP_TOLERANCE = 2e-8
VALUE_GUARD = 1e-9


@dataclass(frozen=True)
class ProxyLabelConditionalCertificate:
    label: int
    proxy_empirical_joint: np.ndarray
    proxy_l1_radius: float
    proxy_coordinate_lowers: np.ndarray | None
    proxy_coordinate_uppers: np.ndarray | None
    confusion_matrix: np.ndarray
    confusion_row_l1_radii: tuple[float, ...]
    observation_model_l1_slack: float
    effective_proxy_l1_radius: float
    nominal_true_joint: np.ndarray
    conditional_centers: np.ndarray
    conditional_l1_radii: tuple[float, ...]
    source_mass_lower_bounds: tuple[float, ...]
    source_mass_centers: tuple[float, ...]
    conditional_lp_solves: int
    confidence_region: str
    method: str


@dataclass(frozen=True)
class ProxyLabelBridgeCertificate:
    labels: tuple[ProxyLabelConditionalCertificate, ...]
    conditional_empirical_distributions: np.ndarray
    conditional_l1_radii: np.ndarray
    family_failure_probability: float
    per_event_failure_probability: float
    calibration_mode: str
    label_count: int
    source_count: int
    token_count: int
    proxy_sample_size: int
    calibration_sample_size: int
    method: str


@dataclass(frozen=True)
class ProxyDependenceCertificate:
    """Simultaneous bound on token-dependent proxy misclassification."""

    pooled_confusion_counts: np.ndarray
    per_label_l1_slack: tuple[float, ...]
    per_event_failure_probability: float
    empty_calibration_cells: tuple[tuple[int, int, int], ...]
    method: str = "clopper_pearson_token_dependence_envelope"


def certify_proxy_token_dependence(
    calibration_counts: Sequence,
    *,
    family_failure_probability: float,
) -> ProxyDependenceCertificate:
    """Bound deviation from ``R independent of C given (S,Y)``.

    ``calibration_counts`` has shape task-label x true-source x token x
    proxy-source.  The current implementation handles a binary proxy, for
    which simultaneous Clopper--Pearson intervals yield an L1 bound between
    every token-specific confusion row and its label/source pooled row.
    Empty cells fail closed with the maximal L1 slack of two.
    """

    counts = _validate_counts(
        calibration_counts,
        ndim=4,
        name="calibration_counts",
    )
    label_count, source_count, token_count, proxy_count = counts.shape
    if proxy_count != 2 or source_count != 2:
        raise ValueError("token-dependence calibration currently requires binary sources")
    if not 0.0 < family_failure_probability < 1.0:
        raise ValueError("family_failure_probability must lie in (0, 1)")
    pooled = counts.sum(axis=2)
    if np.any(pooled.sum(axis=-1) == 0):
        raise ValueError("every label/source pooled calibration row must be nonempty")
    nonempty = int(np.sum(counts.sum(axis=-1) > 0))
    event_count = label_count * source_count + nonempty
    per_event = family_failure_probability / event_count
    slacks = []
    empty = []
    for label in range(label_count):
        label_slack = 0.0
        for source in range(source_count):
            pooled_trials = int(pooled[label, source].sum())
            pooled_successes = int(pooled[label, source, 1])
            pooled_interval = _clopper_pearson_interval(
                pooled_successes,
                pooled_trials,
                per_event,
            )
            for token in range(token_count):
                trials = int(counts[label, source, token].sum())
                if trials == 0:
                    empty.append((label, source, token))
                    label_slack = 2.0
                    continue
                interval = _clopper_pearson_interval(
                    int(counts[label, source, token, 1]),
                    trials,
                    per_event,
                )
                probability_gap = max(
                    abs(interval[0] - pooled_interval[1]),
                    abs(interval[1] - pooled_interval[0]),
                )
                label_slack = max(label_slack, 2.0 * probability_gap)
        slacks.append(min(2.0, label_slack))
    return ProxyDependenceCertificate(
        pooled_confusion_counts=pooled,
        per_label_l1_slack=tuple(float(value) for value in slacks),
        per_event_failure_probability=float(per_event),
        empty_calibration_cells=tuple(empty),
    )


def _validate_counts(values: Sequence, *, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.equal(array, np.floor(array))):
            raise ValueError(f"{name} must contain integer counts")
    array = array.astype(np.int64)
    if np.any(array < 0):
        raise ValueError(f"{name} must contain nonnegative counts")
    return array


def _validate_confusion(
    values: Sequence, source_count: int, token_count: int | None = None
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    valid_shapes = {(source_count, source_count)}
    if token_count is not None:
        valid_shapes.add((source_count, token_count, source_count))
    if matrix.shape not in valid_shapes:
        raise ValueError(
            "confusion matrices must have shape sources x proxy-sources "
            "or sources x tokens x proxy-sources"
        )
    if not np.isfinite(matrix).all() or np.any(matrix < -1e-12):
        raise ValueError("confusion matrices must be finite and nonnegative")
    if not np.allclose(matrix.sum(axis=-1), 1.0, atol=1e-10):
        raise ValueError("every confusion-matrix row must sum to one")
    return np.clip(matrix, 0.0, 1.0)


def _observation_matrix(confusion: np.ndarray, token_count: int) -> np.ndarray:
    source_count = confusion.shape[0]
    matrix = np.zeros(
        (source_count * token_count, source_count * token_count),
        dtype=np.float64,
    )
    for proxy_source in range(source_count):
        for token in range(token_count):
            observation = proxy_source * token_count + token
            for source in range(source_count):
                latent = source * token_count + token
                matrix[observation, latent] = (
                    confusion[source, proxy_source]
                    if confusion.ndim == 2
                    else confusion[source, token, proxy_source]
                )
    return matrix


def _polytope_matrices(
    proxy_empirical: np.ndarray,
    confusion: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[float, float]]]:
    source_count, token_count = proxy_empirical.shape
    latent_count = source_count * token_count
    observed_count = latent_count
    observation = _observation_matrix(confusion, token_count)
    proxy_flat = proxy_empirical.reshape(-1)
    variable_count = latent_count + observed_count
    a_ub = []
    b_ub = []
    for index in range(observed_count):
        positive = np.zeros(variable_count)
        positive[:latent_count] = observation[index]
        positive[latent_count + index] = -1.0
        a_ub.append(positive)
        b_ub.append(float(proxy_flat[index]))
        negative = np.zeros(variable_count)
        negative[:latent_count] = -observation[index]
        negative[latent_count + index] = -1.0
        a_ub.append(negative)
        b_ub.append(float(-proxy_flat[index]))
    radius_row = np.zeros(variable_count)
    radius_row[latent_count:] = 1.0
    a_ub.append(radius_row)
    b_ub.append(float(radius))
    a_eq = np.zeros((1, variable_count))
    a_eq[0, :latent_count] = 1.0
    b_eq = np.asarray([1.0])
    bounds = [(0.0, 1.0)] * latent_count + [(0.0, None)] * observed_count
    return (
        np.asarray(a_ub),
        np.asarray(b_ub),
        a_eq,
        b_eq,
        bounds,
    )


def _nominal_joint(
    proxy_empirical: np.ndarray,
    confusion: np.ndarray,
    effective_radius: float,
) -> np.ndarray:
    source_count, token_count = proxy_empirical.shape
    latent_count = source_count * token_count
    observed_count = latent_count
    observation = _observation_matrix(confusion, token_count)
    proxy_flat = proxy_empirical.reshape(-1)
    variable_count = latent_count + observed_count
    objective = np.zeros(variable_count)
    objective[latent_count:] = 1.0
    a_ub = []
    b_ub = []
    for index in range(observed_count):
        positive = np.zeros(variable_count)
        positive[:latent_count] = observation[index]
        positive[latent_count + index] = -1.0
        a_ub.append(positive)
        b_ub.append(float(proxy_flat[index]))
        negative = np.zeros(variable_count)
        negative[:latent_count] = -observation[index]
        negative[latent_count + index] = -1.0
        a_ub.append(negative)
        b_ub.append(float(-proxy_flat[index]))
    a_eq = np.zeros((1, variable_count))
    a_eq[0, :latent_count] = 1.0
    result = linprog(
        objective,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        A_eq=a_eq,
        b_eq=np.asarray([1.0]),
        bounds=[(0.0, 1.0)] * latent_count + [(0.0, None)] * observed_count,
        method="highs",
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"ABSTAIN_PROXY_NOMINAL_LP: {result.message}")
    residual = float(result.fun)
    if residual > effective_radius + LP_TOLERANCE:
        raise RuntimeError(
            "ABSTAIN_EMPTY_PROXY_CONFIDENCE_POLYTOPE: "
            f"minimum residual {residual} exceeds radius {effective_radius}"
        )
    joint = np.clip(result.x[:latent_count], 0.0, 1.0)
    joint /= joint.sum()
    return joint.reshape(source_count, token_count)


def _source_mass_lower_bound(
    proxy_empirical: np.ndarray,
    confusion: np.ndarray,
    effective_radius: float,
    source: int,
) -> float:
    source_count, token_count = proxy_empirical.shape
    latent_count = source_count * token_count
    objective = np.zeros(2 * latent_count)
    objective[source * token_count : (source + 1) * token_count] = 1.0
    a_ub, b_ub, a_eq, b_eq, bounds = _polytope_matrices(
        proxy_empirical, confusion, effective_radius
    )
    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success or result.fun is None:
        raise RuntimeError(f"ABSTAIN_PROXY_SOURCE_MASS_LP: {result.message}")
    return max(0.0, float(result.fun) - VALUE_GUARD)


def _conditional_support(
    proxy_empirical: np.ndarray,
    confusion: np.ndarray,
    effective_radius: float,
    source: int,
    sign: np.ndarray,
    source_mass_lower: float,
) -> float | None:
    source_count, token_count = proxy_empirical.shape
    latent_count = source_count * token_count
    observed_count = latent_count
    observation = _observation_matrix(confusion, token_count)
    proxy_flat = proxy_empirical.reshape(-1)
    t_index = latent_count
    slack_start = latent_count + 1
    variable_count = latent_count + 1 + observed_count
    objective = np.zeros(variable_count)
    objective[source * token_count : (source + 1) * token_count] = -sign

    a_ub = []
    b_ub = []
    for index in range(observed_count):
        positive = np.zeros(variable_count)
        positive[:latent_count] = observation[index]
        positive[t_index] = -float(proxy_flat[index])
        positive[slack_start + index] = -1.0
        a_ub.append(positive)
        b_ub.append(0.0)
        negative = np.zeros(variable_count)
        negative[:latent_count] = -observation[index]
        negative[t_index] = float(proxy_flat[index])
        negative[slack_start + index] = -1.0
        a_ub.append(negative)
        b_ub.append(0.0)
    radius_row = np.zeros(variable_count)
    radius_row[t_index] = -float(effective_radius)
    radius_row[slack_start:] = 1.0
    a_ub.append(radius_row)
    b_ub.append(0.0)

    a_eq = np.zeros((2, variable_count))
    a_eq[0, :latent_count] = 1.0
    a_eq[0, t_index] = -1.0
    a_eq[
        1, source * token_count : (source + 1) * token_count
    ] = 1.0
    b_eq = np.asarray([0.0, 1.0])
    bounds = (
        [(0.0, None)] * latent_count
        + [(0.0, 1.0 / source_mass_lower)]
        + [(0.0, None)] * observed_count
    )
    result = linprog(
        objective,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if result.status == 3:
        return None
    if not result.success or result.fun is None:
        raise RuntimeError(f"ABSTAIN_PROXY_CONDITIONAL_LP: {result.message}")
    return float(-result.fun)


def _conditional_radius(
    proxy_empirical: np.ndarray,
    confusion: np.ndarray,
    effective_radius: float,
    source: int,
    center: np.ndarray,
    source_mass_lower: float,
) -> tuple[float, int]:
    if source_mass_lower <= VALUE_GUARD:
        return 2.0, 0
    maximum = 0.0
    solves = 0
    for signs in product((-1.0, 1.0), repeat=center.size):
        sign = np.asarray(signs, dtype=np.float64)
        support = _conditional_support(
            proxy_empirical,
            confusion,
            effective_radius,
            source,
            sign,
            source_mass_lower,
        )
        solves += 1
        if support is None:
            return 2.0, solves
        maximum = max(maximum, support - float(sign @ center))
    return min(2.0, max(0.0, maximum) + VALUE_GUARD), solves


def _confusion_from_calibration(
    calibration_counts: np.ndarray,
    *,
    per_event_delta: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    if calibration_counts.ndim not in {2, 3, 4}:
        raise ValueError(
            "calibration counts need shape GxG, JxGxG, or JxGxKxG"
        )
    totals = calibration_counts.sum(axis=-1)
    if np.any(totals <= 0):
        raise ValueError("every calibrated true-source row must be nonempty")
    empirical = calibration_counts / totals[..., None]
    source_count = calibration_counts.shape[-1]
    radii = np.vectorize(
        lambda n: weissman_l1_radius(int(n), source_count, per_event_delta)
    )(totals)
    return empirical, radii, int(calibration_counts.sum())


def _binary_symmetric_confusion_from_calibration(
    calibration_counts: np.ndarray,
    *,
    failure_probability: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    if calibration_counts.shape != (2, 2):
        raise ValueError("binary-symmetric calibration requires a 2x2 table")
    sample_size = int(calibration_counts.sum())
    if sample_size <= 0:
        raise ValueError("binary-symmetric calibration must be nonempty")
    errors = int(calibration_counts[0, 1] + calibration_counts[1, 0])
    error_rate = errors / sample_size
    alpha = float(failure_probability)
    lower = (
        0.0
        if errors == 0
        else float(beta.ppf(alpha / 2.0, errors, sample_size - errors + 1))
    )
    upper = (
        1.0
        if errors == sample_size
        else float(
            beta.ppf(
                1.0 - alpha / 2.0,
                errors + 1,
                sample_size - errors,
            )
        )
    )
    row_l1_radius = 2.0 * max(error_rate - lower, upper - error_rate)
    confusion = np.asarray(
        [[1.0 - error_rate, error_rate], [error_rate, 1.0 - error_rate]],
        dtype=np.float64,
    )
    return confusion, np.full(2, row_l1_radius), sample_size


def _clopper_pearson_interval(
    successes: int,
    trials: int,
    failure_probability: float,
) -> tuple[float, float]:
    lower = (
        0.0
        if successes == 0
        else float(
            beta.ppf(
                failure_probability / 2.0,
                successes,
                trials - successes + 1,
            )
        )
    )
    upper = (
        1.0
        if successes == trials
        else float(
            beta.ppf(
                1.0 - failure_probability / 2.0,
                successes + 1,
                trials - successes,
            )
        )
    )
    return lower, upper


def _proxy_coordinate_intervals(
    counts: np.ndarray,
    *,
    table_failure_probability: float,
    confusion_row_l1_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    sample_size = int(counts.sum())
    coordinate_delta = table_failure_probability / counts.size
    lower = np.zeros_like(counts, dtype=np.float64)
    upper = np.zeros_like(counts, dtype=np.float64)
    expansion = confusion_row_l1_radius / 2.0
    for index in np.ndindex(counts.shape):
        low, high = _clopper_pearson_interval(
            int(counts[index]), sample_size, coordinate_delta
        )
        lower[index] = max(0.0, low - expansion)
        upper[index] = min(1.0, high + expansion)
    return lower, upper


def _box_nominal_joint(
    confusion: np.ndarray,
    coordinate_lowers: np.ndarray,
    coordinate_uppers: np.ndarray,
) -> np.ndarray:
    source_count, token_count = coordinate_lowers.shape
    latent_count = source_count * token_count
    observation = _observation_matrix(confusion, token_count)
    a_ub = np.vstack((observation, -observation))
    b_ub = np.concatenate(
        (coordinate_uppers.reshape(-1), -coordinate_lowers.reshape(-1))
    )
    result = linprog(
        np.zeros(latent_count),
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=np.ones((1, latent_count)),
        b_eq=np.asarray([1.0]),
        bounds=[(0.0, 1.0)] * latent_count,
        method="highs",
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"ABSTAIN_EMPTY_PROXY_COORDINATE_POLYTOPE: {result.message}")
    joint = np.clip(result.x, 0.0, 1.0)
    joint /= joint.sum()
    return joint.reshape(source_count, token_count)


def _box_source_mass_lower_bound(
    confusion: np.ndarray,
    coordinate_lowers: np.ndarray,
    coordinate_uppers: np.ndarray,
    source: int,
) -> float:
    source_count, token_count = coordinate_lowers.shape
    latent_count = source_count * token_count
    observation = _observation_matrix(confusion, token_count)
    objective = np.zeros(latent_count)
    objective[source * token_count : (source + 1) * token_count] = 1.0
    result = linprog(
        objective,
        A_ub=np.vstack((observation, -observation)),
        b_ub=np.concatenate(
            (coordinate_uppers.reshape(-1), -coordinate_lowers.reshape(-1))
        ),
        A_eq=np.ones((1, latent_count)),
        b_eq=np.asarray([1.0]),
        bounds=[(0.0, 1.0)] * latent_count,
        method="highs",
    )
    if not result.success or result.fun is None:
        raise RuntimeError(f"ABSTAIN_PROXY_COORDINATE_SOURCE_MASS_LP: {result.message}")
    return max(0.0, float(result.fun) - VALUE_GUARD)


def _box_conditional_support(
    confusion: np.ndarray,
    coordinate_lowers: np.ndarray,
    coordinate_uppers: np.ndarray,
    source: int,
    sign: np.ndarray,
    source_mass_lower: float,
) -> float:
    source_count, token_count = coordinate_lowers.shape
    latent_count = source_count * token_count
    observation = _observation_matrix(confusion, token_count)
    t_index = latent_count
    objective = np.zeros(latent_count + 1)
    objective[source * token_count : (source + 1) * token_count] = -sign
    a_upper = np.hstack(
        (observation, -coordinate_uppers.reshape(-1, 1))
    )
    a_lower = np.hstack(
        (-observation, coordinate_lowers.reshape(-1, 1))
    )
    a_eq = np.zeros((2, latent_count + 1))
    a_eq[0, :latent_count] = 1.0
    a_eq[0, t_index] = -1.0
    a_eq[
        1, source * token_count : (source + 1) * token_count
    ] = 1.0
    result = linprog(
        objective,
        A_ub=np.vstack((a_upper, a_lower)),
        b_ub=np.zeros(2 * latent_count),
        A_eq=a_eq,
        b_eq=np.asarray([0.0, 1.0]),
        bounds=[(0.0, None)] * latent_count
        + [(0.0, 1.0 / source_mass_lower)],
        method="highs",
    )
    if not result.success or result.fun is None:
        raise RuntimeError(f"ABSTAIN_PROXY_COORDINATE_CONDITIONAL_LP: {result.message}")
    return float(-result.fun)


def _box_conditional_radius(
    confusion: np.ndarray,
    coordinate_lowers: np.ndarray,
    coordinate_uppers: np.ndarray,
    source: int,
    center: np.ndarray,
    source_mass_lower: float,
) -> tuple[float, int]:
    if source_mass_lower <= VALUE_GUARD:
        return 2.0, 0
    maximum = 0.0
    solves = 0
    for signs in product((-1.0, 1.0), repeat=center.size):
        sign = np.asarray(signs, dtype=np.float64)
        support = _box_conditional_support(
            confusion,
            coordinate_lowers,
            coordinate_uppers,
            source,
            sign,
            source_mass_lower,
        )
        solves += 1
        maximum = max(maximum, support - float(sign @ center))
    return min(2.0, max(0.0, maximum) + VALUE_GUARD), solves


def _conditional_observation_matrix(
    confusion: np.ndarray,
    source_masses: np.ndarray,
    token_count: int,
) -> np.ndarray:
    source_count = confusion.shape[0]
    matrix = np.zeros(
        (source_count * token_count, source_count * token_count),
        dtype=np.float64,
    )
    for proxy_source in range(source_count):
        for token in range(token_count):
            observed = proxy_source * token_count + token
            for source in range(source_count):
                conditional = source * token_count + token
                matrix[observed, conditional] = (
                    (
                        confusion[source, proxy_source]
                        if confusion.ndim == 2
                        else confusion[source, token, proxy_source]
                    )
                    * source_masses[source]
                )
    return matrix


def _fixed_mass_l1_problem(
    proxy_empirical: np.ndarray,
    confusion: np.ndarray,
    source_masses: np.ndarray,
    radius: float,
    *,
    objective: np.ndarray | None,
    constrain_radius: bool,
) -> tuple[np.ndarray, float]:
    source_count, token_count = proxy_empirical.shape
    conditional_count = source_count * token_count
    observed_count = conditional_count
    observation = _conditional_observation_matrix(
        confusion, source_masses, token_count
    )
    proxy_flat = proxy_empirical.reshape(-1)
    variable_count = conditional_count + observed_count
    costs = np.zeros(variable_count)
    if objective is None:
        costs[conditional_count:] = 1.0
    else:
        costs[:conditional_count] = objective
    a_ub = []
    b_ub = []
    for index in range(observed_count):
        positive = np.zeros(variable_count)
        positive[:conditional_count] = observation[index]
        positive[conditional_count + index] = -1.0
        a_ub.append(positive)
        b_ub.append(float(proxy_flat[index]))
        negative = np.zeros(variable_count)
        negative[:conditional_count] = -observation[index]
        negative[conditional_count + index] = -1.0
        a_ub.append(negative)
        b_ub.append(float(-proxy_flat[index]))
    if constrain_radius:
        radius_row = np.zeros(variable_count)
        radius_row[conditional_count:] = 1.0
        a_ub.append(radius_row)
        b_ub.append(float(radius))
    a_eq = np.zeros((source_count, variable_count))
    for source in range(source_count):
        a_eq[
            source,
            source * token_count : (source + 1) * token_count,
        ] = 1.0
    result = linprog(
        costs,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        A_eq=a_eq,
        b_eq=np.ones(source_count),
        bounds=[(0.0, 1.0)] * conditional_count
        + [(0.0, None)] * observed_count,
        method="highs",
    )
    if not result.success or result.x is None or result.fun is None:
        raise RuntimeError(f"ABSTAIN_PROXY_FIXED_MASS_LP: {result.message}")
    return np.asarray(result.x[:conditional_count]), float(result.fun)


def _fixed_mass_box_problem(
    confusion: np.ndarray,
    source_masses: np.ndarray,
    coordinate_lowers: np.ndarray,
    coordinate_uppers: np.ndarray,
    *,
    objective: np.ndarray,
) -> tuple[np.ndarray, float]:
    source_count, token_count = coordinate_lowers.shape
    conditional_count = source_count * token_count
    observation = _conditional_observation_matrix(
        confusion, source_masses, token_count
    )
    a_eq = np.zeros((source_count, conditional_count))
    for source in range(source_count):
        a_eq[
            source,
            source * token_count : (source + 1) * token_count,
        ] = 1.0
    result = linprog(
        objective,
        A_ub=np.vstack((observation, -observation)),
        b_ub=np.concatenate(
            (coordinate_uppers.reshape(-1), -coordinate_lowers.reshape(-1))
        ),
        A_eq=a_eq,
        b_eq=np.ones(source_count),
        bounds=[(0.0, 1.0)] * conditional_count,
        method="highs",
    )
    if not result.success or result.x is None or result.fun is None:
        raise RuntimeError(f"ABSTAIN_PROXY_FIXED_MASS_COORDINATE_LP: {result.message}")
    return np.asarray(result.x), float(result.fun)


def _fixed_mass_conditional_certificate(
    proxy_empirical: np.ndarray,
    confusion: np.ndarray,
    source_masses: np.ndarray,
    effective_radius: float,
    coordinate_lowers: np.ndarray | None,
    coordinate_uppers: np.ndarray | None,
) -> tuple[np.ndarray, tuple[float, ...], int]:
    source_count, token_count = proxy_empirical.shape
    conditional_count = source_count * token_count
    if coordinate_lowers is None:
        flat_center, residual = _fixed_mass_l1_problem(
            proxy_empirical,
            confusion,
            source_masses,
            effective_radius,
            objective=None,
            constrain_radius=False,
        )
        if residual > effective_radius + LP_TOLERANCE:
            raise RuntimeError(
                "ABSTAIN_EMPTY_PROXY_FIXED_MASS_POLYTOPE: "
                f"minimum residual {residual} exceeds radius {effective_radius}"
            )
    else:
        assert coordinate_uppers is not None
        flat_center, _ = _fixed_mass_box_problem(
            confusion,
            source_masses,
            coordinate_lowers,
            coordinate_uppers,
            objective=np.zeros(conditional_count),
        )
    centers = flat_center.reshape(source_count, token_count)
    radii = []
    solves = 0
    for source in range(source_count):
        maximum = 0.0
        for signs in product((-1.0, 1.0), repeat=token_count):
            sign = np.asarray(signs, dtype=np.float64)
            objective = np.zeros(conditional_count)
            objective[
                source * token_count : (source + 1) * token_count
            ] = -sign
            if coordinate_lowers is None:
                _, value = _fixed_mass_l1_problem(
                    proxy_empirical,
                    confusion,
                    source_masses,
                    effective_radius,
                    objective=objective,
                    constrain_radius=True,
                )
            else:
                assert coordinate_uppers is not None
                _, value = _fixed_mass_box_problem(
                    confusion,
                    source_masses,
                    coordinate_lowers,
                    coordinate_uppers,
                    objective=objective,
                )
            support = -value
            maximum = max(maximum, support - float(sign @ centers[source]))
            solves += 1
        radii.append(min(2.0, max(0.0, maximum) + VALUE_GUARD))
    return centers, tuple(radii), solves


def certify_proxy_label_conditionals(
    proxy_joint_counts: Sequence[Sequence[Sequence[int]]],
    *,
    family_failure_probability: float,
    known_confusion_matrix: Sequence | None = None,
    calibration_confusion_counts: Sequence | None = None,
    binary_symmetric_calibration: bool = False,
    confidence_region: str = "l1_weissman",
    known_source_masses: Sequence | None = None,
    observation_model_l1_slack: float | Sequence[float] = 0.0,
) -> ProxyLabelBridgeCertificate:
    """Certify true source-conditional token laws from proxy-labeled counts.

    Exactly one of ``known_confusion_matrix`` and
    ``calibration_confusion_counts`` must be supplied. Calibration counts may
    be pooled across task labels (G x G) or label-specific (J x G x G).
    """

    counts = _validate_counts(
        proxy_joint_counts, ndim=3, name="proxy_joint_counts"
    )
    label_count, source_count, token_count = counts.shape
    if min(counts.shape) < 2:
        raise ValueError("at least two labels, sources, and tokens are required")
    if np.any(counts.sum(axis=(1, 2)) <= 0):
        raise ValueError("every task label needs proxy observations")
    if not 0.0 < family_failure_probability < 1.0:
        raise ValueError("family_failure_probability must lie in (0,1)")
    if (known_confusion_matrix is None) == (calibration_confusion_counts is None):
        raise ValueError("provide exactly one confusion source")
    if binary_symmetric_calibration and calibration_confusion_counts is None:
        raise ValueError("binary_symmetric_calibration requires calibration counts")
    if confidence_region not in {"coordinate_clopper_pearson", "l1_weissman"}:
        raise ValueError("unknown confidence_region")
    raw_slack = np.asarray(observation_model_l1_slack, dtype=np.float64)
    if raw_slack.ndim == 0:
        model_slack = np.full(label_count, float(raw_slack))
    elif raw_slack.shape == (label_count,):
        model_slack = raw_slack
    else:
        raise ValueError(
            "observation_model_l1_slack must be scalar or one value per label"
        )
    if (
        not np.isfinite(model_slack).all()
        or np.any(model_slack < 0.0)
        or np.any(model_slack > 2.0)
    ):
        raise ValueError("observation-model L1 slack must lie in [0, 2]")
    fixed_masses = None
    if known_source_masses is not None:
        raw_masses = np.asarray(known_source_masses, dtype=np.float64)
        if raw_masses.ndim == 1 and raw_masses.shape == (source_count,):
            fixed_masses = np.stack([raw_masses for _ in range(label_count)])
        elif raw_masses.shape == (label_count, source_count):
            fixed_masses = raw_masses
        else:
            raise ValueError("known_source_masses must have shape G or JxG")
        if (
            not np.isfinite(fixed_masses).all()
            or np.any(fixed_masses <= 0.0)
            or not np.allclose(fixed_masses.sum(axis=1), 1.0, atol=1e-10)
        ):
            raise ValueError("known source masses must be positive probability rows")

    calibration_sample_size = 0
    if known_confusion_matrix is not None:
        raw = np.asarray(known_confusion_matrix, dtype=np.float64)
        event_count = label_count
        per_event_delta = family_failure_probability / event_count
        if raw.ndim == 2:
            matrices = np.stack(
                [
                    _validate_confusion(raw, source_count, token_count)
                    for _ in range(label_count)
                ]
            )
        elif raw.ndim == 3 and raw.shape[0] == label_count:
            matrices = np.stack(
                [
                    _validate_confusion(
                        raw[label], source_count, token_count
                    )
                    for label in range(label_count)
                ]
            )
        elif raw.shape == (
            label_count,
            source_count,
            token_count,
            source_count,
        ):
            matrices = np.stack(
                [
                    _validate_confusion(
                        raw[label], source_count, token_count
                    )
                    for label in range(label_count)
                ]
            )
        else:
            raise ValueError(
                "known confusion must have shape GxG, JxGxG, or JxGxKxG"
            )
        confusion_radii = np.zeros(matrices.shape[:-1], dtype=np.float64)
        calibration_mode = "known_confusion_matrix"
    else:
        calibration = _validate_counts(
            calibration_confusion_counts,
            ndim=np.asarray(calibration_confusion_counts).ndim,
            name="calibration_confusion_counts",
        )
        if binary_symmetric_calibration:
            if source_count != 2 or calibration.ndim != 2:
                raise ValueError(
                    "binary-symmetric calibration requires two sources and one pooled table"
                )
            event_count = label_count + 1
            per_event_delta = family_failure_probability / event_count
            matrix, row_radii, calibration_sample_size = (
                _binary_symmetric_confusion_from_calibration(
                    calibration, failure_probability=per_event_delta
                )
            )
            matrices = np.stack([matrix for _ in range(label_count)])
            confusion_radii = np.stack([row_radii for _ in range(label_count)])
            calibration_mode = "pooled_binary_symmetric_exact_binomial_calibration"
        elif calibration.ndim == 2:
            if calibration.shape != (source_count, source_count):
                raise ValueError("pooled calibration must have shape GxG")
            event_count = label_count + source_count
            per_event_delta = family_failure_probability / event_count
            matrix, row_radii, calibration_sample_size = _confusion_from_calibration(
                calibration, per_event_delta=per_event_delta
            )
            matrices = np.stack([matrix for _ in range(label_count)])
            confusion_radii = np.stack([row_radii for _ in range(label_count)])
            calibration_mode = "pooled_confusion_calibration"
        elif calibration.ndim == 3:
            if calibration.shape != (label_count, source_count, source_count):
                raise ValueError("label-specific calibration must have shape JxGxG")
            event_count = label_count + label_count * source_count
            per_event_delta = family_failure_probability / event_count
            matrices, confusion_radii, calibration_sample_size = (
                _confusion_from_calibration(
                    calibration, per_event_delta=per_event_delta
                )
            )
            calibration_mode = "label_specific_confusion_calibration"
        elif calibration.ndim == 4:
            if calibration.shape != (
                label_count,
                source_count,
                token_count,
                source_count,
            ):
                raise ValueError(
                    "token-dependent calibration must have shape JxGxKxG"
                )
            event_count = (
                label_count + label_count * source_count * token_count
            )
            per_event_delta = family_failure_probability / event_count
            matrices, confusion_radii, calibration_sample_size = (
                _confusion_from_calibration(
                    calibration, per_event_delta=per_event_delta
                )
            )
            calibration_mode = (
                "label_and_token_specific_confusion_calibration"
            )
        else:
            raise ValueError(
                "calibration counts need shape GxG, JxGxG, or JxGxKxG"
            )

    labels = []
    conditional_centers = np.zeros((label_count, source_count, token_count))
    conditional_radii = np.zeros((label_count, source_count))
    for label in range(label_count):
        sample_size = int(counts[label].sum())
        proxy_empirical = counts[label] / sample_size
        proxy_radius = weissman_l1_radius(
            sample_size, source_count * token_count, per_event_delta
        )
        effective_radius = min(
            2.0,
            proxy_radius
            + float(np.max(confusion_radii[label]))
            + float(model_slack[label]),
        )
        coordinate_lowers = None
        coordinate_uppers = None
        if confidence_region == "coordinate_clopper_pearson":
            coordinate_lowers, coordinate_uppers = _proxy_coordinate_intervals(
                counts[label],
                table_failure_probability=per_event_delta,
                confusion_row_l1_radius=(
                    float(np.max(confusion_radii[label]))
                    + float(model_slack[label])
                ),
            )
            if fixed_masses is None:
                nominal = _box_nominal_joint(
                    matrices[label], coordinate_lowers, coordinate_uppers
                )
        else:
            if fixed_masses is None:
                nominal = _nominal_joint(
                    proxy_empirical,
                    matrices[label],
                    effective_radius,
                )
        if fixed_masses is not None:
            centers, radius_tuple, solves = _fixed_mass_conditional_certificate(
                proxy_empirical,
                matrices[label],
                fixed_masses[label],
                effective_radius,
                coordinate_lowers,
                coordinate_uppers,
            )
            radii = list(radius_tuple)
            masses = np.asarray(fixed_masses[label])
            lower_bounds = list(masses)
            nominal = masses[:, None] * centers
        else:
            masses = nominal.sum(axis=1)
            centers = np.zeros_like(nominal)
            radii = []
            lower_bounds = []
            solves = 0
            for source in range(source_count):
                if masses[source] <= VALUE_GUARD:
                    centers[source] = np.full(token_count, 1.0 / token_count)
                else:
                    centers[source] = nominal[source] / masses[source]
                if confidence_region == "coordinate_clopper_pearson":
                    assert coordinate_lowers is not None
                    assert coordinate_uppers is not None
                    lower = _box_source_mass_lower_bound(
                        matrices[label],
                        coordinate_lowers,
                        coordinate_uppers,
                        source,
                    )
                    radius, count = _box_conditional_radius(
                        matrices[label],
                        coordinate_lowers,
                        coordinate_uppers,
                        source,
                        centers[source],
                        lower,
                    )
                else:
                    lower = _source_mass_lower_bound(
                        proxy_empirical,
                        matrices[label],
                        effective_radius,
                        source,
                    )
                    radius, count = _conditional_radius(
                        proxy_empirical,
                        matrices[label],
                        effective_radius,
                        source,
                        centers[source],
                        lower,
                    )
                lower_bounds.append(lower)
                radii.append(radius)
                solves += count
        conditional_centers[label] = centers
        conditional_radii[label] = radii
        labels.append(
            ProxyLabelConditionalCertificate(
                label=label,
                proxy_empirical_joint=np.asarray(proxy_empirical),
                proxy_l1_radius=float(proxy_radius),
                proxy_coordinate_lowers=(
                    None
                    if coordinate_lowers is None
                    else np.asarray(coordinate_lowers)
                ),
                proxy_coordinate_uppers=(
                    None
                    if coordinate_uppers is None
                    else np.asarray(coordinate_uppers)
                ),
                confusion_matrix=np.asarray(matrices[label]),
                confusion_row_l1_radii=tuple(
                    float(value)
                    for value in np.ravel(confusion_radii[label])
                ),
                observation_model_l1_slack=float(model_slack[label]),
                effective_proxy_l1_radius=float(effective_radius),
                nominal_true_joint=np.asarray(nominal),
                conditional_centers=np.asarray(centers),
                conditional_l1_radii=tuple(float(value) for value in radii),
                source_mass_lower_bounds=tuple(float(value) for value in lower_bounds),
                source_mass_centers=tuple(float(value) for value in masses),
                conditional_lp_solves=solves,
                confidence_region=confidence_region,
                method=(
                    "exact_proxy_fixed_mass_conditional_l1_lp"
                    if fixed_masses is not None
                    else (
                        "exact_proxy_coordinate_polytope_conditional_l1_linear_fractional_lp"
                        if confidence_region == "coordinate_clopper_pearson"
                        else "exact_proxy_l1_polytope_conditional_l1_linear_fractional_lp"
                    )
                ),
            )
        )
    return ProxyLabelBridgeCertificate(
        labels=tuple(labels),
        conditional_empirical_distributions=conditional_centers,
        conditional_l1_radii=conditional_radii,
        family_failure_probability=float(family_failure_probability),
        per_event_failure_probability=float(per_event_delta),
        calibration_mode=calibration_mode,
        label_count=label_count,
        source_count=source_count,
        token_count=token_count,
        proxy_sample_size=int(counts.sum()),
        calibration_sample_size=calibration_sample_size,
        method="simultaneous_proxy_label_target_conditionals_with_exact_lp_envelopes",
    )
