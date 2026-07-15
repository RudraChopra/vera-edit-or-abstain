"""Controlled supported shifts and prospective certification-data allocation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, log, sqrt
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ControlledShift:
    requested_gamma: float
    focus_environment: int
    focus_source: int
    focus_target: int
    focus_probability_within_environment: float
    nonfocus_weight: float
    global_density_ratio_cap: float
    target_profile: Mapping[str, float]
    source_profile: Mapping[str, float]
    reference_size: int
    design_size: int

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["target_profile"] = dict(self.target_profile)
        payload["source_profile"] = dict(self.source_profile)
        return payload


def _validate_metadata(
    environment: Sequence[int], source: Sequence[int], target: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = tuple(
        np.asarray(values, dtype=np.int64)
        for values in (environment, source, target)
    )
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("metadata arrays must be one-dimensional")
    if not arrays[0].size or len({len(array) for array in arrays}) != 1:
        raise ValueError("metadata arrays must be nonempty and aligned")
    return arrays


def conditional_density_ratio_profile(
    weights: Sequence[float],
    groups: Sequence[int],
) -> dict[str, float]:
    """Return exact conditional density-ratio caps induced by global weights."""

    weight = np.asarray(weights, dtype=np.float64)
    group = np.asarray(groups, dtype=np.int64)
    if weight.ndim != 1 or group.ndim != 1 or len(weight) != len(group):
        raise ValueError("weights and groups must be aligned vectors")
    if not len(weight) or not np.isfinite(weight).all() or np.any(weight < 0.0):
        raise ValueError("weights must be nonnegative and finite")
    profile: dict[str, float] = {}
    for value in sorted(map(int, np.unique(group))):
        local = weight[group == value]
        mean = float(local.mean())
        if mean <= 0.0:
            raise ValueError("deployment removes an entire required conditional cell")
        # A maximum is mathematically at least its cell mean. Preserve that
        # exact invariant when floating-point normalization lands just below 1.
        profile[str(value)] = max(1.0, float(local.max() / mean))
    return profile


def design_controlled_shift(
    environment: Sequence[int],
    source: Sequence[int],
    target: Sequence[int],
    design_indices: Sequence[int],
    *,
    requested_gamma: float,
    minimum_design_cell_count: int = 8,
) -> tuple[np.ndarray, ControlledShift]:
    """Choose an outcome-independent tail cell and construct an exact shift law.

    The design sample chooses the rarest supported environment-source-target
    cell with enough observations. Within its environment the selected cell is
    assigned density ratio ``requested_gamma`` and all other observations share
    the residual mass. Every other environment is unchanged.
    """

    env, src, tgt = _validate_metadata(environment, source, target)
    design = np.asarray(design_indices, dtype=np.int64)
    if design.ndim != 1 or not design.size:
        raise ValueError("design_indices must be a nonempty vector")
    if design.min() < 0 or design.max() >= len(env):
        raise ValueError("design index outside the reference population")
    if requested_gamma < 1.0 or not np.isfinite(requested_gamma):
        raise ValueError("requested_gamma must be finite and at least one")
    if minimum_design_cell_count <= 0:
        raise ValueError("minimum_design_cell_count must be positive")

    return design_controlled_shift_from_fold(
        env,
        src,
        tgt,
        env[design],
        src[design],
        tgt[design],
        requested_gamma=requested_gamma,
        minimum_design_cell_count=minimum_design_cell_count,
    )


def design_controlled_shift_from_fold(
    environment: Sequence[int],
    source: Sequence[int],
    target: Sequence[int],
    design_environment: Sequence[int],
    design_source: Sequence[int],
    design_target: Sequence[int],
    *,
    requested_gamma: float,
    minimum_design_cell_count: int = 8,
) -> tuple[np.ndarray, ControlledShift]:
    """Construct a supported shift using a separate metadata design fold."""

    env, src, tgt = _validate_metadata(environment, source, target)
    design_env, design_src, design_tgt = _validate_metadata(
        design_environment, design_source, design_target
    )
    if requested_gamma < 1.0 or not np.isfinite(requested_gamma):
        raise ValueError("requested_gamma must be finite and at least one")
    if minimum_design_cell_count <= 0:
        raise ValueError("minimum_design_cell_count must be positive")

    design_cells: list[tuple[float, int, int, int, int]] = []
    for environment_value in sorted(map(int, np.unique(design_env))):
        environment_design = design_env == environment_value
        environment_count = int(environment_design.sum())
        for source_value in sorted(map(int, np.unique(design_src[environment_design]))):
            for target_value in sorted(map(int, np.unique(design_tgt[environment_design]))):
                mask = (
                    environment_design
                    & (design_src == source_value)
                    & (design_tgt == target_value)
                )
                count = int(mask.sum())
                if count < minimum_design_cell_count:
                    continue
                full_environment = env == environment_value
                if not np.any(full_environment):
                    continue
                full_focus = (
                    full_environment
                    & (src == source_value)
                    & (tgt == target_value)
                )
                probability = float(full_focus.sum() / full_environment.sum())
                if probability < 1.0 and requested_gamma * probability <= 1.0 + 1e-12:
                    design_cells.append(
                        (
                            count / environment_count,
                            environment_value,
                            source_value,
                            target_value,
                            count,
                        )
                    )
    if not design_cells:
        raise ValueError("no design-supported cell can realize the requested shift")
    _, focus_environment, focus_source, focus_target, _ = min(design_cells)
    environment_mask = env == focus_environment
    focus_mask = (
        environment_mask & (src == focus_source) & (tgt == focus_target)
    )
    focus_probability = float(focus_mask.sum() / environment_mask.sum())
    nonfocus_weight = (
        (1.0 - focus_probability * requested_gamma)
        / (1.0 - focus_probability)
    )
    if nonfocus_weight < -1e-12:
        raise RuntimeError("constructed shift has negative residual weight")

    weights = np.ones(len(env), dtype=np.float64)
    weights[environment_mask & ~focus_mask] = max(0.0, nonfocus_weight)
    weights[focus_mask] = requested_gamma
    weights /= weights.mean()
    target_profile = conditional_density_ratio_profile(weights, env)
    source_profile = conditional_density_ratio_profile(weights, src)
    global_cap = float(weights.max())
    if global_cap > requested_gamma + 1e-10:
        raise RuntimeError("constructed law exceeds its requested global cap")
    if abs(float(weights.mean()) - 1.0) > 1e-10:
        raise RuntimeError("constructed density ratios do not integrate to one")

    return weights / weights.sum(), ControlledShift(
        requested_gamma=float(requested_gamma),
        focus_environment=int(focus_environment),
        focus_source=int(focus_source),
        focus_target=int(focus_target),
        focus_probability_within_environment=focus_probability,
        nonfocus_weight=float(max(0.0, nonfocus_weight)),
        global_density_ratio_cap=global_cap,
        target_profile=target_profile,
        source_profile=source_profile,
        reference_size=int(len(env)),
        design_size=int(len(design_env)),
    )


def allocate_integer_budget(
    scores: Mapping[str, float],
    *,
    total_budget: int,
    minimum_per_cell: int,
) -> dict[str, int]:
    """Allocate an exact integer budget proportionally to preregistered scores."""

    if not scores:
        raise ValueError("at least one allocation cell is required")
    if minimum_per_cell <= 0:
        raise ValueError("minimum_per_cell must be positive")
    if total_budget < minimum_per_cell * len(scores):
        raise ValueError("total_budget is smaller than the minimum allocation")
    normalized = {
        str(key): float(value)
        for key, value in scores.items()
    }
    if any(value < 0.0 or not np.isfinite(value) for value in normalized.values()):
        raise ValueError("allocation scores must be finite and nonnegative")
    if not any(value > 0.0 for value in normalized.values()):
        normalized = {key: 1.0 for key in normalized}
    remaining = total_budget - minimum_per_cell * len(normalized)
    score_sum = sum(normalized.values())
    ideals = {
        key: remaining * value / score_sum for key, value in normalized.items()
    }
    allocation = {
        key: minimum_per_cell + int(np.floor(value))
        for key, value in ideals.items()
    }
    leftover = total_budget - sum(allocation.values())
    order = sorted(
        normalized,
        key=lambda key: (-(ideals[key] - np.floor(ideals[key])), key),
    )
    for key in order[:leftover]:
        allocation[key] += 1
    return allocation


def dkw_sufficient_sample_size(
    *,
    robust_range: float,
    margin: float,
    coverage_error: float,
    power_error: float,
) -> int:
    """Sufficient per-cell sample size for certification with stated power."""

    if robust_range <= 0.0 or not np.isfinite(robust_range):
        raise ValueError("robust_range must be finite and positive")
    if margin <= 0.0 or not np.isfinite(margin):
        raise ValueError("margin must be finite and positive")
    if not 0.0 < coverage_error < 1.0 or not 0.0 < power_error < 1.0:
        raise ValueError("error probabilities must lie in (0, 1)")
    log_sum = sqrt(log(2.0 / coverage_error)) + sqrt(log(2.0 / power_error))
    return int(ceil((robust_range * log_sum) ** 2 / (2.0 * margin**2)))


def bernoulli_testing_lower_bound(
    *,
    safe_probability: float,
    unsafe_probability: float,
    type_one_error: float,
    type_two_error: float,
) -> int:
    """Bretagnolle-Huber lower bound for a safe/unsafe Bernoulli pair."""

    if not 0.0 < safe_probability < unsafe_probability < 1.0:
        raise ValueError("probabilities must satisfy 0 < safe < unsafe < 1")
    if not 0.0 < type_one_error < 1.0 or not 0.0 < type_two_error < 1.0:
        raise ValueError("error probabilities must lie in (0, 1)")
    divergence = (
        safe_probability * log(safe_probability / unsafe_probability)
        + (1.0 - safe_probability)
        * log((1.0 - safe_probability) / (1.0 - unsafe_probability))
    )
    numerator = log(1.0 / (2.0 * (type_one_error + type_two_error)))
    return max(0, int(ceil(numerator / divergence)))
