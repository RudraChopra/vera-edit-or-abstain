"""Anytime-valid confidence sequences for categorical laws."""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma, log, sqrt
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class AnytimeMultinomialRegion:
    """An L1 outer approximation to a Dirichlet-mixture confidence sequence."""

    counts: tuple[int, ...]
    empirical_distribution: tuple[float, ...]
    sample_size: int
    failure_probability: float
    dirichlet_prior: float
    log_e_value_at_empirical: float
    kl_radius: float
    l1_radius: float
    method: str = "dirichlet_mixture_e_process_pinsker_outer_region"


def _validate_counts(counts: Sequence[int]) -> np.ndarray:
    values = np.asarray(tuple(counts), dtype=np.int64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("counts must contain at least two categories")
    if np.any(values < 0):
        raise ValueError("counts must be non-negative")
    return values


def _log_multivariate_beta(values: np.ndarray) -> float:
    return float(sum(lgamma(float(value)) for value in values) - lgamma(float(values.sum())))


def dirichlet_mixture_log_e_value(
    counts: Sequence[int],
    candidate_distribution: Sequence[float],
    *,
    prior: float = 0.5,
) -> float:
    """Return the log mixture likelihood ratio against ``candidate_distribution``."""

    values = _validate_counts(counts)
    candidate = np.asarray(tuple(candidate_distribution), dtype=np.float64)
    if candidate.shape != values.shape:
        raise ValueError("candidate_distribution must match counts")
    if prior <= 0.0:
        raise ValueError("prior must be positive")
    if np.any(candidate < 0.0) or not np.isclose(candidate.sum(), 1.0):
        raise ValueError("candidate_distribution must be a probability vector")
    positive = values > 0
    if np.any(candidate[positive] <= 0.0):
        return float("inf")
    log_mixture = _log_multivariate_beta(
        values.astype(np.float64) + prior
    ) - _log_multivariate_beta(
        np.full(values.size, prior, dtype=np.float64)
    )
    log_candidate = float(
        np.dot(values[positive], np.log(candidate[positive]))
    )
    return log_mixture - log_candidate


def anytime_multinomial_region(
    counts: Sequence[int],
    *,
    failure_probability: float,
    prior: float = 0.5,
) -> AnytimeMultinomialRegion:
    """Build the exact-KL/Pinsker outer region at one optional stopping time."""

    values = _validate_counts(counts)
    if not 0.0 < failure_probability < 1.0:
        raise ValueError("failure_probability must lie in (0, 1)")
    if prior <= 0.0:
        raise ValueError("prior must be positive")
    sample_size = int(values.sum())
    if sample_size == 0:
        empirical = np.full(values.size, 1.0 / values.size)
        log_e_empirical = 0.0
        kl_radius = float("inf")
        l1_radius = 2.0
    else:
        empirical = values / sample_size
        log_e_empirical = dirichlet_mixture_log_e_value(
            values, empirical, prior=prior
        )
        kl_radius = max(
            0.0,
            (log(1.0 / failure_probability) - log_e_empirical) / sample_size,
        )
        l1_radius = min(2.0, sqrt(2.0 * kl_radius))
    return AnytimeMultinomialRegion(
        counts=tuple(int(value) for value in values),
        empirical_distribution=tuple(float(value) for value in empirical),
        sample_size=sample_size,
        failure_probability=float(failure_probability),
        dirichlet_prior=float(prior),
        log_e_value_at_empirical=float(log_e_empirical),
        kl_radius=float(kl_radius),
        l1_radius=float(l1_radius),
    )


def simultaneous_anytime_regions(
    counts_by_stratum: Sequence[Sequence[int]],
    *,
    failure_probability: float,
    prior: float = 0.5,
) -> tuple[AnytimeMultinomialRegion, ...]:
    """Union-bound registered strata while remaining valid at every time."""

    counts = np.asarray(counts_by_stratum, dtype=np.int64)
    if counts.ndim != 2 or counts.shape[0] < 1 or counts.shape[1] < 2:
        raise ValueError("counts_by_stratum must be a nonempty matrix")
    if not 0.0 < failure_probability < 1.0:
        raise ValueError("failure_probability must lie in (0, 1)")
    per_stratum = failure_probability / counts.shape[0]
    return tuple(
        anytime_multinomial_region(
            row, failure_probability=per_stratum, prior=prior
        )
        for row in counts
    )
