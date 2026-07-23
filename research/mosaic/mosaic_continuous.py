"""Covering-number certificates for continuous release-channel classes."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Sequence

import numpy as np

from mosaic_channel import (
    ChannelAttackerEnvelope,
    adaptive_channel_attacker_confidence_bound,
    l1_ball_expectation_upper,
)


@dataclass(frozen=True)
class ContinuousReleaseCertificate:
    """Universal-attacker bound for a covered continuous channel class."""

    attacker: ChannelAttackerEnvelope
    per_source_l1_radii: tuple[float, ...]
    cover_size: int
    approximation_l1: float
    output_count: int
    method: str = "finite_l1_cover_coordinate_hoeffding"


@dataclass(frozen=True)
class ThresholdClassCertificate:
    """Uniform certificate for a post-selected threshold on a real score."""

    threshold: float
    empirical_binary_laws: np.ndarray
    per_source_l1_radii: tuple[float, ...]
    attacker: ChannelAttackerEnvelope
    method: str = "dkw_uniform_over_all_real_thresholds"


def dkw_threshold_l1_radius(
    sample_size: int, failure_probability: float
) -> float:
    """L1 radius valid simultaneously for every threshold on a real score.

    DKW gives ``sup_t |F_n(t)-F(t)| <= eps``.  A threshold induces a
    Bernoulli law, whose L1 error is twice the CDF error.
    """

    if sample_size <= 0:
        return 2.0
    if not 0.0 < failure_probability < 1.0:
        raise ValueError("failure_probability must lie in (0, 1)")
    epsilon = sqrt(log(2.0 / failure_probability) / (2.0 * sample_size))
    return float(min(2.0, 2.0 * epsilon))


def threshold_class_attacker_certificate(
    scores_by_source: Sequence[Sequence[float]],
    *,
    threshold: float,
    failure_probabilities: Sequence[float],
) -> ThresholdClassCertificate:
    """Certify a threshold selected after seeing the registered score samples.

    The event is uniform over every real threshold, so ``threshold`` may be
    selected from the same samples without another finite-grid correction.
    """

    if len(scores_by_source) < 2:
        raise ValueError("at least two source samples are required")
    if len(failure_probabilities) != len(scores_by_source):
        raise ValueError("failure_probabilities must match the source samples")
    rows = []
    radii = []
    for values, delta in zip(
        scores_by_source, failure_probabilities, strict=True
    ):
        scores = np.asarray(tuple(values), dtype=np.float64)
        if scores.ndim != 1 or scores.size == 0:
            raise ValueError("each source sample must be a nonempty vector")
        if not np.isfinite(scores).all():
            raise ValueError("scores must be finite")
        probability = float(np.mean(scores <= threshold))
        rows.append((probability, 1.0 - probability))
        radii.append(
            dkw_threshold_l1_radius(scores.size, float(delta))
        )
    empirical = np.asarray(rows, dtype=np.float64)
    attacker = adaptive_channel_attacker_confidence_bound(
        empirical,
        np.eye(2, dtype=np.float64),
        l1_radii=radii,
    )
    return ThresholdClassCertificate(
        threshold=float(threshold),
        empirical_binary_laws=empirical,
        per_source_l1_radii=tuple(radii),
        attacker=attacker,
    )


def covered_output_l1_radius(
    *,
    sample_size: int,
    output_count: int,
    cover_size: int,
    failure_probability: float,
    approximation_l1: float,
) -> float:
    """Uniform L1 radius for a selected member of a continuous channel class.

    A registered finite epsilon-net has ``cover_size`` elements under
    ``sup_x ||q(x)-q_net(x)||_1 <= approximation_l1``.  Coordinate Hoeffding
    bounds are unioned over the net and output coordinates.  Moving from the
    selected channel to its net representative and back costs
    ``2 * approximation_l1``.
    """

    if sample_size <= 0:
        return 2.0
    if output_count < 2 or cover_size < 1:
        raise ValueError("output_count must be >=2 and cover_size must be >=1")
    if not 0.0 < failure_probability < 1.0:
        raise ValueError("failure_probability must lie in (0, 1)")
    if not 0.0 <= approximation_l1 <= 2.0:
        raise ValueError("approximation_l1 must lie in [0, 2]")
    coordinate_radius = sqrt(
        log(2.0 * output_count * cover_size / failure_probability)
        / (2.0 * sample_size)
    )
    return float(
        min(2.0, output_count * coordinate_radius + 2.0 * approximation_l1)
    )


def continuous_release_attacker_certificate(
    empirical_selected_output_laws: Sequence[Sequence[float]],
    *,
    sample_sizes: Sequence[int],
    cover_size: int,
    failure_probabilities: Sequence[float],
    approximation_l1: float,
) -> ContinuousReleaseCertificate:
    """Certify a data-selected continuous release channel through its cover."""

    empirical = np.asarray(empirical_selected_output_laws, dtype=np.float64)
    if empirical.ndim != 2 or empirical.shape[0] < 2 or empirical.shape[1] < 2:
        raise ValueError("at least two source laws and two outputs are required")
    if not np.isfinite(empirical).all() or np.any(empirical < -1e-12):
        raise ValueError("empirical output laws must be finite and non-negative")
    if not np.allclose(empirical.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("each empirical output law must sum to one")
    if len(sample_sizes) != empirical.shape[0]:
        raise ValueError("sample_sizes must have one entry per source")
    if len(failure_probabilities) != empirical.shape[0]:
        raise ValueError("failure_probabilities must have one entry per source")

    radii = tuple(
        covered_output_l1_radius(
            sample_size=int(sample_size),
            output_count=empirical.shape[1],
            cover_size=cover_size,
            failure_probability=float(delta),
            approximation_l1=approximation_l1,
        )
        for sample_size, delta in zip(
            sample_sizes, failure_probabilities, strict=True
        )
    )
    identity = np.eye(empirical.shape[1], dtype=np.float64)
    attacker = adaptive_channel_attacker_confidence_bound(
        empirical,
        identity,
        l1_radii=radii,
    )
    return ContinuousReleaseCertificate(
        attacker=attacker,
        per_source_l1_radii=radii,
        cover_size=int(cover_size),
        approximation_l1=float(approximation_l1),
        output_count=empirical.shape[1],
    )


def continuous_decoder_error_bound(
    empirical_output_law: Sequence[float],
    decoder_losses: Sequence[float],
    *,
    sample_size: int,
    cover_size: int,
    failure_probability: float,
    approximation_l1: float,
) -> float:
    """Upper-bound selected-channel decoder loss under the same cover event."""

    empirical = np.asarray(tuple(empirical_output_law), dtype=np.float64)
    losses = np.asarray(tuple(decoder_losses), dtype=np.float64)
    if empirical.shape != losses.shape or empirical.ndim != 1:
        raise ValueError("empirical law and losses must be matching vectors")
    if np.any(losses < 0.0) or np.any(losses > 1.0):
        raise ValueError("decoder losses must lie in [0, 1]")
    radius = covered_output_l1_radius(
        sample_size=sample_size,
        output_count=empirical.size,
        cover_size=cover_size,
        failure_probability=failure_probability,
        approximation_l1=approximation_l1,
    )
    return l1_ball_expectation_upper(
        empirical, losses, l1_radius=radius
    )
