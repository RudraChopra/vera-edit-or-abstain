"""Corrected numerical bridge repair for MOSAIC's strict replay.

Version 1 added the feasibility guard to every denominator, including output
columns whose transform is identically zero.  Such a column represents the
exact inequality ``lower >= retained * 0`` and therefore places no upper bound
on retained mass.  Turning that exact zero into the guard incorrectly forced
retained mass to zero.  This module makes only that structural-zero correction;
all active-column guards and the release optimizer are unchanged.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from mosaic_bridge import (
    BridgeLabelCertificate,
    BridgeMembershipCertificate,
    certify_bridge_membership,
)
from mosaic_channel import l1_ball_expectation_lower, l1_ball_expectation_upper
from mosaic_strict_certification import (
    DEFAULT_FEASIBILITY_GUARD,
    DEFAULT_OPTIMIZATION_GUARD,
    DEFAULT_VALUE_GUARD,
    StrictReleaseCertificate,
    optimize_transform_exact_channel_strict,
)


def _validate_guard(value: float, *, name: str, maximum: float) -> float:
    guard = float(value)
    if not np.isfinite(guard) or not 0.0 < guard < maximum:
        raise ValueError(f"{name} must lie in (0, {maximum})")
    return guard


def _strict_label(
    label: BridgeLabelCertificate,
    reference: np.ndarray,
    reference_radii: np.ndarray,
    bridge: np.ndarray,
    bridge_radii: np.ndarray,
    *,
    feasibility_guard: float,
) -> BridgeLabelCertificate:
    token_count = reference.shape[1]
    indicators = np.eye(token_count, dtype=np.float64)
    lowers = np.asarray(
        [
            [
                l1_ball_expectation_lower(
                    bridge[source],
                    indicators[output],
                    l1_radius=float(bridge_radii[source]),
                )
                for output in range(token_count)
            ]
            for source in range(reference.shape[0])
        ],
        dtype=np.float64,
    )
    uppers = np.asarray(
        [
            [
                l1_ball_expectation_upper(
                    reference[source],
                    label.transform[:, output],
                    l1_radius=float(reference_radii[source]),
                )
                for output in range(token_count)
            ]
            for source in range(reference.shape[0])
        ],
        dtype=np.float64,
    )

    lower_safe = np.maximum(0.0, lowers - feasibility_guard)
    upper_safe = np.minimum(1.0, uppers + feasibility_guard)
    transform = np.asarray(label.transform, dtype=np.float64)
    active_outputs = np.any(transform != 0.0, axis=0)
    ratios = [float(label.retained_mass)]
    for source in range(reference.shape[0]):
        for output in range(token_count):
            if active_outputs[output] and upper_safe[source, output] > 0.0:
                ratios.append(
                    float(lower_safe[source, output] / upper_safe[source, output])
                )
    retained = max(0.0, min(ratios) - feasibility_guard)
    if retained > 0.0:
        retained = float(np.nextafter(retained, 0.0))

    # Recheck every original inequality, including the skipped structural zeros.
    slacks = lowers - retained * uppers
    minimum_slack = float(np.min(slacks))
    if minimum_slack < 0.0:
        raise RuntimeError(
            "strict v2 bridge repair failed to produce nonnegative membership slack"
        )
    return BridgeLabelCertificate(
        transform=transform,
        retained_mass=retained,
        contamination=1.0 - retained,
        optimal_retained_mass_upper=label.optimal_retained_mass_upper,
        reference_l1_radii=label.reference_l1_radii,
        bridge_l1_radii=label.bridge_l1_radii,
        bridge_coordinate_lowers=tuple(
            tuple(float(value) for value in row) for row in lowers
        ),
        minimum_membership_slack=minimum_slack,
        transform_trace=label.transform_trace,
        solver_status=label.solver_status,
        solver_iterations=label.solver_iterations,
        method="strict_v2_structural_zero_outward_repaired_l1_bridge",
    )


def certify_bridge_membership_strict(
    reference_empirical_distributions: Sequence[Sequence[Sequence[float]]],
    *,
    reference_l1_radii: Sequence[Sequence[float]],
    bridge_empirical_distributions: Sequence[Sequence[Sequence[float]]],
    bridge_l1_radii: Sequence[Sequence[float]],
    feasibility_guard: float = DEFAULT_FEASIBILITY_GUARD,
) -> BridgeMembershipCertificate:
    """Return the v2 strict bridge certificate with all slacks rechecked."""

    guard = _validate_guard(
        feasibility_guard, name="feasibility_guard", maximum=1e-4
    )
    reference = np.asarray(reference_empirical_distributions, dtype=np.float64)
    bridge = np.asarray(bridge_empirical_distributions, dtype=np.float64)
    reference_radii = np.asarray(reference_l1_radii, dtype=np.float64)
    target_radii = np.asarray(bridge_l1_radii, dtype=np.float64)
    base = certify_bridge_membership(
        reference,
        reference_l1_radii=reference_radii,
        bridge_empirical_distributions=bridge,
        bridge_l1_radii=target_radii,
    )
    labels = tuple(
        _strict_label(
            label,
            reference[label_index],
            reference_radii[label_index],
            bridge[label_index],
            target_radii[label_index],
            feasibility_guard=guard,
        )
        for label_index, label in enumerate(base.labels)
    )
    return BridgeMembershipCertificate(
        labels=labels,
        label_count=base.label_count,
        source_count=base.source_count,
        token_count=base.token_count,
        method="strict_v2_simultaneous_l1_bridge_membership",
    )
