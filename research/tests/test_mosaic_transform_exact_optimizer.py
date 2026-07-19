from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from mosaic_optimizer import optimize_invariant_channel
from mosaic_transform_exact import (
    transform_exact_attacker_confidence_bound,
    transform_exact_utility_confidence_bound,
)
from mosaic_transform_exact_optimizer import optimize_transform_exact_channel


def _optimization_problem() -> tuple[np.ndarray, np.ndarray, tuple[tuple[np.ndarray, ...], ...]]:
    empirical = np.asarray(
        [
            [[0.76, 0.19, 0.05], [0.62, 0.32, 0.06]],
            [[0.05, 0.23, 0.72], [0.05, 0.36, 0.59]],
        ]
    )
    radii = np.full((2, 2), 0.08)
    smoothing = np.asarray(
        [[0.9, 0.1, 0.0], [0.05, 0.9, 0.05], [0.0, 0.1, 0.9]]
    )
    libraries = ((np.eye(3), smoothing), (np.eye(3), smoothing))
    return empirical, radii, libraries


def test_exact_optimizer_recomputes_every_certificate() -> None:
    empirical, radii, libraries = _optimization_problem()
    solution = optimize_transform_exact_channel(
        empirical,
        l1_radii=radii,
        common_channels_by_label=libraries,
        contaminations=(0.1, 0.1),
        privacy_advantage_thresholds=(0.45, 0.45),
        released_token_count=2,
    )
    assert solution.solver_mip_gap <= 1e-10
    assert solution.max_constraint_violation <= 2e-7
    assert solution.solved_decoder_assignments == 4
    assert all(c.normalized_advantage <= 0.45 + 2e-7 for c in solution.privacy_certificates)
    assert abs(
        solution.certified_worst_conditional_error
        - max(c.error_probability for row in solution.utility_certificates for c in row)
    ) < 1e-12


def test_exact_optimizer_is_no_worse_than_capacity_transfer_optimizer() -> None:
    empirical, radii, libraries = _optimization_problem()
    arguments = dict(
        l1_radii=radii,
        common_channels_by_label=libraries,
        contaminations=(0.1, 0.1),
        privacy_advantage_thresholds=(0.45, 0.45),
        released_token_count=2,
    )
    exact = optimize_transform_exact_channel(empirical, **arguments)
    transfer = optimize_invariant_channel(empirical, **arguments)
    assert exact.certified_worst_conditional_error <= (
        transfer.certified_worst_conditional_error + 2e-7
    )


def test_exact_optimizer_beats_or_matches_dense_channel_grid() -> None:
    empirical, radii, libraries = _optimization_problem()
    eta = (0.1, 0.1)
    thresholds = (0.45, 0.45)
    exact = optimize_transform_exact_channel(
        empirical,
        l1_radii=radii,
        common_channels_by_label=libraries,
        contaminations=eta,
        privacy_advantage_thresholds=thresholds,
        released_token_count=2,
    )

    grid_best = 1.0
    for rows in product(np.linspace(0.0, 1.0, 11), repeat=3):
        channel = np.asarray([[value, 1.0 - value] for value in rows])
        for decoder in product(range(2), repeat=2):
            privacy = [
                transform_exact_attacker_confidence_bound(
                    empirical[label],
                    channel,
                    l1_radii=radii[label],
                    common_fine_token_channels=libraries[label],
                    contamination=eta[label],
                )
                for label in range(2)
            ]
            if any(
                certificate.normalized_advantage > thresholds[label] + 1e-12
                for label, certificate in enumerate(privacy)
            ):
                continue
            utility = [
                transform_exact_utility_confidence_bound(
                    empirical[label, source],
                    channel,
                    decoder,
                    true_label=label,
                    l1_radius=radii[label, source],
                    common_fine_token_channels=libraries[label],
                    contamination=eta[label],
                ).error_probability
                for label in range(2)
                for source in range(2)
            ]
            grid_best = min(grid_best, max(utility))
    assert exact.certified_worst_conditional_error <= grid_best + 2e-7


def test_constraint_generation_matches_full_attacker_enumeration() -> None:
    empirical, radii, libraries = _optimization_problem()
    arguments = dict(
        l1_radii=radii,
        common_channels_by_label=libraries,
        contaminations=(0.1, 0.1),
        privacy_advantage_thresholds=(0.45, 0.45),
        released_token_count=2,
    )
    full = optimize_transform_exact_channel(empirical, **arguments)
    generated = optimize_transform_exact_channel(
        empirical, attacker_constraint_generation=True, **arguments
    )
    assert generated.certified_worst_conditional_error == pytest.approx(
        full.certified_worst_conditional_error, abs=2e-7
    )
    assert generated.active_attacker_assignments <= 4
    assert generated.constraint_generation_iterations >= 1
    assert all(
        certificate.normalized_advantage <= 0.45 + 2e-7
        for certificate in generated.privacy_certificates
    )
