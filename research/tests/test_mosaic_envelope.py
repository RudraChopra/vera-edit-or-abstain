from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "mosaic"
sys.path.insert(0, str(SCRIPTS))

from mosaic_envelope import (  # noqa: E402
    coarsen_distribution,
    coarsened_confidence_certificate,
    coarsened_multiclass_confidence_certificate,
    multinomial_robust_tv_certificate,
    robust_multiclass_attacker_accuracy,
    robust_multiclass_attacker_confidence_bound,
    robust_selected_rule_error_bound,
    robust_total_variation,
    robust_total_variation_confidence_bound,
    upper_event_mass,
    weissman_l1_radius,
)


def test_gamma_one_recovers_ordinary_total_variation() -> None:
    p0 = np.asarray([0.45, 0.30, 0.20, 0.05])
    p1 = np.asarray([0.10, 0.25, 0.40, 0.25])
    certificate = robust_total_variation(p0, p1, gamma=1.0)
    expected = 0.5 * np.abs(p1 - p0).sum()
    assert certificate.value == pytest.approx(expected)
    assert certificate.universal_balanced_attacker_accuracy == pytest.approx(
        0.5 * (1.0 + expected)
    )


def test_likelihood_ratio_envelope_matches_eventwise_exhaustion() -> None:
    p0 = np.asarray([0.52, 0.23, 0.17, 0.08])
    p1 = np.asarray([0.12, 0.31, 0.19, 0.38])
    gamma = 1.7
    certificate = robust_total_variation(p0, p1, gamma=gamma)

    scores = []
    for mask in range(1 << len(p0)):
        selector = np.asarray([(mask >> token) & 1 for token in range(len(p0))])
        a0 = float(np.dot(selector, p0))
        a1 = float(np.dot(selector, p1))
        lower_0 = max(a0 / gamma, 1.0 - gamma * (1.0 - a0))
        upper_1 = min(gamma * a1, 1.0 - (1.0 - a1) / gamma)
        scores.append(upper_1 - lower_0)
    assert certificate.value == pytest.approx(max(scores))


def test_confidence_envelope_covers_any_truth_in_the_stated_l1_balls() -> None:
    empirical_0 = np.asarray([0.55, 0.25, 0.15, 0.05])
    empirical_1 = np.asarray([0.15, 0.20, 0.35, 0.30])
    true_0 = np.asarray([0.50, 0.28, 0.16, 0.06])
    true_1 = np.asarray([0.18, 0.18, 0.33, 0.31])
    radius_0 = float(np.abs(empirical_0 - true_0).sum())
    radius_1 = float(np.abs(empirical_1 - true_1).sum())
    bound = robust_total_variation_confidence_bound(
        empirical_0,
        empirical_1,
        gamma=1.4,
        l1_radius_group_0=radius_0,
        l1_radius_group_1=radius_1,
    )
    exact_truth = robust_total_variation(true_0, true_1, gamma=1.4)
    assert bound.value + 1e-12 >= exact_truth.value


def test_coarsening_cannot_increase_the_population_robust_envelope() -> None:
    p0 = np.asarray([0.32, 0.21, 0.18, 0.15, 0.09, 0.05])
    p1 = np.asarray([0.08, 0.14, 0.17, 0.24, 0.22, 0.15])
    mapping = (0, 0, 1, 1, 2, 2)
    fine = robust_total_variation(p0, p1, gamma=1.6)
    coarse = robust_total_variation(
        coarsen_distribution(p0, mapping),
        coarsen_distribution(p1, mapping),
        gamma=1.6,
    )
    assert coarse.value <= fine.value + 1e-12


def test_same_fine_confidence_event_certifies_a_data_selected_coarsening() -> None:
    fine_empirical_0 = np.asarray([0.42, 0.17, 0.16, 0.14, 0.11])
    fine_empirical_1 = np.asarray([0.10, 0.22, 0.25, 0.18, 0.25])
    fine_truth_0 = np.asarray([0.40, 0.18, 0.18, 0.13, 0.11])
    fine_truth_1 = np.asarray([0.12, 0.20, 0.24, 0.19, 0.25])
    radius_0 = float(np.abs(fine_empirical_0 - fine_truth_0).sum())
    radius_1 = float(np.abs(fine_empirical_1 - fine_truth_1).sum())

    candidate_mappings = ((0, 0, 1, 1, 2), (0, 1, 1, 2, 2))
    selected = min(
        candidate_mappings,
        key=lambda mapping: coarsened_confidence_certificate(
            fine_empirical_0,
            fine_empirical_1,
            mapping,
            gamma=1.25,
            fine_l1_radius_group_0=radius_0,
            fine_l1_radius_group_1=radius_1,
        ).value,
    )
    bound = coarsened_confidence_certificate(
        fine_empirical_0,
        fine_empirical_1,
        selected,
        gamma=1.25,
        fine_l1_radius_group_0=radius_0,
        fine_l1_radius_group_1=radius_1,
    )
    exact = robust_total_variation(
        coarsen_distribution(fine_truth_0, selected),
        coarsen_distribution(fine_truth_1, selected),
        gamma=1.25,
    )
    assert bound.value + 1e-12 >= exact.value


def test_multinomial_certificate_reports_a_valid_but_conservative_radius() -> None:
    certificate = multinomial_robust_tv_certificate(
        (53, 22, 15, 10),
        (14, 21, 31, 34),
        gamma=1.1,
        failure_probability=0.05,
    )
    assert 0.0 <= certificate.value <= 1.0
    assert certificate.l1_radius_group_0 == pytest.approx(
        weissman_l1_radius(100, 4, 0.025)
    )
    assert certificate.l1_radius_group_1 == pytest.approx(
        weissman_l1_radius(100, 4, 0.025)
    )


def test_weissman_radius_supports_large_finite_alphabets_stably() -> None:
    n = 10_000
    token_count = 64
    delta = 0.001
    radius = weissman_l1_radius(n, token_count, delta)
    expected = np.sqrt(
        2.0 * (np.log((1 << token_count) - 2) - np.log(delta)) / n
    )
    assert np.isfinite(radius)
    assert radius == pytest.approx(expected)


def test_event_mass_upper_bound_has_the_expected_endpoints() -> None:
    assert float(upper_event_mass(0.0, 2.0)) == pytest.approx(0.0)
    assert float(upper_event_mass(1.0, 2.0)) == pytest.approx(1.0)


def test_multiclass_gamma_one_equals_the_bayes_source_attacker() -> None:
    distributions = np.asarray(
        [[0.70, 0.20, 0.10], [0.10, 0.70, 0.20], [0.20, 0.10, 0.70]]
    )
    envelope = robust_multiclass_attacker_accuracy(distributions, gamma=1.0)
    expected = np.max(distributions, axis=0).sum() / distributions.shape[0]
    assert envelope.balanced_accuracy == pytest.approx(expected)


def test_multiclass_confidence_and_selected_coarsening_cover_truth() -> None:
    truth = np.asarray(
        [[0.50, 0.25, 0.15, 0.10], [0.12, 0.48, 0.22, 0.18], [0.20, 0.15, 0.45, 0.20]]
    )
    empirical = np.asarray(
        [[0.47, 0.27, 0.16, 0.10], [0.15, 0.45, 0.21, 0.19], [0.18, 0.17, 0.46, 0.19]]
    )
    radii = tuple(float(np.abs(t - e).sum()) for t, e in zip(truth, empirical))
    bound = robust_multiclass_attacker_confidence_bound(
        empirical, gamma=1.25, l1_radii=radii
    )
    exact = robust_multiclass_attacker_accuracy(truth, gamma=1.25)
    assert bound.balanced_accuracy + 1e-12 >= exact.balanced_accuracy

    mapping = (0, 0, 1, 1)
    coarse_bound = coarsened_multiclass_confidence_certificate(
        empirical,
        mapping,
        gamma=1.25,
        fine_l1_radii=radii,
    )
    coarse_truth = robust_multiclass_attacker_accuracy(
        tuple(coarsen_distribution(row, mapping) for row in truth), gamma=1.25
    )
    assert coarse_bound.balanced_accuracy + 1e-12 >= coarse_truth.balanced_accuracy


def test_selected_rule_error_uses_the_uniform_token_event() -> None:
    truth = np.asarray([0.40, 0.30, 0.20, 0.10])
    empirical = np.asarray([0.36, 0.32, 0.21, 0.11])
    radius = float(np.abs(truth - empirical).sum())
    errors = (False, True, False, True)
    bound = robust_selected_rule_error_bound(
        empirical, errors, gamma=1.4, l1_radius=radius
    )
    truth_mass = truth[np.asarray(errors)].sum()
    assert bound + 1e-12 >= float(upper_event_mass(truth_mass, 1.4))
