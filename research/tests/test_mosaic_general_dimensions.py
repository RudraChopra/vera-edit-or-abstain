from __future__ import annotations

import numpy as np

from mosaic_bridge import certify_bridge_membership
from mosaic_exact import exact_external_attacker_risk
from mosaic_transform_exact import transform_exact_attacker_confidence_bound
from mosaic_transform_exact_optimizer import optimize_transform_exact_channel


def test_three_source_exact_envelope_matches_population_evaluator():
    rng = np.random.default_rng(314159)
    reference = rng.dirichlet(np.ones(3), size=3)
    release = rng.dirichlet(np.ones(2), size=3)
    transforms = (np.eye(3), rng.dirichlet(np.ones(3), size=3))
    certificate = transform_exact_attacker_confidence_bound(
        reference,
        release,
        l1_radii=(0.0, 0.0, 0.0),
        common_fine_token_channels=transforms,
        contamination=0.17,
    )
    population = exact_external_attacker_risk(
        reference,
        release,
        transforms,
        contamination=0.17,
    )
    assert certificate.source_count == 3
    assert abs(certificate.balanced_accuracy - population.balanced_accuracy) < 1e-12
    assert certificate.maximizing_assignment == population.maximizing_assignment


def test_bridge_recovers_common_transform_for_three_labels_and_sources():
    rng = np.random.default_rng(271828)
    reference = rng.dirichlet(np.ones(3), size=(3, 3))
    transform = rng.dirichlet(np.ones(3), size=3)
    target = reference @ transform
    certificate = certify_bridge_membership(
        reference,
        reference_l1_radii=np.zeros((3, 3)),
        bridge_empirical_distributions=target,
        bridge_l1_radii=np.zeros((3, 3)),
    )
    assert certificate.label_count == 3
    assert certificate.source_count == 3
    assert min(certificate.retained_masses) >= 1.0 - 2e-7
    for label, result in enumerate(certificate.labels):
        pushed = reference[label] @ result.transform
        assert np.all(target[label] >= result.retained_mass * pushed - 1e-9)


def test_optimizer_enumerates_three_class_decoders_and_three_source_attackers():
    rng = np.random.default_rng(161803)
    empirical = rng.dirichlet(np.ones(3), size=(3, 3))
    solution = optimize_transform_exact_channel(
        empirical,
        l1_radii=np.zeros((3, 3)),
        common_channels_by_label=((np.eye(3),),) * 3,
        contaminations=(0.0, 0.0, 0.0),
        privacy_advantage_thresholds=(0.9, 0.9, 0.9),
        released_token_count=2,
    )
    assert solution.label_count == 3
    assert solution.source_count == 3
    assert solution.solved_decoder_assignments == 3**2
    assert len(solution.decoder) == 2
    assert all(
        certificate.normalized_advantage <= 0.9 + 2e-7
        for certificate in solution.privacy_certificates
    )
