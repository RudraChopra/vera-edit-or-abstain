import numpy as np

from mosaic_continuous import (
    dkw_threshold_l1_radius,
    threshold_class_attacker_certificate,
)


def test_dkw_radius_decreases_with_sample_size() -> None:
    assert dkw_threshold_l1_radius(4000, 0.01) < dkw_threshold_l1_radius(
        1000, 0.01
    )


def test_threshold_certificate_accepts_postselected_cut() -> None:
    source_zero = np.linspace(-2.0, 1.0, 2000)
    source_one = np.linspace(-1.0, 2.0, 2000)
    result = threshold_class_attacker_certificate(
        [source_zero, source_one],
        threshold=0.25,
        failure_probabilities=[0.01, 0.01],
    )
    assert result.empirical_binary_laws.shape == (2, 2)
    assert all(0.0 < radius < 0.2 for radius in result.per_source_l1_radii)
    assert result.attacker.normalized_advantage >= 0.0
