from __future__ import annotations

import numpy as np

from run_mosaic_real_proxy_mass_confirmation import (
    SEED,
    expected_protocol,
    mass_calibrated_certificate,
)


def test_mass_confirmation_protocol_is_fixed() -> None:
    protocol = expected_protocol()
    assert protocol["seed"] == SEED
    assert protocol["fine_token_count"] == 2
    assert protocol["source_mass_family_allocation"].startswith("5 percent")
    assert protocol["conditional_center"].startswith("exact L1")


def test_mass_confirmation_derives_label_source_counts() -> None:
    true = np.asarray(
        [
            [[100, 50], [40, 110]],
            [[80, 70], [60, 90]],
        ],
        dtype=np.int64,
    )
    confusion = np.asarray([[0.9, 0.1], [0.1, 0.9]])
    proxy = np.zeros_like(true)
    calibration = np.zeros((2, 2, 2, 2), dtype=np.int64)
    for label, source, token in np.ndindex(true.shape):
        count = int(true[label, source, token])
        first = int(round(confusion[source, 0] * count))
        proxy[label, 0, token] += first
        proxy[label, 1, token] += count - first
        calibration[label, source, token, 0] = 9 * first
        calibration[label, source, token, 1] = 9 * (count - first)
    certificate = mass_calibrated_certificate(
        proxy,
        family_failure_probability=0.05,
        calibration_confusion_counts=calibration,
        confidence_region="coordinate_clopper_pearson",
    )
    assert certificate.source_mass_calibration_sample_size == int(
        calibration.sum()
    )
    assert certificate.source_mass_per_event_failure_probability is not None
