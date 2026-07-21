from __future__ import annotations

import numpy as np

from mosaic_certified_release import Mosaic, MosaicConfig


def _balanced_rows(per_stratum: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    targets = np.repeat((0, 0, 1, 1), per_stratum)
    sources = np.repeat((0, 1, 0, 1), per_stratum)
    features = np.column_stack(
        (
            5.0 * (2 * targets - 1) + rng.normal(0.0, 0.25, len(targets)),
            rng.normal(0.0, 1.0, len(targets)),
        )
    )
    return features, targets, sources


def test_three_step_api_certifies_and_releases_persistently(tmp_path) -> None:
    construction, construction_y, _ = _balanced_rows(500, 1)
    reference, reference_y, reference_s = _balanced_rows(800, 2)
    bridge, bridge_y, bridge_s = _balanced_rows(800, 3)
    model = Mosaic(
        MosaicConfig(
            utility_error_threshold=0.45,
            privacy_advantage_threshold=0.35,
            seed=4,
        )
    ).fit(construction, construction_y)

    certificate = model.certify(
        reference,
        reference_y,
        reference_s,
        bridge,
        bridge_y,
        bridge_s,
    )
    first = model.release_or_abstain("item-1", reference[0])
    repeated = model.release_or_abstain("item-1", reference[0])

    assert certificate.certified
    assert first.status == "RELEASED"
    assert first == repeated
    assert first.predicted_label in (0, 1)

    artifact = tmp_path / "mosaic.joblib"
    model.save(artifact)
    restored = Mosaic.load(artifact)
    assert restored.release_or_abstain("item-1", reference[0]) == first


def test_missing_source_label_support_abstains() -> None:
    construction, construction_y, _ = _balanced_rows(100, 5)
    reference, reference_y, reference_s = _balanced_rows(100, 6)
    bridge, bridge_y, bridge_s = _balanced_rows(100, 7)
    keep = ~((bridge_y == 1) & (bridge_s == 1))
    model = Mosaic().fit(construction, construction_y)

    certificate = model.certify(
        reference,
        reference_y,
        reference_s,
        bridge[keep],
        bridge_y[keep],
        bridge_s[keep],
    )
    release = model.release_or_abstain("item-2", reference[0])

    assert not certificate.certified
    assert certificate.reason == "MISSING_SOURCE_LABEL_SUPPORT"
    assert release.status == "ABSTAIN"
    assert release.reason == certificate.reason
