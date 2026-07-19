from __future__ import annotations

import numpy as np

from mosaic_release_utility_common import expected_release_metrics, token_decoder


def test_token_decoder_uses_construction_majority_without_diagnostic_labels() -> None:
    decoder = token_decoder(np.asarray([0, 0, 1, 1]), np.asarray([0, 0, 1, 0]), 2)
    assert decoder.tolist() == [0, 0]


def test_release_metrics_are_exact_one_draw_expectations() -> None:
    metrics = expected_release_metrics(
        np.asarray([0, 1]),
        np.asarray([0, 1]),
        np.asarray([[0.8, 0.2], [0.3, 0.7]]),
        np.asarray([0, 1]),
    )
    assert np.isclose(metrics["expected_accuracy"], 0.75)
    assert np.isclose(metrics["expected_balanced_accuracy"], 0.75)
