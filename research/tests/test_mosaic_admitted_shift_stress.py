import numpy as np
import pytest

from analyze_mosaic_admitted_shift_stress import (
    population_metrics,
    probability_tensor,
    worst_admitted_law,
)


def test_probability_tensor_normalizes_complete_counts():
    counts = np.asarray(
        [
            [[8, 2, 0, 0], [6, 4, 0, 0]],
            [[0, 0, 7, 3], [0, 0, 5, 5]],
        ]
    )
    probabilities = probability_tensor(counts)
    assert probabilities.shape == (2, 2, 4)
    assert np.allclose(probabilities.sum(axis=2), 1.0)


def test_worst_admitted_law_matches_exact_direct_privacy():
    reference = np.asarray(
        [
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        ]
    )
    membership = {
        "labels": [
            {
                "transform": np.eye(4).tolist(),
                "retained_mass": 0.6,
                "contamination": 0.4,
            },
            {
                "transform": np.eye(4).tolist(),
                "retained_mass": 0.6,
                "contamination": 0.4,
            },
        ]
    }
    direct_channel = np.asarray(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    laws, details, error = worst_admitted_law(
        reference, membership, direct_channel
    )
    privacy, utility = population_metrics(laws, direct_channel, (0, 1))
    assert error == 0.0
    assert np.allclose(laws.sum(axis=2), 1.0)
    assert privacy == pytest.approx(0.4)
    assert utility == pytest.approx(0.4)
    assert max(
        row["exact_direct_worst_normalized_advantage"] for row in details
    ) == pytest.approx(0.4)
