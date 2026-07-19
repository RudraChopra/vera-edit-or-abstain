from __future__ import annotations

import numpy as np

from mosaic_envelope import weissman_l1_radius
from run_mosaic_scaling_study import population_table, required_sample_size


def test_scaling_population_is_a_valid_conditional_table() -> None:
    table = population_table(64, 4)
    assert table.shape == (2, 4, 64)
    assert np.all(table >= 0.0)
    assert np.allclose(table.sum(axis=2), 1.0)


def test_required_n_restores_the_k4_radius() -> None:
    source_count = 4
    delta = 0.05 / (2 * source_count)
    target = weissman_l1_radius(2_000, 4, delta)
    required = required_sample_size(64, source_count, target_radius=target)
    assert required > 2_000
    assert weissman_l1_radius(required, 64, delta) <= target + 1e-12
