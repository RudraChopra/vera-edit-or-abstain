from __future__ import annotations

import numpy as np

from run_mosaic_baseline_extension import (
    candidate_family,
    candidate_safety_p_value,
    holm_rejections,
    table_region_grid_selection,
)
from run_mosaic_synthetic_pilot import simultaneous_radii, witness_scenario


def test_holm_step_down_matches_manual_example() -> None:
    p_values = (0.001, 0.02, 0.30, 0.90)
    # 0.001 <= .05/4, then .02 > .05/3, so only the first is rejected.
    assert holm_rejections(p_values, delta=0.05) == (0,)


def test_candidate_family_is_the_registered_500_pairs() -> None:
    scenario = witness_scenario(
        privacy_threshold=0.35, utility_threshold=0.45, contamination=0.10
    )
    family = candidate_family(scenario)
    assert len(family) == 5**3 * 2**2
    assert all(channel.shape == (3, 2) and len(decoder) == 2 for channel, decoder in family)


def test_candidate_composite_p_value_and_table_grid_are_well_formed() -> None:
    scenario = witness_scenario(
        privacy_threshold=0.35, utility_threshold=0.45, contamination=0.10
    )
    empirical = scenario.population.copy()
    channel, decoder = candidate_family(scenario)[17]
    p_value = candidate_safety_p_value(
        empirical, channel, decoder, scenario, n=250
    )
    assert 0.0 <= p_value <= 1.0

    radii = simultaneous_radii(
        250,
        label_count=2,
        source_count=2,
        fine_count=3,
        delta=0.05,
    )
    selected = table_region_grid_selection(empirical, radii, scenario)
    np.testing.assert_allclose(selected.channel.sum(axis=1), 1.0)
    assert 0.0 <= selected.criterion <= 1.0
