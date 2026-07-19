from __future__ import annotations

import numpy as np

from run_mosaic_bridge_misspecification import (
    aggregate,
    exact_population_retained_masses,
    run_replicate,
    scenario_registry,
)


def test_population_bridge_separates_registered_misspecification() -> None:
    scenarios = scenario_registry()
    minima = {
        scenario.name: min(exact_population_retained_masses(scenario.target))
        for scenario in scenarios
    }
    assert minima["compatible_common_transform"] > 0.999999
    assert minima["underdeclared_contamination"] < 0.80
    assert minima["source_specific_transform"] < 0.80


def test_replicate_and_aggregate_are_well_formed() -> None:
    scenario = scenario_registry()[0]
    population = exact_population_retained_masses(scenario.target)
    row = run_replicate(
        (
            991,
            500,
            scenario,
            0.05,
            0.80,
            1e-7,
            population,
            (0.5, 0.5),
        )
    )
    assert np.asarray(row.reference_empirical).shape == (2, 2, 2)
    assert np.asarray(row.target_empirical).shape == (2, 2, 2)
    cells = aggregate([row])
    assert len(cells) == 1
    assert cells[0]["replicates"] == 1
    assert 0.0 <= cells[0]["acceptance_cp95_lower"] <= 1.0
    assert 0.0 <= cells[0]["acceptance_cp95_upper"] <= 1.0
