from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from run_mosaic_synthetic_pilot import witness_scenario
from run_mosaic_transform_exact_pilot import (
    METHODS,
    aggregate,
    result_for_solution,
    run_refinement_replicate,
)


def test_refinement_replicate_is_complete_and_replayable() -> None:
    scenario = witness_scenario(
        privacy_threshold=0.35,
        utility_threshold=0.45,
        contamination=0.10,
    )
    results = run_refinement_replicate((91827, 250, scenario, 0.05))
    assert tuple(result.method for result in results) == METHODS
    assert all(result.seed == 91827 for result in results)
    assert all(len(result.release_channel) == 3 for result in results)
    assert all(len(result.empirical_table) == 2 for result in results)
    assert results[1].certified_worst_conditional_error <= (
        results[0].certified_worst_conditional_error + 2e-7
    )
    cells = aggregate(results)
    assert len(cells) == 2
    assert all(cell["replicates"] == 1 for cell in cells)


def test_deployment_rechecks_privacy_at_strict_decision_tolerance() -> None:
    scenario = witness_scenario(
        privacy_threshold=0.35,
        utility_threshold=0.99,
        contamination=0.10,
    )
    solution = SimpleNamespace(
        release_channel=np.full((3, 2), 0.5),
        decoder=(0, 1),
        certified_worst_conditional_error=0.5,
        privacy_certificates=(
            SimpleNamespace(normalized_advantage=0.3500000002),
            SimpleNamespace(normalized_advantage=0.2),
        ),
        solver_status="test",
        solver_mip_gap=0.0,
        solver_dual_bound=0.5,
        max_constraint_violation=2e-10,
    )
    result = result_for_solution(
        seed=7,
        n=125,
        scenario=scenario,
        method="capacity_transfer",
        empirical=np.asarray(scenario.population, dtype=np.float64),
        radii=np.zeros((2, 2), dtype=np.float64),
        event=True,
        solution=solution,
    )
    assert result.deployed is False
