from __future__ import annotations

from dataclasses import replace

from audit_mosaic_baseline_extension import aggregate_rows, replay_table
from run_mosaic_synthetic_pilot import witness_scenario


def test_replay_table_returns_all_registered_methods() -> None:
    scenario = replace(
        witness_scenario(
            privacy_threshold=0.35,
            utility_threshold=0.45,
            contamination=0.10,
        ),
        name="audit_smoke",
    )
    rows, diagnostic = replay_table((19, 250, scenario, 0.05))
    assert {row["method"] for row in rows} == {
        "mosaic_continuum",
        "table_region_grid",
        "holm_ltt_grid",
        "fare_style_deterministic",
    }
    assert diagnostic["seed"] == 19
    assert diagnostic["scenario"] == "audit_smoke"
    assert all(len(row["release_channel"]) == 3 for row in rows)


def test_audit_aggregate_counts_and_intervals_are_well_formed() -> None:
    scenario = replace(
        witness_scenario(
            privacy_threshold=0.30,
            utility_threshold=0.40,
            contamination=0.20,
        ),
        name="aggregate_smoke",
    )
    rows = []
    for seed in (31, 32):
        replayed, _ = replay_table((seed, 125, scenario, 0.05))
        for row in replayed:
            row["scenario"] = scenario.name
            rows.append(row)
    cells = aggregate_rows(rows)
    assert len(cells) == 4
    assert all(cell["replicates"] == 2 for cell in cells)
    assert all(
        0.0 <= cell["false_acceptance_cp95_lower"]
        <= cell["false_acceptance_cp95_upper"]
        <= 1.0
        for cell in cells
    )
