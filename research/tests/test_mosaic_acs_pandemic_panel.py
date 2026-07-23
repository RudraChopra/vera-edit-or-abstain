from __future__ import annotations

import numpy as np

from run_mosaic_acs_pandemic_panel import (
    FUTURE_TABLE_DELTA,
    diagnostic_bounds,
    expected_protocol,
    paired_witnesses,
    summarize,
)


def test_protocol_registers_full_family_before_outcomes():
    protocol = expected_protocol()
    assert protocol["population_jobs"] == 60
    assert protocol["future_year"] == "2021"
    assert protocol["future_tables_in_family"] == 240
    assert FUTURE_TABLE_DELTA == 0.05 / 240
    assert protocol["target_states"] == ["FL", "IL", "NY", "WA"]
    assert protocol["tasks"] == ["employment", "income", "public_coverage"]


def test_diagnostic_bounds_enclose_empirical_metrics():
    counts = np.asarray(
        [
            [[80, 20], [20, 80]],
            [[70, 30], [30, 70]],
        ],
        dtype=np.int64,
    )
    radii = np.full((2, 2), 0.10)
    channel = np.eye(2)
    bounds = diagnostic_bounds(counts, radii, channel, [0, 1])
    assert bounds["source_advantage_lower"] <= bounds["source_advantage_empirical"]
    assert bounds["source_advantage_empirical"] <= bounds["source_advantage_upper"]
    assert (
        bounds["worst_conditional_error_lower"]
        <= bounds["worst_conditional_error_empirical"]
    )
    assert (
        bounds["worst_conditional_error_empirical"]
        <= bounds["worst_conditional_error_upper"]
    )
    assert bounds["source_advantage_lower"] > 0.35


def test_confirmed_witness_requires_direct_failure_and_mosaic_catch():
    rows = [
        {
            "target_state": "FL",
            "task": "income",
            "seed": 1400,
            "rule": "direct",
            "decision_2018": "deploy",
            "runtime_action_2021": "release",
            "future_diagnostic": {
                "source_contract_violation_empirical": True,
                "utility_contract_violation_empirical": False,
                "source_contract_violation_confirmed": True,
                "utility_contract_violation_confirmed": False,
                "both_contracts_safe_confirmed": False,
            },
        },
        {
            "target_state": "FL",
            "task": "income",
            "seed": 1400,
            "rule": "mosaic",
            "decision_2018": "deploy",
            "runtime_action_2021": "abstain_out_of_bridge_class",
        },
    ]
    witnesses = paired_witnesses(rows)
    assert witnesses[0]["confirmed_natural_failure_witness"]
    summary = summarize(rows, witnesses)
    assert summary["confirmed_natural_failure_witnesses"] == 1
    assert summary["mosaic_runtime_abstentions_2021"] == 1
