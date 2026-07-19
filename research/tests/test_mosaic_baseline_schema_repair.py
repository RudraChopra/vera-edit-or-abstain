from __future__ import annotations

import pytest

from repair_mosaic_baseline_report_schema import repair_report


def test_schema_repair_only_adds_registered_scenario() -> None:
    report = {
        "replicate_results": [
            {"seed": 1, "sample_size_per_stratum": 125, "value": 0.4},
            {"seed": 2, "sample_size_per_stratum": 250, "value": 0.3},
        ]
    }
    repaired, changed = repair_report(
        report,
        sample_size_to_scenario={
            125: "hard_safety_boundary",
            250: "retention_and_stochastic_value",
        },
    )
    assert changed == 2
    assert "scenario" not in report["replicate_results"][0]
    assert repaired["replicate_results"] == [
        {
            "seed": 1,
            "sample_size_per_stratum": 125,
            "value": 0.4,
            "scenario": "hard_safety_boundary",
        },
        {
            "seed": 2,
            "sample_size_per_stratum": 250,
            "value": 0.3,
            "scenario": "retention_and_stochastic_value",
        },
    ]


def test_schema_repair_rejects_unregistered_or_existing_labels() -> None:
    with pytest.raises(ValueError, match="unregistered sample size"):
        repair_report(
            {"replicate_results": [{"sample_size_per_stratum": 500}]},
            sample_size_to_scenario={125: "known"},
        )
    with pytest.raises(ValueError, match="already has a scenario"):
        repair_report(
            {
                "replicate_results": [
                    {"sample_size_per_stratum": 125, "scenario": "known"}
                ]
            },
            sample_size_to_scenario={125: "known"},
        )
