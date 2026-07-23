from __future__ import annotations

import json
from pathlib import Path

from audit_mosaic_qwen_powered_confirmation import (
    normalize,
    sha256,
    validate_summary,
)


def test_normalize_rounds_floats_and_orders_nested_records() -> None:
    value = {
        "z": [0.12345678901234],
        "a": {"b": 2, "a": 1},
    }
    assert normalize(value) == {
        "a": {"a": 1, "b": 2},
        "z": [0.123456789012],
    }


def test_validate_summary_matches_locked_runner_schema(
    tmp_path: Path,
) -> None:
    prereg = {
        "seeds": [1, 2],
        "utility_thresholds": [0.4, 0.49],
        "main_paper_inclusion_gate": {
            "minimum_primary_releases": 1,
        },
    }
    receipts = [
        {
            "status": "complete",
            "primary_release": True,
            "heldout_primary_violation": False,
            "operational_replays": [{"draw": 0}],
            "operational_violation_count": 0,
            "threshold_decisions": {"0.40": True, "0.49": True},
        },
        {
            "status": "complete",
            "primary_release": False,
            "heldout_primary_violation": False,
            "operational_replays": [],
            "operational_violation_count": 0,
            "threshold_decisions": {"0.40": False, "0.49": True},
        },
    ]
    for seed, receipt in zip(prereg["seeds"], receipts, strict=True):
        (tmp_path / f"seed-{seed}.json").write_text(
            json.dumps(receipt), encoding="utf-8"
        )
    summary = {
        "registered_jobs": 2,
        "completed_jobs": 2,
        "error_jobs": 0,
        "primary_releases": 1,
        "primary_abstentions": 1,
        "heldout_primary_violations": 0,
        "operational_primary_trials": 1,
        "operational_primary_violations": 0,
        "main_paper_inclusion_gate_pass": True,
        "receipt_sha256": {
            f"seed-{seed}.json": sha256(tmp_path / f"seed-{seed}.json")
            for seed in prereg["seeds"]
        },
    }
    (tmp_path / "summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    validated = validate_summary(prereg, tmp_path)
    assert validated["primary_releases"] == 1
    assert validated["utility_threshold_release_counts"] == {
        "0.40": 1,
        "0.49": 2,
    }
