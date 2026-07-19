from __future__ import annotations

from copy import deepcopy

from audit_mosaic_bridge_strict_correction_v2 import compare_pair


def _receipt(*, zero_column: bool, retained: float) -> dict[str, object]:
    transform = (
        [[1.0, 0.0], [1.0, 0.0]]
        if zero_column
        else [[1.0, 0.0], [0.0, 1.0]]
    )
    return {
        "dataset": "test",
        "seed": 1,
        "protocol": {"utility_thresholds": [0.4]},
        "original_receipt_sha256": "raw",
        "preregistration_sha256": "prereg",
        "numerical_policy": {
            "bridge_feasibility_guard": 1e-9,
            "release_optimization_guard": 1e-6,
            "reported_value_guard": 1e-9,
            "decision_tolerance": 0.0,
        },
        "results": [
            {
                "candidate": "identity",
                "method": "identity",
                "strength": 0.0,
                "provenance": "test",
                "bridge_membership": {
                    "labels": [
                        {
                            "transform": transform,
                            "retained_mass": retained,
                            "optimal_retained_mass_upper": 0.8,
                            "transform_trace": 1.0,
                        }
                    ]
                },
                "release_l2": {"value": 1},
            }
        ],
    }


def test_scope_audit_accepts_structural_zero_retention_restore() -> None:
    v1 = _receipt(zero_column=True, retained=0.0)
    v2 = deepcopy(v1)
    v2["results"][0]["bridge_membership"]["labels"][0]["retained_mass"] = 0.8
    failures, summary = compare_pair(v1, v2)
    assert failures == []
    assert summary["changed_labels"] == 1
    assert summary["zero_to_positive_labels"] == 1


def test_scope_audit_rejects_unlocked_retention_change() -> None:
    v1 = _receipt(zero_column=False, retained=0.0)
    v2 = deepcopy(v1)
    v2["results"][0]["bridge_membership"]["labels"][0]["retained_mass"] = 0.8
    failures, _ = compare_pair(v1, v2)
    assert any("without a structural-zero output" in failure for failure in failures)
