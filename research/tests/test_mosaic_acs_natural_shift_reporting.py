from __future__ import annotations

from audit_mosaic_acs_natural_shift import audit_selection
from summarize_mosaic_acs_natural_shift import aggregate_cell, cp95, matched_deployment


def _selection(decision: str, candidate: str | None, *, false: bool = False) -> dict:
    row = {
        "decision": decision,
        "candidate": candidate,
        "method": candidate.split(":", 1)[0] if candidate else None,
    }
    if decision == "deploy":
        row.update(
            {
                "diagnostic_estimable": True,
                "diagnostic_safe": not false,
                "false_acceptance": false,
                "operational_replay": {
                    "draws": 100,
                    "primary_contract_violations": int(false),
                },
            }
        )
    return row


def _job(mosaic: dict, direct: dict) -> dict:
    return {
        "alphabets": {
            "4": {
                "selection_by_rule_and_threshold": {
                    "mosaic": {"0.40": mosaic},
                    "direct": {"0.40": direct},
                }
            }
        }
    }


def test_aggregate_cell_counts_only_deployed_estimable_jobs() -> None:
    rows = [
        _selection("deploy", "LEACE:a"),
        _selection("deploy", "INLP:b", false=True),
        _selection("abstain", None),
    ]
    result = aggregate_cell(rows)
    assert result["jobs"] == 3
    assert result["deployments"] == 2
    assert result["diagnostic_estimable_deployments"] == 2
    assert result["false_acceptances"] == 1
    assert result["operational_draws"] == 200
    assert result["operational_contract_violations"] == 1
    assert result["selected_method_counts"] == {"INLP": 1, "LEACE": 1}


def test_cp95_handles_boundary_counts_without_nan() -> None:
    assert cp95(0, 0) is None
    assert cp95(0, 10)[0] == 0.0
    assert cp95(10, 10)[1] == 1.0


def test_matched_deployment_reconstructs_all_four_paired_cells() -> None:
    deploy = _selection("deploy", "LEACE:a")
    abstain = _selection("abstain", None)
    jobs = [
        _job(deploy, deploy),
        _job(deploy, abstain),
        _job(abstain, deploy),
        _job(abstain, abstain),
    ]
    result = matched_deployment(jobs, "4", "0.40")
    assert result["both_deploy"] == 1
    assert result["mosaic_only"] == 1
    assert result["direct_only"] == 1
    assert result["neither"] == 1
    assert result["exact_two_sided_mcnemar_p"] == 1.0


def test_audit_selection_uses_registered_tie_break() -> None:
    rows = [
        {
            "candidate": "LEACE:z",
            "mosaic_release": {
                "certified_worst_conditional_error_upper": 0.39,
                "threshold_decisions": {"0.40": {"deployed": True}},
            },
        },
        {
            "candidate": "INLP:a",
            "mosaic_release": {
                "certified_worst_conditional_error_upper": 0.39,
                "threshold_decisions": {"0.40": {"deployed": True}},
            },
        },
    ]
    alphabet = {
        "selection_by_rule_and_threshold": {
            "mosaic": {"0.40": {"decision": "deploy", "candidate": "INLP:a"}}
        }
    }
    assert audit_selection(rows, alphabet, "mosaic", "0.40") == []
