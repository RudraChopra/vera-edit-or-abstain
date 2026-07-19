from __future__ import annotations

from run_mosaic_bridge_comparator_extension import select_candidate, threshold_key


def release(error: float, *, deployed: bool, diagnostic_error: float) -> dict[str, object]:
    return {
        "selection_worst_conditional_error": error,
        "selection_source_advantages": [0.2, 0.2],
        "diagnostic": {
            "estimable": True,
            "worst_privacy_advantage": 0.2,
            "worst_conditional_error": diagnostic_error,
            "missing_strata": [],
        },
        "threshold_decisions": {
            threshold_key(0.40): {
                "deployed": deployed,
                "diagnostic_safe": diagnostic_error <= 0.40,
                "false_acceptance": deployed and diagnostic_error > 0.40,
            }
        },
    }


def test_contract_rule_abstains_but_always_deploy_rule_selects() -> None:
    rows = [
        {
            "candidate": "B",
            "method": "M",
            "strength": "b",
            "validation_plugin": release(0.42, deployed=False, diagnostic_error=0.55),
        },
        {
            "candidate": "A",
            "method": "M",
            "strength": "a",
            "validation_plugin": release(0.41, deployed=False, diagnostic_error=0.52),
        },
    ]
    selective = select_candidate(
        rows,
        rule="validation_plugin",
        release_key="validation_plugin",
        source_threshold=0.35,
        utility_threshold=0.40,
    )
    always = select_candidate(
        rows,
        rule="always_deploy_validation",
        release_key="validation_plugin",
        source_threshold=0.35,
        utility_threshold=0.40,
        force_deploy=True,
    )
    assert selective["decision"] == "abstain"
    assert always["decision"] == "deploy"
    assert always["candidate"] == "A"
    assert always["false_acceptance"] is True


def test_contract_rule_uses_error_then_lexical_tie_break() -> None:
    rows = [
        {
            "candidate": candidate,
            "method": "M",
            "strength": candidate.lower(),
            "mosaic_transform_exact": release(
                0.30, deployed=True, diagnostic_error=0.35
            ),
        }
        for candidate in ("B", "A")
    ]
    selected = select_candidate(
        rows,
        rule="mosaic_transform_exact",
        release_key="mosaic_transform_exact",
        source_threshold=0.35,
        utility_threshold=0.40,
    )
    assert selected["decision"] == "deploy"
    assert selected["candidate"] == "A"
    assert selected["false_acceptance"] is False
