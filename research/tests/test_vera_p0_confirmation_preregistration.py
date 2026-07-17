from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from prepare_vera_p0_confirmation_preregistration import (  # noqa: E402
    GAMMA_GRID,
    P0_SEEDS,
    build_payload,
    load_json,
)
from run_official_eraser_frontier import (  # noqa: E402
    EXPANDED_HELDOUT_ATTACKER_CONFIG,
    EXPANDED_REGISTERED_ATTACKER_CONFIG,
)


def test_p0_protocol_freezes_fresh_seeds_and_expanded_attackers() -> None:
    parent_path = ROOT / "prereg_controlled_shift_followup.json"
    result_path = ROOT / "maintrack" / "CONTROLLED_SHIFT_FOLLOWUP_RESULT_SUMMARY.json"

    payload = build_payload(
        load_json(parent_path),
        load_json(result_path),
        parent_path=parent_path,
        result_path=result_path,
    )

    study = payload["real_study"]
    assert payload["status"] == "locked_before_claim_grade_runs"
    assert payload["schema_version"] == 2
    assert payload["supersedes"]["outcomes_present_before_supersession"] is False
    assert payload["data_policy"]["confirmatory_seeds"] == P0_SEEDS
    assert set(P0_SEEDS).isdisjoint(payload["data_policy"]["development_seeds"])
    assert study["leakage_attackers"] == EXPANDED_REGISTERED_ATTACKER_CONFIG
    assert study["heldout_attacker"] == EXPANDED_HELDOUT_ATTACKER_CONFIG
    assert study["controlled_shift_protocol"]["primary_requested_gamma"] == 1.25
    assert set(GAMMA_GRID) == {1.0, 1.1, 1.25, 1.5}
    required = study["construction_receipt_schema"]["required_arrays"]
    assert "target_harm_construction" in required
    assert "leakage_correct_construction__boosted_tree" in required
    assert "certification" in study["controlled_shift_protocol"]["stress_design_rule"]


def test_p0_protocol_scopes_the_natural_mixture_study_honestly() -> None:
    parent_path = ROOT / "prereg_controlled_shift_followup.json"
    result_path = ROOT / "maintrack" / "CONTROLLED_SHIFT_FOLLOWUP_RESULT_SUMMARY.json"
    payload = build_payload(
        load_json(parent_path),
        load_json(result_path),
        parent_path=parent_path,
        result_path=result_path,
    )

    natural = payload["real_study"]["natural_group_mixture_protocol"]
    assert "conditional-stability assumption" in natural["interpretation"]
    assert "separate" in natural["interpretation"]
    assert payload["data_policy"]["prior_results_may_choose_protocol_but_not_be_pooled"]
