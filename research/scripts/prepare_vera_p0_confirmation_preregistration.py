"""Freeze VERA's final P0 study before any new claim-grade outcome is created."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_official_eraser_frontier import (
    EXPANDED_HELDOUT_ATTACKER_CONFIG,
    EXPANDED_REGISTERED_ATTACKER_CONFIG,
)


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
DEFAULT_PARENT = ROOT / "prereg_controlled_shift_followup.json"
DEFAULT_PARENT_RESULT = ROOT / "maintrack" / "CONTROLLED_SHIFT_FOLLOWUP_RESULT_SUMMARY.json"
SUPERSEDED_PROTOCOL = ROOT / "prereg_vera_p0_confirmation.json"
DEFAULT_OUTPUT = ROOT / "prereg_vera_p0_confirmation_v2.json"
P0_RECEIPT_DIR = Path("/Volumes/Backups/FARO/artifacts/vera_p0_confirmation_v2_receipts")
P0_AUDIT_DIR = Path("/Volumes/Backups/FARO/artifacts/vera_p0_confirmation_v2_audit_arrays")
P0_SEEDS = list(range(173, 237))
DEVELOPMENT_SEEDS = list(range(45, 173))
SUPPORTED_DATASETS = ("Bios", "CivilComments-WILDS", "GaitPDB", "Waterbirds")
GAMMA_GRID = [1.0, 1.1, 1.25, 1.5]
PRIMARY_GAMMA = 1.25
PRIMARY_BUDGET = 12000
MINIMUM_CELL_FRACTION = 0.15


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_outcomes_absent() -> None:
    for directory in (P0_RECEIPT_DIR, P0_AUDIT_DIR):
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(f"P0 outcome directory is not empty: {directory}")


def verify_superseded_protocol_unused() -> None:
    """Do not amend an already-observed protocol under a new label."""
    previous = load_json(SUPERSEDED_PROTOCOL)
    freshness = previous.get("real_study", {}).get("freshness_guard", {})
    if not isinstance(freshness, dict):
        raise RuntimeError("superseded P0 protocol lacks its freshness guard")
    for key in ("fresh_receipt_dir", "fresh_audit_dir"):
        directory = Path(str(freshness.get(key, "")))
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(
                "cannot supersede P0 protocol after outcomes exist: " f"{directory}"
            )


def validate_parent(parent: dict[str, Any], result: dict[str, Any]) -> None:
    require(
        parent.get("status") == "locked_before_claim_grade_runs",
        "parent preregistration is not locked",
    )
    study = parent.get("real_study")
    require(isinstance(study, dict), "parent preregistration has no study")
    require(
        [int(seed) for seed in study.get("seeds", [])] == list(range(109, 173)),
        "parent must be the completed independent follow-up",
    )
    require(
        set(P0_SEEDS).isdisjoint(DEVELOPMENT_SEEDS),
        "P0 seeds overlap development seeds",
    )
    require(
        result.get("analysis_status") == "complete_independent_followup_success",
        "parent follow-up result is not complete",
    )
    require(
        result.get("overall_confirmatory_success") is True,
        "parent follow-up did not pass its own registered gates",
    )


def build_payload(
    parent: dict[str, Any],
    result: dict[str, Any],
    *,
    parent_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    validate_parent(parent, result)
    parent_study = parent["real_study"]
    study = copy.deepcopy(parent_study)
    study["datasets"] = {
        name: copy.deepcopy(parent_study["datasets"][name])
        for name in SUPPORTED_DATASETS
    }
    study["seeds"] = P0_SEEDS
    study["candidate_count_total"] = 12
    study["deployment_gamma"] = PRIMARY_GAMMA
    study["leakage_attackers"] = copy.deepcopy(EXPANDED_REGISTERED_ATTACKER_CONFIG)
    study["heldout_attacker"] = copy.deepcopy(EXPANDED_HELDOUT_ATTACKER_CONFIG)
    study["analysis_tiers"] = {
        "development": (
            "all completed seeds 45-172 may choose this protocol, but are never "
            "pooled with P0 confirmation"
        ),
        "primary": (
            "independent P0 confirmation on seeds 173-236 at Gamma=1.25 with "
            "the expanded registered attacker portfolio"
        ),
        "secondary": (
            "predeclared Gamma=1, 1.1, and 1.5 profiles, evidence budgets, "
            "and natural group-mixture diagnostics on the same fresh seeds"
        ),
    }
    study["controlled_shift_protocol"] = {
        "scientific_object": (
            "each seed's finite empirical reference law over untouched official "
            "validation audit atoms"
        ),
        "shift_profile_family": "bounded density-ratio reweightings with Gamma in {1, 1.1, 1.25, 1.5}",
        "primary_requested_gamma": PRIMARY_GAMMA,
        "secondary_requested_gammas": [1.0, 1.1, 1.5],
        "profile": (
            "use the exact induced environment-conditional and source-conditional "
            "caps, never the global Gamma label alone"
        ),
        "stress_design_rule": (
            "On construction-only data, first select the edit with the largest "
            "construction target balanced accuracy (ties: candidate key). Among "
            "(environment, source, target) cells present in both construction and "
            "certification, select the lexicographically tied cell with the largest "
            "positive surplus: the maximum of mean paired target harm minus that "
            "dataset's target-harm threshold and, across the five registered "
            "attackers, mean source-correctness minus that dataset's balanced-"
            "leakage threshold. Certification, external, and held-out KNN outcomes "
            "cannot enter either selection."
        ),
        "weight_rule": (
            "assign the selected supported cell density ratio Gamma and assign the "
            "unique nonnegative residual ratio to every other atom in that "
            "environment; all other environments retain ratio one"
        ),
        "certification_stream": (
            "independent with-replacement draws from the finite reference law under "
            "the locked evidence allocation"
        ),
        "final_evaluation": (
            "exact Q-weighted risks are primary labels; a separate 50,000-draw "
            "stream is a sampling diagnostic only"
        ),
        "membership_receipt": (
            "record the Q probability hash, selected construction-only cell, exact "
            "induced profiles, and machine-checked density-ratio membership"
        ),
        "claim_boundary": (
            "this validates the controlled finite-reference stress family, not "
            "membership of future people, images, comments, recordings, or hospitals"
        ),
    }
    study["construction_receipt_schema"] = {
        "purpose": (
            "make the construction-only design decision independently replayable "
            "without revealing certification or external outcomes"
        ),
        "required_arrays": [
            "target_harm_construction",
            "identity_target_error_construction",
            "edited_target_error_construction",
            "source_construction",
            "environment_construction",
            "target_construction",
            *[
                f"leakage_correct_construction__{name}"
                for name in EXPANDED_REGISTERED_ATTACKER_CONFIG
            ],
        ],
        "forbidden_for_design": [
            "target_harm_certification",
            "target_harm_external",
            "heldout_leakage_correct_construction__knn_distance",
            "heldout_leakage_correct_certification__knn_distance",
            "heldout_leakage_correct_external__knn_distance",
        ],
    }
    study["natural_group_mixture_protocol"] = {
        "purpose": "practical relevance diagnostic distinct from the exact controlled study",
        "deployment_split": "the untouched native external split supplied by each public benchmark",
        "estimand": (
            "observed target harm and portfolio leakage under its naturally occurring "
            "environment mixture, together with the validation-to-external mixture shift"
        ),
        "supported_shift_assumption": (
            "within-environment conditional laws are stable and only environment "
            "mixture weights change"
        ),
        "required_reporting": [
            "validation and external environment mixtures",
            "mixture-ratio cap and support overlap",
            "per-environment observed risks",
            "separate within-environment discrepancy diagnostics",
            "IID LTT versus VERA deployment, safety, and retention",
            "one accepted, one abstained, and one unsupported-support case",
        ],
        "interpretation": (
            "the observed natural split is a realistic supported-mixture evaluation "
            "only to the degree that the stated conditional-stability assumption is "
            "credible; it is reported separately from the exact guarantee"
        ),
    }
    study["evidence_allocation"] = {
        "primary_total_contract_observation_budget": PRIMARY_BUDGET,
        "primary_rule": "targeted_floor_0.15",
        "minimum_fraction_per_registered_cell": MINIMUM_CELL_FRACTION,
        "score_rule": (
            "use construction-only margins for the fixed design edit and allocate "
            "the remaining integer budget by squared Gamma-over-margin scores with "
            "deterministic largest-remainder rounding"
        ),
        "secondary_budgets": [4000, 8000, 16000],
        "allocation_may_not_read": [
            "certification outcomes",
            "external outcomes",
            "held-out KNN outcomes",
        ],
    }
    study["primary_endpoints"] = {
        "iid_ltt_exposure": {
            "unit": "dataset-seed deployment decision",
            "estimand": (
                "fraction of IID LTT deployments that satisfy the IID certificate "
                "but violate the exact declared shifted contract"
            ),
            "success": (
                "at least 20% IID LTT shifted violations in at least one of the "
                "four predeclared Gamma profiles, reported with a one-sided interval"
            ),
        },
        "vera_safety": {
            "unit": "one prespecified rotating dataset decision per seed at Gamma=1.25",
            "event": "a deployed VERA edit violates any registered target or five-attacker leakage contract",
            "interval": "one-sided 95% Clopper-Pearson upper bound",
            "success": "upper bound <= 0.05 and zero observed sentinel events",
        },
        "vera_usefulness": {
            "unit": "whole seed cluster",
            "estimand": "VERA vector safe deployments divided by exact shifted-law safe opportunities",
            "interval": "95% percentile bootstrap over 64 seeds with 20,000 fixed resamples",
            "success": "lower confidence bound >= 0.20 at Gamma=1.1 and Gamma=1.25",
        },
        "paired_comparison": {
            "unit": "seed cluster aggregated across four supported datasets",
            "test": "two-sided exact sign test for IID LTT minus VERA shifted violations",
            "success": "positive effect with p < 0.05",
        },
        "heldout_knn_stress": {
            "scope": "outside the formal guarantee",
            "success": "report all deployments and a one-sided 95% interval; do not describe zero events as universal erasure",
        },
    }
    study["missingness"] = {
        "policy": "a missing or corrupt receipt fails the matrix and is restored with the same seed; no performance-based replacement or exclusion is allowed",
        "outcome_reporting": "all predeclared primary and secondary results are reported regardless of success",
    }
    study["freshness_guard"] = {
        "fresh_receipt_dir": str(P0_RECEIPT_DIR),
        "fresh_audit_dir": str(P0_AUDIT_DIR),
        "required_order": [
            "commit and push this protocol with its SHA-256 sidecar",
            "run only fresh seeds 173-236 into the P0 directories",
            "seal two independent analyzers before reading scientific outcomes",
            "report failed and successful endpoints without pooling prior seeds",
        ],
    }
    return {
        "schema_version": 2,
        "project": "VERA",
        "phase": "final P0 IID-LTT, attacker-portfolio, and natural-mixture confirmation",
        "status": "locked_before_claim_grade_runs",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_generator_git_commit": git_commit(),
        "primary_claim": (
            "On fresh data, VERA evaluates a representation edit against a declared "
            "supported shift and five registered retrained attacker families; IID "
            "LTT is evaluated against the same edits, evidence, contracts, and shifts."
        ),
        "parent": {
            "path": str(parent_path.relative_to(REPOSITORY)),
            "sha256": sha256(parent_path),
            "result_path": str(result_path.relative_to(REPOSITORY)),
            "result_sha256": sha256(result_path),
            "result_canonical_sha256": canonical_sha256(result),
        },
        "supersedes": {
            "path": str(SUPERSEDED_PROTOCOL.relative_to(REPOSITORY)),
            "sha256": sha256(SUPERSEDED_PROTOCOL),
            "reason": (
                "No P0 outcome was generated under version 1. Version 2 adds the "
                "construction-fold audit arrays required to independently replay "
                "the preregistered stress-design decision."
            ),
            "outcomes_present_before_supersession": False,
        },
        "data_policy": {
            "development_seeds": DEVELOPMENT_SEEDS,
            "confirmatory_seeds": P0_SEEDS,
            "seed_blocks_disjoint": True,
            "prior_results_may_choose_protocol_but_not_be_pooled": True,
            "external_outcomes_may_not_change_construction_or_certification_choices": True,
        },
        "real_study": study,
        "human_only_gates_not_satisfied": [
            "independent proof review",
            "external cold ML and statistics reviews",
            "human authorship and AAAI policy verification",
            "submission to AAAI",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--parent-result", type=Path, default=DEFAULT_PARENT_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-freshness-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite preregistration: {args.output}")
    if not args.skip_freshness_check:
        verify_outcomes_absent()
        verify_superseded_protocol_unused()
    payload = build_payload(
        load_json(args.parent),
        load_json(args.parent_result),
        parent_path=args.parent,
        result_path=args.parent_result,
    )
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    digest = sha256(args.output)
    args.output.with_suffix(".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
