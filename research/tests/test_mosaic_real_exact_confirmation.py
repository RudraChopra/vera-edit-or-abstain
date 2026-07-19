from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run_mosaic_official_frontier_exact_confirmation import (
    candidate_result,
    identity_candidate,
    select_certified_result,
)
from run_mosaic_real_exact_confirmation import build_manifest


def _variant(error: float, *, deployed: bool = True) -> dict[str, object]:
    return {
        "certified_worst_conditional_error": error,
        "certified_privacy_advantages": [0.1, 0.2],
        "deployed": deployed,
        "external_estimable": True,
        "external_worst_privacy_advantage": 0.1,
        "external_worst_conditional_error": 0.2,
        "external_safe": True,
        "false_acceptance": False,
    }


def test_paired_selection_is_variant_specific_and_stable() -> None:
    results = [
        {
            "candidate": "B",
            "method": "Method",
            "strength": "one",
            "capacity_transfer": _variant(0.30),
            "transform_exact": _variant(0.20),
        },
        {
            "candidate": "A",
            "method": "Method",
            "strength": "two",
            "capacity_transfer": _variant(0.30),
            "transform_exact": _variant(0.25),
        },
    ]
    fallback = select_certified_result(results, "capacity_transfer")
    exact = select_certified_result(results, "transform_exact")
    assert fallback["candidate"] == "A"
    assert fallback["certificate_variant"] == "capacity_transfer"
    assert exact["candidate"] == "B"
    assert exact["certificate_variant"] == "transform_exact"


def test_paired_selection_abstains_when_variant_has_no_deployment() -> None:
    results = [
        {
            "candidate": "A",
            "method": "Method",
            "strength": "one",
            "capacity_transfer": _variant(0.3),
            "transform_exact": _variant(0.2, deployed=False),
        }
    ]
    assert select_certified_result(results, "capacity_transfer")["decision"] == "deploy"
    assert select_certified_result(results, "transform_exact")["decision"] == "abstain"


def test_paired_candidate_uses_identical_table_and_exact_dominates() -> None:
    rng = np.random.default_rng(17)
    y_construction = np.repeat([0, 1], 80)
    construction = np.column_stack(
        (
            2.0 * y_construction - 1.0 + rng.normal(0.0, 0.7, len(y_construction)),
            rng.normal(size=len(y_construction)),
        )
    ).astype(np.float32)
    y_certification = np.tile(np.repeat([0, 1], 20), 2)
    s_certification = np.repeat([0, 1], 40)
    y_external = y_certification.copy()
    s_external = s_certification.copy()

    def features(target: np.ndarray, source: np.ndarray) -> np.ndarray:
        return np.column_stack(
            (
                2.0 * target - 1.0 + rng.normal(0.0, 0.8, len(target)),
                0.4 * (2.0 * source - 1.0) + rng.normal(0.0, 0.8, len(target)),
            )
        ).astype(np.float32)

    certification = features(y_certification, s_certification)
    external = features(y_external, s_external)
    candidate = identity_candidate(
        np.empty((0, 2), dtype=np.float32),
        construction,
        np.concatenate((certification, external), axis=0),
    )
    result = candidate_result(
        candidate,
        y_construction=y_construction,
        y_certification=y_certification,
        y_external=y_external,
        s_certification=s_certification,
        s_external=s_external,
        certification_count=len(certification),
        seed=17,
        delta=0.05,
        contamination=0.05,
        privacy_threshold=0.35,
        utility_threshold=0.49,
        smoothing=0.10,
    )
    fallback = result["capacity_transfer"]
    exact = result["transform_exact"]
    assert np.asarray(result["certification_token_counts"]).shape == (2, 2, 4)
    assert exact["certified_worst_conditional_error"] <= (
        fallback["certified_worst_conditional_error"] + 2e-7
    )
    assert np.asarray(exact["release_channel"]).shape == (4, 2)
    assert np.asarray(fallback["release_channel"]).shape == (4, 2)


def test_manifest_applies_all_registered_paired_gates(tmp_path: Path) -> None:
    datasets = (
        "Waterbirds",
        "Camelyon17-WILDS",
        "CivilComments-WILDS",
        "BiasBios-Clinical",
        "GaitPDB",
    )
    outputs = []
    for dataset in datasets:
        for seed in range(105, 110):
            estimable = dataset != "Camelyon17-WILDS"
            safe = estimable
            candidate_variant = {
                "external_estimable": estimable,
                "external_safe": safe,
            }
            results = [
                {
                    "candidate": f"candidate-{index}",
                    "capacity_transfer": dict(candidate_variant),
                    "transform_exact": dict(candidate_variant),
                }
                for index in range(13)
            ]
            fallback_deploy = dataset == "BiasBios-Clinical"
            exact_deploy = fallback_deploy or dataset == "Waterbirds"

            def selection(deploy: bool) -> dict[str, object]:
                if not deploy:
                    return {"decision": "abstain", "candidate": None}
                return {
                    "decision": "deploy",
                    "candidate": "candidate-0",
                    "external_estimable": estimable,
                    "external_safe": safe,
                    "false_acceptance": False,
                }

            payload = {
                "dataset": dataset,
                "seed": seed,
                "results": results,
                "selection": {
                    "capacity_transfer": selection(fallback_deploy),
                    "transform_exact": selection(exact_deploy),
                },
            }
            output = tmp_path / f"{dataset}-{seed}.json"
            output.write_text(json.dumps(payload), encoding="utf-8")
            outputs.append(output)
    audit_path = tmp_path / "audit.json"
    audit_path.write_text("{}", encoding="utf-8")
    prereg = {
        "datasets": {dataset: {} for dataset in datasets},
        "claim_boundary": "This confirmation evaluates a frozen paired protocol.",
        "decision_gates": {
            "minimum_strict_candidate_improvements": 100,
            "minimum_additional_safe_selected_deployments": 1,
            "minimum_biasbios_exact_deployments": 4,
            "maximum_selected_false_acceptances": 0,
        },
    }
    audit = {
        "passed": True,
        "candidate_rows_replayed": 325,
        "optimization_replays": 650,
        "strict_objective_improvements": 101,
        "pointwise_dominance": True,
    }
    manifest = build_manifest(prereg, "registered", outputs, audit, audit_path)
    assert manifest["all_pass"]
    assert manifest["candidate_row_count"] == 325
    assert manifest["additional_safe_exact_selected_deployments"] == 5
