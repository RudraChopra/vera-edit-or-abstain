"""Analyze the frozen, disjoint-seed VERA stress replication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

from analyze_vera_attacker_ablation import load_candidates
from analyze_vera_balanced_existing import (
    balanced_external_metrics,
    balanced_point_metrics,
    balanced_samples,
    choose,
)
from analyze_vera_real_study import cp_interval, holm_adjust
from vera_robust_certificate import (
    certify_balanced_iut_fixed_profile,
    certify_balanced_shift_envelope,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREG = ROOT / "prereg_independent_stress_replication.json"
DEFAULT_HASH = ROOT / "prereg_independent_stress_replication.sha256"
DEFAULT_AUDIT = ROOT / "artifacts" / "independent_stress_replication_receipt_audit.json"
DEFAULT_RECEIPTS = ROOT / "artifacts" / "independent_stress_replication_receipts"
DEFAULT_ROWS = ROOT / "artifacts" / "vera_independent_stress_rule_rows.csv"
DEFAULT_CANDIDATES = ROOT / "artifacts" / "vera_independent_stress_candidate_rows.csv"
DEFAULT_REPORT = ROOT / "artifacts" / "vera_independent_stress_report.json"
DEFAULT_ABSTRACT = ROOT / "artifacts" / "vera_independent_stress_abstract_numbers.json"
RULES = (
    "always_deploy_balanced",
    "point_selection_balanced",
    "vera_balanced_iut",
    "vera_balanced_envelope",
    "external_balanced_oracle",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def exact_one_sided_mcnemar(point_only: int, vera_only: int) -> float:
    """Test whether point selection violates more often than VERA."""

    discordant = int(point_only) + int(vera_only)
    if discordant == 0:
        return 1.0
    return float(
        binomtest(
            int(point_only),
            discordant,
            0.5,
            alternative="greater",
        ).pvalue
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    estimable = [row for row in rows if row["external_contract_estimable"]]
    deployments = sum(bool(row["deployed"]) for row in rows)
    estimable_deployments = sum(bool(row["deployed"]) for row in estimable)
    violations = sum(bool(row["measured_external_contract_violation"]) for row in estimable)
    safe = sum(
        bool(row["deployed"] and row["external_contract_satisfied"] is True)
        for row in estimable
    )
    denominator = len(estimable)
    return {
        "configuration_count": len(rows),
        "estimable_configuration_count": denominator,
        "deployment_count": deployments,
        "deployment_rate": 0.0 if not rows else deployments / len(rows),
        "estimable_deployment_count": estimable_deployments,
        "safe_deployment_count": safe,
        "measured_external_violation_count": violations,
        "measured_external_violation_rate": (
            None if denominator == 0 else violations / denominator
        ),
        "measured_external_violation_cp95": list(cp_interval(violations, denominator)),
        "violation_rate_conditional_on_estimable_deployment": (
            None if estimable_deployments == 0 else violations / estimable_deployments
        ),
        "procedurally_unsupported_deployment_count": sum(
            bool(row["procedurally_unsupported_deployment"]) for row in rows
        ),
    }


def registered_environments(
    dataset_config: dict[str, Any], target: dict[str, np.ndarray]
) -> list[int]:
    observed = sorted(int(key.rsplit("=", 1)[1]) for key in target)
    return list(dataset_config.get("certification_environment_classes", observed)) + list(
        dataset_config.get("unsupported_external_environment_classes", [])
    )


def prepare_candidates(
    receipt_dir: Path,
    study: dict[str, Any],
    dataset: str,
    seed: int,
    target_threshold: float | None,
    leakage_threshold: float | None,
) -> list[dict[str, Any]]:
    loaded, labels = load_candidates(receipt_dir, study, dataset, seed)
    if len(loaded) != 12:
        raise RuntimeError(f"expected 12 candidates for {dataset}/seed-{seed}")
    indices = np.arange(len(labels[0]), dtype=np.int64)
    support_mismatch = bool(
        study["datasets"][dataset].get("force_abstain_for_unsupported_environment")
    )
    gamma = float(study["deployment_gamma"])
    delta = float(study["delta"])
    gamma_cap = float(study["gamma_cap"])
    prepared: list[dict[str, Any]] = []
    for candidate in loaded:
        target, leakage, source = balanced_samples(candidate["arrays"], indices)
        target_point, leakage_point = balanced_point_metrics(target, leakage, source)
        family_size = len(loaded) * (len(target) + len(leakage))
        if support_mismatch:
            iut_decision = "ABSTAIN"
            iut_limiting_contracts = ["unsupported_deployment_environment"]
            envelope_eligible = False
            envelope_radius = 0.0
            observed_envelope_radius = 0.0
            unsupported_environments = list(
                study["datasets"][dataset].get(
                    "unsupported_external_environment_classes", []
                )
            )
        else:
            if target_threshold is None or leakage_threshold is None:
                raise RuntimeError(f"supported dataset lacks a locked contract: {dataset}")
            iut = certify_balanced_iut_fixed_profile(
                target,
                leakage,
                source,
                gamma=gamma,
                delta=delta,
                candidate_count=len(loaded),
                target_threshold=target_threshold,
                leakage_threshold=leakage_threshold,
            )
            envelope = certify_balanced_shift_envelope(
                target,
                leakage,
                source,
                delta=delta,
                family_size=family_size,
                target_threshold=target_threshold,
                leakage_threshold=leakage_threshold,
                registered_target_environments=registered_environments(
                    study["datasets"][dataset], target
                ),
                gamma_cap=gamma_cap,
            )
            iut_decision = iut.decision
            iut_limiting_contracts = list(iut.limiting_contracts)
            envelope_eligible = envelope.deployment_common_radius >= gamma
            envelope_radius = float(envelope.deployment_common_radius)
            observed_envelope_radius = float(envelope.observed_common_radius)
            unsupported_environments = list(
                envelope.unsupported_target_environments
        )
        external_target, external_leakage = balanced_external_metrics(candidate["arrays"])
        external_estimable = (
            external_leakage is not None
            and target_threshold is not None
            and leakage_threshold is not None
        )
        external_satisfied = (
            None
            if not external_estimable
            or target_threshold is None
            or leakage_threshold is None
            else external_target <= target_threshold
            and float(external_leakage) <= leakage_threshold
        )
        prepared.append(
            {
                "candidate": candidate["candidate"],
                "method": candidate["method"],
                "validation_max_target_harm": float(target_point),
                "validation_max_balanced_leakage": float(leakage_point),
                "point_feasible": (
                    not support_mismatch
                    and target_threshold is not None
                    and leakage_threshold is not None
                    and target_point <= target_threshold
                    and leakage_point <= leakage_threshold
                ),
                "iut_eligible": (
                    not support_mismatch and iut_decision == "EDIT"
                ),
                "envelope_eligible": (
                    not support_mismatch and envelope_eligible
                ),
                "envelope_radius": envelope_radius,
                "observed_envelope_radius": observed_envelope_radius,
                "iut_limiting_contracts": iut_limiting_contracts,
                "envelope_unsupported_target_environments": unsupported_environments,
                "external_max_target_harm": float(external_target),
                "external_max_balanced_leakage": (
                    None if external_leakage is None else float(external_leakage)
                ),
                "external_contract_estimable": external_estimable,
                "external_contract_satisfied": external_satisfied,
                "external_feasible": external_satisfied is True,
                "certification_n": len(indices),
                "family_size": family_size,
            }
        )
    return prepared


def select_for_rule(
    candidates: list[dict[str, Any]], rule: str
) -> dict[str, Any] | None:
    predicates = {
        "always_deploy_balanced": None,
        "point_selection_balanced": "point_feasible",
        "vera_balanced_iut": "iut_eligible",
        "vera_balanced_envelope": "envelope_eligible",
        "external_balanced_oracle": "external_feasible",
    }
    return choose(candidates, predicates[rule])


def make_abstract_record(
    *,
    prereg_hash: str,
    supported_count: int,
    point_rate: float,
    vera_rate: float,
    retention: float,
    passed: bool,
    camelyon_forced_count: int,
) -> dict[str, Any]:
    if passed:
        mode = "independent_empirical_replication"
        sentence = (
            f"Across {supported_count} preregistered, disjoint-seed deployment decisions, "
            f"validation-only selection deployed contract-violating edits in "
            f"{100 * point_rate:.1f}% versus {100 * vera_rate:.1f}% for VERA, "
            f"while VERA retained {100 * retention:.1f}% of external-oracle opportunities."
        )
    else:
        mode = "theory_and_support_impossibility"
        sentence = (
            "The independent stress replication did not satisfy every preregistered "
            "empirical endpoint; VERA nevertheless forced abstention in all "
            f"{camelyon_forced_count} registered Camelyon17 decisions because the "
            "deployment hospital was outside certification support."
        )
    return {
        "verified": True,
        "registered_pass_conditions_met": passed,
        "headline_mode": mode,
        "prereg_sha256": prereg_hash,
        "supported_configuration_count": supported_count,
        "point_selection_violation_rate": point_rate,
        "vera_iut_violation_rate": vera_rate,
        "safe_retention": retention,
        "camelyon_forced_abstention_count": camelyon_forced_count,
        "sentence": sentence,
    }


def analyze(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    prereg_hash = sha256(args.prereg)
    expected_hash = args.hash_file.read_text(encoding="utf-8").split()[0]
    if prereg_hash != expected_hash:
        raise RuntimeError("independent-replication preregistration hash mismatch")
    prereg = load_json(args.prereg)
    if prereg.get("phase") != "independent disjoint-seed stress replication":
        raise RuntimeError("wrong independent-replication phase")
    audit = load_json(args.receipt_audit)
    if audit.get("passed") is not True or audit.get("prereg_sha256") != prereg_hash:
        raise RuntimeError("independent receipt matrix has not passed its locked audit")

    study = prereg["real_study"]
    seeds = [int(seed) for seed in study["seeds"]]
    contracts = study["locked_dataset_contracts"]
    supported = list(contracts)
    if set(supported) != {
        dataset
        for dataset, config in study["datasets"].items()
        if not config.get("force_abstain_for_unsupported_environment")
    }:
        raise RuntimeError("locked contracts do not match supported datasets")

    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for dataset, dataset_config in study["datasets"].items():
        support_mismatch = bool(
            dataset_config.get("force_abstain_for_unsupported_environment")
        )
        contract = contracts.get(dataset)
        target_threshold = (
            float(contract["target_harm_threshold"]) if contract else None
        )
        leakage_threshold = (
            float(contract["balanced_leakage_threshold"]) if contract else None
        )
        for seed in seeds:
            candidates = prepare_candidates(
                args.receipt_dir,
                study,
                dataset,
                seed,
                target_threshold,
                leakage_threshold,
            )
            config_id = (
                f"{dataset}|seed={seed}|support-boundary"
                if contract is None
                else f"{dataset}|seed={seed}|tau={target_threshold:g}|lambda={leakage_threshold:g}"
            )
            for candidate in candidates:
                candidate_rows.append(
                    {
                        "config_id": config_id,
                        "dataset": dataset,
                        "seed": seed,
                        "target_threshold": target_threshold,
                        "leakage_threshold": leakage_threshold,
                        "candidate": candidate["candidate"],
                        "method": candidate["method"],
                        "certification_n": candidate["certification_n"],
                        "family_size": candidate["family_size"],
                        "validation_max_target_harm": candidate[
                            "validation_max_target_harm"
                        ],
                        "validation_max_balanced_leakage": candidate[
                            "validation_max_balanced_leakage"
                        ],
                        "point_feasible": candidate["point_feasible"],
                        "iut_eligible": candidate["iut_eligible"],
                        "envelope_eligible": candidate["envelope_eligible"],
                        "envelope_radius": candidate["envelope_radius"],
                        "observed_envelope_radius": candidate[
                            "observed_envelope_radius"
                        ],
                        "iut_limiting_contracts": json.dumps(
                            candidate["iut_limiting_contracts"], sort_keys=True
                        ),
                        "envelope_unsupported_target_environments": json.dumps(
                            candidate["envelope_unsupported_target_environments"],
                            sort_keys=True,
                        ),
                        "external_max_target_harm": candidate[
                            "external_max_target_harm"
                        ],
                        "external_max_balanced_leakage": candidate[
                            "external_max_balanced_leakage"
                        ],
                        "external_contract_estimable": candidate[
                            "external_contract_estimable"
                        ],
                        "external_contract_satisfied": (
                            "NA"
                            if candidate["external_contract_satisfied"] is None
                            else candidate["external_contract_satisfied"]
                        ),
                    }
                )
            for rule in RULES:
                selected = select_for_rule(candidates, rule)
                deployed = selected is not None
                estimable = bool(
                    selected["external_contract_estimable"] if deployed else not support_mismatch
                )
                satisfied = (
                    selected["external_contract_satisfied"]
                    if deployed and estimable
                    else None
                )
                rows.append(
                    {
                        "config_id": config_id,
                        "dataset": dataset,
                        "seed": seed,
                        "target_threshold": target_threshold,
                        "leakage_threshold": leakage_threshold,
                        "rule": rule,
                        "deployed": deployed,
                        "selected_candidate": selected["candidate"] if deployed else "",
                        "selected_method": selected["method"] if deployed else "",
                        "external_contract_estimable": estimable,
                        "external_contract_satisfied": (
                            "NA" if satisfied is None else satisfied
                        ),
                        "measured_external_contract_violation": (
                            False if satisfied is None else bool(deployed and not satisfied)
                        ),
                        "procedurally_unsupported_deployment": (
                            deployed and support_mismatch
                        ),
                        "support_mismatch_forced_abstention": (
                            support_mismatch
                            and rule in {"vera_balanced_iut", "vera_balanced_envelope"}
                            and not deployed
                        ),
                        "confirmatory": True,
                    }
                )

    expected_rows = len(study["datasets"]) * len(seeds) * len(RULES)
    expected_candidates = len(study["datasets"]) * len(seeds) * 12
    if len(rows) != expected_rows or len(candidate_rows) != expected_candidates:
        raise RuntimeError("independent-replication table shape mismatch")

    def select_rows(dataset: str | None, rule: str) -> list[dict[str, Any]]:
        return [
            row
            for row in rows
            if row["rule"] == rule and (dataset is None or row["dataset"] == dataset)
        ]

    summaries = {
        rule: summarize(
            [row for row in rows if row["dataset"] in supported and row["rule"] == rule]
        )
        for rule in RULES
    }
    by_dataset = {
        dataset: {rule: summarize(select_rows(dataset, rule)) for rule in RULES}
        for dataset in study["datasets"]
    }

    raw_p: dict[str, float] = {}
    discordance: dict[str, dict[str, int]] = {}
    point_by_key = {
        (row["dataset"], row["seed"]): row
        for row in rows
        if row["rule"] == "point_selection_balanced"
    }
    vera_by_key = {
        (row["dataset"], row["seed"]): row
        for row in rows
        if row["rule"] == "vera_balanced_iut"
    }
    for dataset in supported:
        point_only = sum(
            bool(point_by_key[(dataset, seed)]["measured_external_contract_violation"])
            and not bool(
                vera_by_key[(dataset, seed)]["measured_external_contract_violation"]
            )
            for seed in seeds
        )
        vera_only = sum(
            not bool(point_by_key[(dataset, seed)]["measured_external_contract_violation"])
            and bool(
                vera_by_key[(dataset, seed)]["measured_external_contract_violation"]
            )
            for seed in seeds
        )
        discordance[dataset] = {
            "point_only_violation": point_only,
            "vera_only_violation": vera_only,
            "concordant_both_violate": sum(
                bool(point_by_key[(dataset, seed)]["measured_external_contract_violation"])
                and bool(
                    vera_by_key[(dataset, seed)]["measured_external_contract_violation"]
                )
                for seed in seeds
            ),
            "concordant_neither_violates": sum(
                not bool(point_by_key[(dataset, seed)]["measured_external_contract_violation"])
                and not bool(
                    vera_by_key[(dataset, seed)]["measured_external_contract_violation"]
                )
                for seed in seeds
            ),
        }
        raw_p[dataset] = exact_one_sided_mcnemar(point_only, vera_only)
    adjusted_p = holm_adjust(raw_p)

    dataset_pass: dict[str, dict[str, Any]] = {}
    delta = float(study["delta"])
    for dataset in supported:
        point_rate = float(
            by_dataset[dataset]["point_selection_balanced"][
                "measured_external_violation_rate"
            ]
        )
        vera_rate = float(
            by_dataset[dataset]["vera_balanced_iut"][
                "measured_external_violation_rate"
            ]
        )
        dataset_pass[dataset] = {
            "point_violation_rate_at_least_20pct": point_rate >= 0.20,
            "vera_violation_rate_at_most_delta": vera_rate <= delta,
            "holm_adjusted_one_sided_mcnemar_at_most_0_05": (
                adjusted_p[dataset] <= 0.05
            ),
        }
        dataset_pass[dataset]["passed_all_three"] = all(
            dataset_pass[dataset].values()
        )

    supported_point = summaries["point_selection_balanced"]
    supported_vera = summaries["vera_balanced_iut"]
    oracle_rows = {
        (row["dataset"], row["seed"]): row
        for row in rows
        if row["dataset"] in supported and row["rule"] == "external_balanced_oracle"
    }
    oracle_opportunities = sum(bool(row["deployed"]) for row in oracle_rows.values())
    vera_safe_on_opportunity = sum(
        bool(
            oracle_row["deployed"]
            and vera_by_key[key]["deployed"]
            and vera_by_key[key]["external_contract_satisfied"] is True
        )
        for key, oracle_row in oracle_rows.items()
    )
    retention = (
        0.0
        if oracle_opportunities == 0
        else vera_safe_on_opportunity / oracle_opportunities
    )

    camelyon = [row for row in rows if row["dataset"] == "Camelyon17-WILDS"]
    camelyon_vera = [
        row
        for row in camelyon
        if row["rule"] in {"vera_balanced_iut", "vera_balanced_envelope"}
    ]
    camelyon_abstention = bool(camelyon_vera) and all(
        not row["deployed"]
        and row["support_mismatch_forced_abstention"]
        and not row["external_contract_estimable"]
        for row in camelyon_vera
    )
    global_vera_control = float(
        supported_vera["measured_external_violation_rate"]
    ) <= delta
    supported_pass_count = sum(
        record["passed_all_three"] for record in dataset_pass.values()
    )
    passed = (
        supported_pass_count == len(supported) == 4
        and global_vera_control
        and camelyon_abstention
    )

    report = {
        "name": "VERA preregistered independent stress replication",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "confirmatory": True,
        "passed": passed,
        "prereg_sha256": prereg_hash,
        "receipt_audit_sha256": sha256(args.receipt_audit),
        "design_seeds_excluded": prereg["data_policy"]["design_seeds"],
        "replication_seeds": seeds,
        "datasets": list(study["datasets"]),
        "supported_datasets": supported,
        "erasers": [method["display_name"] for method in study["methods"].values()],
        "candidate_count_per_dataset_seed": 12,
        "rule_row_count": len(rows),
        "candidate_row_count": len(candidate_rows),
        "deployment_rules": list(RULES),
        "delta": delta,
        "locked_dataset_contracts": contracts,
        "supported_summaries": summaries,
        "by_dataset": by_dataset,
        "one_sided_mcnemar_discordance": discordance,
        "one_sided_mcnemar_raw_p": raw_p,
        "one_sided_mcnemar_holm_p": adjusted_p,
        "dataset_pass_conditions": dataset_pass,
        "supported_datasets_passing_all_three": supported_pass_count,
        "global_vera_control": global_vera_control,
        "camelyon_forced_abstention": camelyon_abstention,
        "camelyon_forced_abstention_count": sum(
            bool(row["support_mismatch_forced_abstention"])
            for row in camelyon_vera
        ),
        "certification_tax": {
            "external_oracle_opportunity_count": oracle_opportunities,
            "vera_safe_deployment_count": vera_safe_on_opportunity,
            "safe_retention": retention,
            "safe_retention_cp95": list(
                cp_interval(vera_safe_on_opportunity, oracle_opportunities)
            ),
            "point_safe_deployment_count": supported_point["safe_deployment_count"],
            "vera_safe_deployment_count_all": supported_vera[
                "safe_deployment_count"
            ],
        },
        "selected_method_counts": {
            rule: dict(
                sorted(
                    Counter(
                        row["selected_method"]
                        for row in rows
                        if row["rule"] == rule and row["deployed"]
                    ).items()
                )
            )
            for rule in RULES
        },
        "pass_conditions": {
            "four_supported_datasets_pass_all_three": (
                supported_pass_count == len(supported) == 4
            ),
            "global_vera_violation_rate_at_most_delta": global_vera_control,
            "camelyon_forced_abstention_all_registered_vera_rules": camelyon_abstention,
        },
        "claim_boundary": prereg["claim_boundary"],
    }
    abstract = make_abstract_record(
        prereg_hash=prereg_hash,
        supported_count=len(supported) * len(seeds),
        point_rate=float(supported_point["measured_external_violation_rate"]),
        vera_rate=float(supported_vera["measured_external_violation_rate"]),
        retention=retention,
        passed=passed,
        camelyon_forced_count=report["camelyon_forced_abstention_count"],
    )
    return rows, candidate_rows, report, abstract


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--hash-file", type=Path, default=DEFAULT_HASH)
    parser.add_argument("--receipt-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--receipt-dir", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--candidate-rows", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--abstract", type=Path, default=DEFAULT_ABSTRACT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, candidate_rows, report, abstract = analyze(args)
    write_csv(args.rows, rows)
    write_csv(args.candidate_rows, candidate_rows)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.abstract.write_text(
        json.dumps(abstract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "pass_conditions": report["pass_conditions"],
                "dataset_pass_conditions": report["dataset_pass_conditions"],
                "abstract": abstract["sentence"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
