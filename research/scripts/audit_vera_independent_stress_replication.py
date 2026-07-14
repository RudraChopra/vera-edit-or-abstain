"""Independently replay the disjoint-seed VERA stress replication from raw NPZs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

from audit_vera_confirmatory_analysis import (
    as_bool,
    choose,
    holm,
    independent_balanced_accuracy,
    independent_balanced_leakage_ucb,
    independent_envelope_radius,
    independent_external_metrics,
    independent_summary,
    independent_target_ucb,
    load_csv,
    load_json,
    sha256,
    values_match,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREG = ROOT / "prereg_independent_stress_replication.json"
DEFAULT_HASH = ROOT / "prereg_independent_stress_replication.sha256"
DEFAULT_RECEIPT_AUDIT = ROOT / "artifacts" / "independent_stress_replication_receipt_audit.json"
DEFAULT_RECEIPT_DIR = ROOT / "artifacts" / "independent_stress_replication_receipts"
DEFAULT_ROWS = ROOT / "artifacts" / "vera_independent_stress_rule_rows.csv"
DEFAULT_CANDIDATES = ROOT / "artifacts" / "vera_independent_stress_candidate_rows.csv"
DEFAULT_REPORT = ROOT / "artifacts" / "vera_independent_stress_report.json"
DEFAULT_ABSTRACT = ROOT / "artifacts" / "vera_independent_stress_abstract_numbers.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "vera_independent_stress_analysis_audit.json"
RULE_PREDICATES = {
    "always_deploy_balanced": None,
    "point_selection_balanced": "point_feasible",
    "vera_balanced_iut": "iut_eligible",
    "vera_balanced_envelope": "envelope_eligible",
    "external_balanced_oracle": "external_feasible",
}


def exact_one_sided_mcnemar(point_only: int, vera_only: int) -> float:
    discordant = point_only + vera_only
    if discordant == 0:
        return 1.0
    return float(
        binomtest(point_only, discordant, 0.5, alternative="greater").pvalue
    )


def compare_summary(
    reported: dict[str, Any],
    expected: dict[str, Any],
    prefix: str,
    failures: list[str],
) -> None:
    for key, value in expected.items():
        if key not in reported or not values_match(reported[key], value):
            failures.append(f"{prefix}.{key} differs from independent replay")


def replay_raw_candidates(
    prereg: dict[str, Any],
    receipt_dir: Path,
    failures: list[str],
) -> tuple[dict[str, list[dict[str, Any]]], int, int]:
    study = prereg["real_study"]
    contracts = study["locked_dataset_contracts"]
    delta = float(study["delta"])
    gamma = float(study["deployment_gamma"])
    gamma_cap = float(study["gamma_cap"])
    by_config: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    npz_count = 0
    checksum_count = 0

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
        for seed in map(int, study["seeds"]):
            config_id = (
                f"{dataset}|seed={seed}|support-boundary"
                if contract is None
                else f"{dataset}|seed={seed}|tau={target_threshold:g}|lambda={leakage_threshold:g}"
            )
            loaded: list[tuple[str, str, dict[str, np.ndarray]]] = []
            reference: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
            for method_key, method_config in study["methods"].items():
                receipt_path = receipt_dir / f"{dataset}__{method_key}__seed-{seed}.json"
                receipt = load_json(receipt_path)
                for candidate in receipt["candidates"]:
                    npz_path = Path(candidate["audit_npz"])
                    if sha256(npz_path) != candidate["audit_npz_sha256"]:
                        failures.append(f"raw NPZ checksum mismatch: {npz_path}")
                        continue
                    checksum_count += 1
                    with np.load(npz_path) as archive:
                        arrays = {key: np.asarray(archive[key]) for key in archive.files}
                    npz_count += 1
                    labels = (
                        arrays["target_certification"],
                        arrays["source_certification"],
                        arrays["environment_certification"],
                    )
                    if reference is None:
                        reference = labels
                    elif not all(
                        np.array_equal(left, right)
                        for left, right in zip(reference, labels)
                    ):
                        failures.append(
                            f"raw candidate labels differ: {dataset}/seed-{seed}"
                        )
                    loaded.append(
                        (
                            f"{method_config['display_name']}::{candidate['strength']}",
                            str(method_config["display_name"]),
                            arrays,
                        )
                    )
            if reference is None or len(loaded) != 12:
                failures.append(
                    f"raw candidate family has {len(loaded)} rows: {dataset}/seed-{seed}"
                )
                continue

            for candidate_key, method, arrays in loaded:
                environment = arrays["environment_certification"]
                target_harm = arrays["target_harm_certification"]
                source = arrays["source_certification"]
                target = {
                    f"target::environment={group}": target_harm[
                        environment == group
                    ]
                    for group in sorted(map(int, np.unique(environment)))
                }
                leakage = {
                    name.removeprefix("leakage_correct_certification__"): values
                    for name, values in arrays.items()
                    if name.startswith("leakage_correct_certification__")
                }
                target_point = max(float(values.mean()) for values in target.values())
                leakage_point = max(
                    float(independent_balanced_accuracy(values, source))
                    for values in leakage.values()
                )
                family_size = len(loaded) * (len(target) + len(leakage))
                if support_mismatch:
                    iut_eligible = False
                    envelope_eligible = False
                    envelope_radius = 0.0
                    observed_envelope_radius = 0.0
                    point_feasible = False
                else:
                    if target_threshold is None or leakage_threshold is None:
                        raise RuntimeError(f"supported dataset lacks contract: {dataset}")
                    iut_alpha = delta / len(loaded)
                    iut_eligible = all(
                        independent_target_ucb(
                            values, gamma=gamma, alpha=iut_alpha
                        )
                        <= target_threshold
                        for values in target.values()
                    ) and all(
                        independent_balanced_leakage_ucb(
                            values, source, gamma=gamma, alpha=iut_alpha
                        )
                        <= leakage_threshold
                        for values in leakage.values()
                    )
                    envelope_alpha = delta / family_size
                    envelope_radius = independent_envelope_radius(
                        target,
                        leakage,
                        source,
                        alpha=envelope_alpha,
                        target_threshold=target_threshold,
                        leakage_threshold=leakage_threshold,
                        gamma_cap=gamma_cap,
                    )
                    observed_envelope_radius = envelope_radius
                    envelope_eligible = envelope_radius >= gamma
                    point_feasible = (
                        target_point <= target_threshold
                        and leakage_point <= leakage_threshold
                    )
                external_target, external_leakage = independent_external_metrics(arrays)
                external_estimable = external_leakage is not None and contract is not None
                external_satisfied = (
                    None
                    if not external_estimable
                    else external_target <= target_threshold
                    and float(external_leakage) <= leakage_threshold
                )
                by_config[config_id].append(
                    {
                        "config_id": config_id,
                        "dataset": dataset,
                        "seed": seed,
                        "target_threshold": target_threshold,
                        "leakage_threshold": leakage_threshold,
                        "candidate": candidate_key,
                        "method": method,
                        "certification_n": len(source),
                        "family_size": family_size,
                        "validation_max_target_harm": target_point,
                        "validation_max_balanced_leakage": leakage_point,
                        "point_feasible": point_feasible,
                        "iut_eligible": iut_eligible,
                        "envelope_eligible": envelope_eligible,
                        "envelope_radius": envelope_radius,
                        "observed_envelope_radius": observed_envelope_radius,
                        "external_max_target_harm": external_target,
                        "external_max_balanced_leakage": external_leakage,
                        "external_contract_estimable": external_estimable,
                        "external_contract_satisfied": external_satisfied,
                        "external_feasible": external_satisfied is True,
                    }
                )
    return dict(by_config), npz_count, checksum_count


def main() -> int:
    args = parse_args()
    failures: list[str] = []
    prereg_hash = sha256(args.prereg)
    expected_hash = args.hash_file.read_text(encoding="utf-8").split()[0]
    if prereg_hash != expected_hash:
        failures.append("preregistration hash sidecar mismatch")
    prereg = load_json(args.prereg)
    receipt_audit = load_json(args.receipt_audit)
    if receipt_audit.get("passed") is not True:
        failures.append("official receipt audit did not pass")
    if receipt_audit.get("prereg_sha256") != prereg_hash:
        failures.append("official receipt audit uses another preregistration")

    rows = load_csv(args.rows)
    candidates = load_csv(args.candidates)
    report = load_json(args.report)
    abstract = load_json(args.abstract)
    raw_by_config, raw_npz_count, raw_checksum_count = replay_raw_candidates(
        prereg, args.receipt_dir, failures
    )
    study = prereg["real_study"]
    expected_candidates = (
        len(study["datasets"]) * len(study["seeds"]) * 12
    )
    expected_rule_rows = (
        len(study["datasets"]) * len(study["seeds"]) * len(RULE_PREDICATES)
    )
    if (
        raw_npz_count != expected_candidates
        or raw_checksum_count != expected_candidates
    ):
        failures.append(
            f"expected {expected_candidates} checksum-verified raw NPZs, found "
            f"{raw_npz_count}/{raw_checksum_count}"
        )

    candidates_by_config: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    rules_by_config: defaultdict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for candidate in candidates:
        candidates_by_config[candidate["config_id"]].append(candidate)
    for row in rows:
        if row["rule"] in rules_by_config[row["config_id"]]:
            failures.append(f"duplicate rule row: {row['config_id']} / {row['rule']}")
        rules_by_config[row["config_id"]][row["rule"]] = row
    if set(raw_by_config) != set(candidates_by_config):
        failures.append("raw and reported candidate configuration keys differ")
    if set(raw_by_config) != set(rules_by_config):
        failures.append("raw and rule configuration keys differ")

    raw_candidate_rows = 0
    raw_candidate_mismatches = 0
    for config_id, raw_values in raw_by_config.items():
        reported = {
            candidate["candidate"]: candidate
            for candidate in candidates_by_config.get(config_id, [])
        }
        for raw in raw_values:
            raw_candidate_rows += 1
            observed = reported.get(raw["candidate"])
            if observed is None:
                raw_candidate_mismatches += 1
                continue
            external_leakage = raw["external_max_balanced_leakage"]
            comparisons = (
                observed["method"] == raw["method"],
                values_match(
                    float(observed["validation_max_target_harm"]),
                    raw["validation_max_target_harm"],
                ),
                values_match(
                    float(observed["validation_max_balanced_leakage"]),
                    raw["validation_max_balanced_leakage"],
                ),
                as_bool(observed["point_feasible"]) == raw["point_feasible"],
                as_bool(observed["iut_eligible"]) == raw["iut_eligible"],
                as_bool(observed["envelope_eligible"]) == raw["envelope_eligible"],
                values_match(float(observed["envelope_radius"]), raw["envelope_radius"]),
                values_match(
                    float(observed["observed_envelope_radius"]),
                    raw["observed_envelope_radius"],
                ),
                values_match(
                    float(observed["external_max_target_harm"]),
                    raw["external_max_target_harm"],
                ),
                (
                    observed["external_max_balanced_leakage"] == ""
                    if external_leakage is None
                    else values_match(
                        float(observed["external_max_balanced_leakage"]),
                        external_leakage,
                    )
                ),
                as_bool(observed["external_contract_estimable"])
                == raw["external_contract_estimable"],
                (
                    observed["external_contract_satisfied"] == "NA"
                    if raw["external_contract_satisfied"] is None
                    else as_bool(observed["external_contract_satisfied"])
                    == raw["external_contract_satisfied"]
                ),
                int(observed["family_size"]) == raw["family_size"],
                int(observed["certification_n"]) == raw["certification_n"],
            )
            if not all(comparisons):
                raw_candidate_mismatches += 1
    if len(candidates) != expected_candidates:
        failures.append(
            f"reported candidate table has {len(candidates)} rows, expected "
            f"{expected_candidates}"
        )
    if len(rows) != expected_rule_rows:
        failures.append(
            f"reported rule table has {len(rows)} rows, expected {expected_rule_rows}"
        )
    if raw_candidate_rows != expected_candidates:
        failures.append(f"raw replay generated {raw_candidate_rows} candidate rows")
    if raw_candidate_mismatches:
        failures.append(
            f"{raw_candidate_mismatches} candidate rows differ from raw replay"
        )

    selection_mismatches = 0
    semantic_mismatches = 0
    for config_id, raw_candidates in raw_by_config.items():
        config_rules = rules_by_config.get(config_id, {})
        if len(raw_candidates) != 12:
            failures.append(f"{config_id} has {len(raw_candidates)} raw candidates")
            continue
        if set(config_rules) != set(RULE_PREDICATES):
            failures.append(f"{config_id} has the wrong rule family")
            continue
        support_mismatch = raw_candidates[0]["dataset"] == "Camelyon17-WILDS"
        for rule, predicate in RULE_PREDICATES.items():
            selected = choose(raw_candidates, predicate)
            row = config_rules[rule]
            deployed = selected is not None
            if (
                as_bool(row["deployed"]) != deployed
                or row["selected_candidate"]
                != (selected["candidate"] if selected else "")
                or row["selected_method"] != (selected["method"] if selected else "")
            ):
                selection_mismatches += 1
            estimable = (
                bool(selected["external_contract_estimable"])
                if selected
                else not support_mismatch
            )
            satisfied = (
                selected["external_contract_satisfied"]
                if selected and estimable
                else None
            )
            violation = False if satisfied is None else bool(deployed and not satisfied)
            if (
                as_bool(row["external_contract_estimable"]) != estimable
                or row["external_contract_satisfied"]
                != ("NA" if satisfied is None else str(bool(satisfied)))
                or as_bool(row["measured_external_contract_violation"]) != violation
            ):
                semantic_mismatches += 1
    if selection_mismatches:
        failures.append(f"{selection_mismatches} rule selections do not replay")
    if semantic_mismatches:
        failures.append(f"{semantic_mismatches} rule semantics do not replay")

    supported = list(study["locked_dataset_contracts"])
    supported_rows = [row for row in rows if row["dataset"] in supported]
    for rule in RULE_PREDICATES:
        expected = independent_summary(
            [row for row in supported_rows if row["rule"] == rule]
        )
        compare_summary(
            report.get("supported_summaries", {}).get(rule, {}),
            expected,
            f"supported.{rule}",
            failures,
        )
    for dataset in study["datasets"]:
        for rule in RULE_PREDICATES:
            expected = independent_summary(
                [
                    row
                    for row in rows
                    if row["dataset"] == dataset and row["rule"] == rule
                ]
            )
            compare_summary(
                report.get("by_dataset", {}).get(dataset, {}).get(rule, {}),
                expected,
                f"dataset.{dataset}.{rule}",
                failures,
            )

    point = {
        (row["dataset"], int(row["seed"])): row
        for row in rows
        if row["rule"] == "point_selection_balanced"
    }
    vera = {
        (row["dataset"], int(row["seed"])): row
        for row in rows
        if row["rule"] == "vera_balanced_iut"
    }
    discordance: dict[str, dict[str, int]] = {}
    raw_p: dict[str, float] = {}
    for dataset in supported:
        point_only = sum(
            as_bool(point[(dataset, seed)]["measured_external_contract_violation"])
            and not as_bool(
                vera[(dataset, seed)]["measured_external_contract_violation"]
            )
            for seed in map(int, study["seeds"])
        )
        vera_only = sum(
            not as_bool(point[(dataset, seed)]["measured_external_contract_violation"])
            and as_bool(
                vera[(dataset, seed)]["measured_external_contract_violation"]
            )
            for seed in map(int, study["seeds"])
        )
        both = sum(
            as_bool(point[(dataset, seed)]["measured_external_contract_violation"])
            and as_bool(vera[(dataset, seed)]["measured_external_contract_violation"])
            for seed in map(int, study["seeds"])
        )
        discordance[dataset] = {
            "point_only_violation": point_only,
            "vera_only_violation": vera_only,
            "concordant_both_violate": both,
            "concordant_neither_violates": len(study["seeds"])
            - point_only
            - vera_only
            - both,
        }
        raw_p[dataset] = exact_one_sided_mcnemar(point_only, vera_only)
    adjusted_p = holm(raw_p)
    if report.get("one_sided_mcnemar_discordance") != discordance:
        failures.append("paired discordance table does not replay")
    for key, expected in (
        ("one_sided_mcnemar_raw_p", raw_p),
        ("one_sided_mcnemar_holm_p", adjusted_p),
    ):
        observed = report.get(key, {})
        if set(observed) != set(expected) or any(
            not values_match(observed[dataset], value)
            for dataset, value in expected.items()
        ):
            failures.append(f"{key} does not replay")

    delta = float(study["delta"])
    dataset_pass: dict[str, dict[str, bool]] = {}
    for dataset in supported:
        point_summary = independent_summary(
            [row for row in rows if row["dataset"] == dataset and row["rule"] == "point_selection_balanced"]
        )
        vera_summary = independent_summary(
            [row for row in rows if row["dataset"] == dataset and row["rule"] == "vera_balanced_iut"]
        )
        dataset_pass[dataset] = {
            "point_violation_rate_at_least_20pct": float(
                point_summary["measured_external_violation_rate"]
            )
            >= 0.20,
            "vera_violation_rate_at_most_delta": float(
                vera_summary["measured_external_violation_rate"]
            )
            <= delta,
            "holm_adjusted_one_sided_mcnemar_at_most_0_05": (
                adjusted_p[dataset] <= 0.05
            ),
        }
        dataset_pass[dataset]["passed_all_three"] = all(
            dataset_pass[dataset].values()
        )
    if report.get("dataset_pass_conditions") != dataset_pass:
        failures.append("dataset pass conditions do not replay")

    supported_vera = independent_summary(
        [row for row in supported_rows if row["rule"] == "vera_balanced_iut"]
    )
    global_control = float(
        supported_vera["measured_external_violation_rate"]
    ) <= delta
    camelyon_vera = [
        row
        for row in rows
        if row["dataset"] == "Camelyon17-WILDS"
        and row["rule"] in {"vera_balanced_iut", "vera_balanced_envelope"}
    ]
    camelyon_pass = bool(camelyon_vera) and all(
        not as_bool(row["deployed"])
        and as_bool(row["support_mismatch_forced_abstention"])
        and not as_bool(row["external_contract_estimable"])
        for row in camelyon_vera
    )
    supported_pass_count = sum(value["passed_all_three"] for value in dataset_pass.values())
    expected_pass = supported_pass_count == 4 and global_control and camelyon_pass
    if report.get("passed") is not expected_pass:
        failures.append("overall preregistered pass flag does not replay")

    oracle = {
        (row["dataset"], int(row["seed"])): row
        for row in supported_rows
        if row["rule"] == "external_balanced_oracle"
    }
    opportunities = sum(as_bool(row["deployed"]) for row in oracle.values())
    safe_retained = sum(
        as_bool(oracle[key]["deployed"])
        and as_bool(vera[key]["deployed"])
        and vera[key]["external_contract_satisfied"] == "True"
        for key in oracle
    )
    retention = 0.0 if opportunities == 0 else safe_retained / opportunities
    reported_tax = report.get("certification_tax", {})
    for key, expected in (
        ("external_oracle_opportunity_count", opportunities),
        ("vera_safe_deployment_count", safe_retained),
        ("safe_retention", retention),
    ):
        if not values_match(reported_tax.get(key), expected):
            failures.append(f"certification_tax.{key} does not replay")

    point_summary = independent_summary(
        [row for row in supported_rows if row["rule"] == "point_selection_balanced"]
    )
    point_rate = float(point_summary["measured_external_violation_rate"])
    vera_rate = float(supported_vera["measured_external_violation_rate"])
    expected_mode = (
        "independent_empirical_replication"
        if expected_pass
        else "theory_and_support_impossibility"
    )
    for key, expected in (
        ("registered_pass_conditions_met", expected_pass),
        ("headline_mode", expected_mode),
        ("supported_configuration_count", len(supported) * len(study["seeds"])),
        ("point_selection_violation_rate", point_rate),
        ("vera_iut_violation_rate", vera_rate),
        ("safe_retention", retention),
        ("camelyon_forced_abstention_count", len(camelyon_vera)),
    ):
        if not values_match(abstract.get(key), expected):
            failures.append(f"abstract field {key} does not replay")
    if abstract.get("verified") is not True:
        failures.append("abstract record is not marked verified")

    audit = {
        "name": "Independent raw audit of VERA disjoint-seed stress replication",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": not failures,
        "confirmatory_passed": report.get("passed") is True,
        "prereg_sha256": prereg_hash,
        "receipt_audit_sha256": sha256(args.receipt_audit),
        "rule_rows_sha256": sha256(args.rows),
        "candidate_rows_sha256": sha256(args.candidates),
        "report_sha256": sha256(args.report),
        "abstract_sha256": sha256(args.abstract),
        "rule_rows_replayed": len(rows),
        "candidate_rows_replayed": len(candidates),
        "raw_candidate_rows_recomputed": raw_candidate_rows,
        "raw_npz_files_recomputed": raw_npz_count,
        "raw_npz_checksums_verified": raw_checksum_count,
        "raw_candidate_mismatches": raw_candidate_mismatches,
        "selection_mismatches": selection_mismatches,
        "semantic_mismatches": semantic_mismatches,
        "one_sided_mcnemar_holm_p_replayed": adjusted_p,
        "global_vera_control_replayed": global_control,
        "camelyon_forced_abstention_replayed": camelyon_pass,
        "safe_retention_replayed": retention,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--hash-file", type=Path, default=DEFAULT_HASH)
    parser.add_argument("--receipt-audit", type=Path, default=DEFAULT_RECEIPT_AUDIT)
    parser.add_argument("--receipt-dir", type=Path, default=DEFAULT_RECEIPT_DIR)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--abstract", type=Path, default=DEFAULT_ABSTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
