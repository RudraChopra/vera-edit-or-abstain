"""Audit frozen independent-stress rows without requiring raw NPZ arrays."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analyze_vera_independent_stress_replication import (
    RULES,
    exact_one_sided_mcnemar,
    make_abstract_record,
    summarize,
)
from analyze_vera_real_study import holm_adjust


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREG = ROOT / "prereg_independent_stress_replication.json"
DEFAULT_ROWS = ROOT / "artifacts" / "vera_independent_stress_rule_rows.csv"
DEFAULT_CANDIDATES = ROOT / "artifacts" / "vera_independent_stress_candidate_rows.csv"
DEFAULT_REPORT = ROOT / "artifacts" / "vera_independent_stress_report.json"
DEFAULT_ABSTRACT = ROOT / "artifacts" / "vera_independent_stress_abstract_numbers.json"
DEFAULT_FULL_AUDIT = ROOT / "artifacts" / "vera_independent_stress_analysis_audit.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "vera_independent_stress_compact_audit.json"

RULE_PREDICATES = {
    "always_deploy_balanced": None,
    "point_selection_balanced": "point_feasible",
    "vera_balanced_iut": "iut_eligible",
    "vera_balanced_envelope": "envelope_eligible",
    "external_balanced_oracle": "external_contract_satisfied",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: str | bool) -> bool:
    return value if isinstance(value, bool) else value.strip().lower() == "true"


def maybe_number(value: str) -> float | None:
    if value == "" or value == "NA" or value is None:
        return None
    return float(value)


def values_match(observed: Any, expected: Any) -> bool:
    if observed is None or expected is None:
        return observed is expected
    if isinstance(expected, bool) or isinstance(expected, (str, int)):
        return observed == expected
    if isinstance(expected, list):
        return (
            isinstance(observed, list)
            and len(observed) == len(expected)
            and all(values_match(left, right) for left, right in zip(observed, expected))
        )
    if isinstance(expected, dict):
        return (
            isinstance(observed, dict)
            and set(observed) == set(expected)
            and all(values_match(observed[key], expected[key]) for key in expected)
        )
    return abs(float(observed) - float(expected)) <= 1e-12


def choose(
    candidates: list[dict[str, str]], predicate: str | None
) -> dict[str, str] | None:
    eligible = candidates
    if predicate is not None:
        if predicate == "external_contract_satisfied":
            eligible = [
                candidate
                for candidate in candidates
                if candidate["external_contract_satisfied"] == "True"
            ]
        else:
            eligible = [candidate for candidate in candidates if as_bool(candidate[predicate])]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda candidate: (
            float(candidate["validation_max_balanced_leakage"]),
            float(candidate["validation_max_target_harm"]),
            candidate["candidate"],
        ),
    )


def row_for_summary(row: dict[str, str]) -> dict[str, Any]:
    satisfied_raw = row["external_contract_satisfied"]
    satisfied: bool | None
    if satisfied_raw == "NA":
        satisfied = None
    else:
        satisfied = as_bool(satisfied_raw)
    return {
        "deployed": as_bool(row["deployed"]),
        "external_contract_estimable": as_bool(row["external_contract_estimable"]),
        "external_contract_satisfied": satisfied,
        "measured_external_contract_violation": as_bool(
            row["measured_external_contract_violation"]
        ),
        "procedurally_unsupported_deployment": as_bool(
            row["procedurally_unsupported_deployment"]
        ),
    }


def compare_summary(
    observed: dict[str, Any], expected: dict[str, Any], prefix: str, failures: list[str]
) -> None:
    for key, value in expected.items():
        if key not in observed or not values_match(observed[key], value):
            failures.append(f"{prefix}.{key} differs from compact replay")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--abstract", type=Path, default=DEFAULT_ABSTRACT)
    parser.add_argument("--full-audit", type=Path, default=DEFAULT_FULL_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prereg = load_json(args.prereg)
    report = load_json(args.report)
    abstract = load_json(args.abstract)
    full_audit = load_json(args.full_audit)
    rows = load_csv(args.rows)
    candidates = load_csv(args.candidates)
    failures: list[str] = []

    frozen_expectations = {
        "rule_rows_sha256": sha256(args.rows),
        "candidate_rows_sha256": sha256(args.candidates),
        "report_sha256": sha256(args.report),
        "abstract_sha256": sha256(args.abstract),
    }
    if full_audit.get("passed") is not True:
        failures.append("full raw-array audit did not pass before freezing")
    for key, expected in frozen_expectations.items():
        if full_audit.get(key) != expected:
            failures.append(f"full audit hash mismatch: {key}")

    study = prereg["real_study"]
    expected_rule_rows = len(study["datasets"]) * len(study["seeds"]) * len(RULES)
    expected_candidate_rows = len(study["datasets"]) * len(study["seeds"]) * 12
    if len(rows) != expected_rule_rows or len(candidates) != expected_candidate_rows:
        failures.append("frozen row dimensions differ from the locked protocol")

    candidates_by_config: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    rows_by_config: defaultdict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for candidate in candidates:
        candidates_by_config[candidate["config_id"]].append(candidate)
    for row in rows:
        rows_by_config[row["config_id"]][row["rule"]] = row
    if set(candidates_by_config) != set(rows_by_config):
        failures.append("candidate and rule configuration sets differ")

    selection_mismatches = 0
    for config_id, config_candidates in candidates_by_config.items():
        if len(config_candidates) != 12:
            failures.append(f"{config_id} does not contain 12 candidates")
            continue
        if set(rows_by_config[config_id]) != set(RULE_PREDICATES):
            failures.append(f"{config_id} does not contain all deployment rules")
            continue
        for rule, predicate in RULE_PREDICATES.items():
            selected = choose(config_candidates, predicate)
            observed = rows_by_config[config_id][rule]
            expected_candidate = "" if selected is None else selected["candidate"]
            if (
                as_bool(observed["deployed"]) != (selected is not None)
                or observed["selected_candidate"] != expected_candidate
            ):
                selection_mismatches += 1
    if selection_mismatches:
        failures.append(f"{selection_mismatches} compact rule selections differ")

    supported = set(report["supported_datasets"])
    for rule in RULES:
        expected = summarize(
            [
                row_for_summary(row)
                for row in rows
                if row["dataset"] in supported and row["rule"] == rule
            ]
        )
        compare_summary(
            report.get("supported_summaries", {}).get(rule, {}),
            expected,
            f"supported.{rule}",
            failures,
        )

    raw_p: dict[str, float] = {}
    discordance: dict[str, dict[str, int]] = {}
    point_by_key = {
        (row["dataset"], int(row["seed"])): row
        for row in rows
        if row["rule"] == "point_selection_balanced"
    }
    vera_by_key = {
        (row["dataset"], int(row["seed"])): row
        for row in rows
        if row["rule"] == "vera_balanced_iut"
    }
    for dataset in report["supported_datasets"]:
        point_only = sum(
            as_bool(point_by_key[(dataset, int(seed))]["measured_external_contract_violation"])
            and not as_bool(
                vera_by_key[(dataset, int(seed))][
                    "measured_external_contract_violation"
                ]
            )
            for seed in study["seeds"]
        )
        vera_only = sum(
            not as_bool(
                point_by_key[(dataset, int(seed))][
                    "measured_external_contract_violation"
                ]
            )
            and as_bool(
                vera_by_key[(dataset, int(seed))][
                    "measured_external_contract_violation"
                ]
            )
            for seed in study["seeds"]
        )
        discordance[dataset] = {
            "point_only_violation": point_only,
            "vera_only_violation": vera_only,
        }
        raw_p[dataset] = exact_one_sided_mcnemar(point_only, vera_only)
    adjusted_p = holm_adjust(raw_p)
    for dataset, value in raw_p.items():
        if not values_match(report["one_sided_mcnemar_raw_p"][dataset], value):
            failures.append(f"raw McNemar differs for {dataset}")
    for dataset, value in adjusted_p.items():
        if not values_match(report["one_sided_mcnemar_holm_p"][dataset], value):
            failures.append(f"Holm McNemar differs for {dataset}")
    for dataset, values in discordance.items():
        for key, value in values.items():
            if report["one_sided_mcnemar_discordance"][dataset].get(key) != value:
                failures.append(f"discordance differs for {dataset}.{key}")

    supported_point = report["supported_summaries"]["point_selection_balanced"]
    supported_vera = report["supported_summaries"]["vera_balanced_iut"]
    tax = report["certification_tax"]
    expected_abstract = make_abstract_record(
        prereg_hash=report["prereg_sha256"],
        supported_count=len(report["supported_datasets"]) * len(study["seeds"]),
        point_rate=float(supported_point["measured_external_violation_rate"]),
        vera_rate=float(supported_vera["measured_external_violation_rate"]),
        retention=float(tax["safe_retention"]),
        passed=report["passed"] is True,
        camelyon_forced_count=int(report["camelyon_forced_abstention_count"]),
    )
    for key, expected in expected_abstract.items():
        if key not in abstract or not values_match(abstract[key], expected):
            failures.append(f"abstract field differs from compact replay: {key}")

    audit = {
        "name": "VERA independent stress compact frozen-row audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": not failures,
        "confirmatory_passed": report.get("passed") is True,
        "full_raw_audit_sha256": sha256(args.full_audit),
        "full_raw_audit_verified": full_audit.get("passed") is True,
        "rule_rows_replayed": len(rows),
        "candidate_rows_replayed": len(candidates),
        "selection_mismatches": selection_mismatches,
        "headline_verified": not any(
            failure.startswith("abstract field") for failure in failures
        ),
        "rule_rows_sha256": sha256(args.rows),
        "candidate_rows_sha256": sha256(args.candidates),
        "report_sha256": sha256(args.report),
        "abstract_sha256": sha256(args.abstract),
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
