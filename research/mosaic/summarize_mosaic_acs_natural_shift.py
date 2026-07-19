#!/usr/bin/env python3
"""Aggregate every registered ACS natural-shift receipt without selective omission."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from scipy.stats import beta, binomtest

from mosaic_real import sha256


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
DEFAULT_PREREG = ROOT / "prereg_mosaic_acs_natural_shift_v1.json"
DEFAULT_DATA_LOCK = ROOT / "prereg_mosaic_acs_natural_shift_data_v1.json"
DEFAULT_RECEIPTS = REPOSITORY / "research/artifacts/mosaic_acs_natural_shift_v1_receipts"
DEFAULT_OUTPUT = REPOSITORY / "research/artifacts/mosaic_acs_natural_shift_v1_summary.json"


def cp95(successes: int, trials: int) -> list[float] | None:
    if trials == 0:
        return None
    lower = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(0.975, successes + 1, trials - successes))
    return [lower, upper]


def job_key(payload: dict[str, Any]) -> tuple[str, str, int]:
    return str(payload["task"]), str(payload["target_state"]), int(payload["seed"])


def selection(payload: dict[str, Any], alphabet: str, rule: str, threshold: str) -> dict[str, Any]:
    return payload["alphabets"][alphabet]["selection_by_rule_and_threshold"][rule][threshold]


def aggregate_cell(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    deployments = [row for row in values if row["decision"] == "deploy"]
    estimable = [row for row in deployments if row.get("diagnostic_estimable")]
    false_acceptances = sum(bool(row.get("false_acceptance")) for row in deployments)
    operational = [row["operational_replay"] for row in deployments if "operational_replay" in row]
    operational_draws = sum(int(row["draws"]) for row in operational)
    operational_violations = sum(int(row["primary_contract_violations"]) for row in operational)
    return {
        "jobs": len(values),
        "deployments": len(deployments),
        "deployment_rate": len(deployments) / len(values) if values else None,
        "deployment_cp95": cp95(len(deployments), len(values)),
        "abstentions": len(values) - len(deployments),
        "diagnostic_estimable_deployments": len(estimable),
        "diagnostic_safe_deployments": sum(bool(row.get("diagnostic_safe")) for row in estimable),
        "false_acceptances": false_acceptances,
        "false_acceptance_rate_among_estimable": false_acceptances / len(estimable) if estimable else None,
        "false_acceptance_cp95": cp95(false_acceptances, len(estimable)),
        "selected_method_counts": dict(sorted(Counter(row["method"] for row in deployments).items())),
        "operational_draws": operational_draws,
        "operational_contract_violations": operational_violations,
        "operational_violation_rate": operational_violations / operational_draws if operational_draws else None,
        "operational_violation_cp95": cp95(operational_violations, operational_draws),
    }


def matched_deployment(rows: list[dict[str, Any]], alphabet: str, threshold: str) -> dict[str, Any]:
    both = mosaic_only = direct_only = neither = 0
    for payload in rows:
        mosaic = selection(payload, alphabet, "mosaic", threshold)["decision"] == "deploy"
        direct = selection(payload, alphabet, "direct", threshold)["decision"] == "deploy"
        if mosaic and direct:
            both += 1
        elif mosaic:
            mosaic_only += 1
        elif direct:
            direct_only += 1
        else:
            neither += 1
    discordant = mosaic_only + direct_only
    p_value = float(binomtest(min(mosaic_only, direct_only), discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "both_deploy": both,
        "mosaic_only": mosaic_only,
        "direct_only": direct_only,
        "neither": neither,
        "exact_two_sided_mcnemar_p": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--data-lock", type=Path, default=DEFAULT_DATA_LOCK)
    parser.add_argument("--receipts", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    prereg = json.loads(args.prereg.read_text(encoding="utf-8"))
    prereg_sha = sha256(args.prereg)
    data_lock_sha = sha256(args.data_lock)
    expected = {
        (str(row["task"]), str(row["target_state"]), int(row["seed"]))
        for row in prereg["jobs"]
    }
    paths = sorted(args.receipts.glob("ACS-*.json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    observed = {job_key(row) for row in rows}
    if len(rows) != len(expected) or observed != expected:
        raise ValueError(
            f"receipt set differs from lock: expected {len(expected)}, found {len(rows)}, "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    for row in rows:
        if row.get("preregistration_sha256") != prereg_sha:
            raise ValueError(f"receipt has wrong preregistration: {job_key(row)}")
        if not row["puma_partition"]["disjoint"]:
            raise ValueError(f"PUMA overlap: {job_key(row)}")
    thresholds = [f"{float(value):.2f}" for value in prereg["protocol"]["utility_thresholds"]]
    alphabets = [str(value) for value in prereg["protocol"]["fine_token_counts"]]
    cells: dict[str, Any] = {}
    breakdowns: dict[str, Any] = {}
    for alphabet in alphabets:
        cells[alphabet] = {}
        for rule in ("mosaic", "direct"):
            cells[alphabet][rule] = {
                threshold: aggregate_cell(selection(row, alphabet, rule, threshold) for row in rows)
                for threshold in thresholds
            }
        primary = "0.40"
        breakdowns[alphabet] = {"task": {}, "state": {}}
        for field, values in (
            ("task", prereg["protocol"]["tasks"]),
            ("state", prereg["protocol"]["target_states"]),
        ):
            for value in values:
                subset = [row for row in rows if str(row["task" if field == "task" else "target_state"]) == str(value)]
                breakdowns[alphabet][field][str(value)] = {
                    rule: aggregate_cell(selection(row, alphabet, rule, primary) for row in subset)
                    for rule in ("mosaic", "direct")
                }
    optimization_failures = defaultdict(int)
    for row in rows:
        for alphabet in alphabets:
            for candidate in row["alphabets"][alphabet]["rows"]:
                if "optimization_error" in candidate:
                    optimization_failures[f"K={alphabet}:{candidate['method']}"] += 1
    report = {
        "name": "MOSAIC multi-state ACS natural-shift confirmation summary",
        "status": "complete",
        "preregistration_sha256": prereg_sha,
        "data_lock_sha256": data_lock_sha,
        "receipt_count": len(rows),
        "registered_job_count": len(expected),
        "alphabets": alphabets,
        "thresholds": thresholds,
        "cells": cells,
        "primary_matched_deployment": {
            alphabet: matched_deployment(rows, alphabet, "0.40") for alphabet in alphabets
        },
        "primary_breakdowns": breakdowns,
        "optimization_failures": dict(sorted(optimization_failures.items())),
        "claim_boundary": prereg["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
