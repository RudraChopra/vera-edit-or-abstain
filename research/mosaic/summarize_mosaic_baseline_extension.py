#!/usr/bin/env python3
"""Summarize the audited paired MOSAIC baseline extension."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from scipy.stats import beta, binomtest

from run_mosaic_official_frontier_exact_confirmation import atomic_json_dump


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
DEFAULT_PREREG = ROOT / "prereg_mosaic_baseline_extension_v1.json"
DEFAULT_REPORT = (
    REPOSITORY
    / "research/artifacts/mosaic_baseline_extension_v1_schema_repaired.json"
)
DEFAULT_AUDIT = (
    REPOSITORY / "research/artifacts/mosaic_baseline_extension_audit_v1.json"
)
DEFAULT_OUTPUT = (
    REPOSITORY / "research/artifacts/mosaic_baseline_extension_summary_v1.json"
)
PRIMARY_METHOD = "mosaic_continuum"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def exact_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> list[float] | None:
    if trials == 0:
        return None
    alpha = 1.0 - confidence
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(
            beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes)
        )
    )
    return [lower, upper]


def aggregate_cell(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    trials = len(rows)
    deployments = sum(bool(row["deployed"]) for row in rows)
    safe_deployments = sum(
        bool(row["deployed"]) and bool(row["exact_safe"]) for row in rows
    )
    false_acceptances = sum(bool(row["false_acceptance"]) for row in rows)
    return {
        "trials": trials,
        "deployments": deployments,
        "deployment_rate": deployments / trials if trials else None,
        "deployment_exact_95_interval": exact_interval(deployments, trials),
        "safe_deployments": safe_deployments,
        "safe_deployment_rate": safe_deployments / trials if trials else None,
        "false_acceptances": false_acceptances,
        "false_acceptance_rate": false_acceptances / trials if trials else None,
        "false_acceptance_exact_95_interval": exact_interval(
            false_acceptances, trials
        ),
    }


def paired_contrast(
    rows: Sequence[dict[str, object]], *, comparator: str
) -> dict[str, object]:
    by_seed: dict[int, dict[str, bool]] = defaultdict(dict)
    for row in rows:
        by_seed[int(row["seed"])][str(row["method"])] = bool(row["deployed"])
    primary_only = 0
    comparator_only = 0
    both = 0
    neither = 0
    for seed, decisions in by_seed.items():
        if set(decisions) != {PRIMARY_METHOD, comparator}:
            raise ValueError(f"paired comparison is incomplete for seed {seed}")
        primary = decisions[PRIMARY_METHOD]
        other = decisions[comparator]
        primary_only += int(primary and not other)
        comparator_only += int(other and not primary)
        both += int(primary and other)
        neither += int(not primary and not other)
    discordant = primary_only + comparator_only
    p_value = (
        float(binomtest(primary_only, discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "pairs": len(by_seed),
        "mosaic_only_deployments": primary_only,
        "comparator_only_deployments": comparator_only,
        "both_deploy": both,
        "neither_deploys": neither,
        "paired_deployment_difference": (
            (primary_only - comparator_only) / len(by_seed) if by_seed else None
        ),
        "exact_two_sided_mcnemar_p": p_value,
    }


def summarize(
    prereg_path: Path, report_path: Path, audit_path: Path
) -> dict[str, object]:
    prereg = load_object(prereg_path)
    report = load_object(report_path)
    audit = load_object(audit_path)
    if audit.get("pass") is not True:
        raise ValueError("baseline replay audit did not pass")
    if audit.get("baseline_report_sha256") != sha256(report_path):
        raise ValueError("audit does not authenticate the baseline report")
    if audit.get("baseline_preregistration_sha256") != sha256(prereg_path):
        raise ValueError("audit does not authenticate the baseline preregistration")

    rows = report.get("replicate_results")
    if not isinstance(rows, list):
        raise ValueError("baseline report has no replicate rows")
    expected_rows = int(prereg["pass_conditions"]["complete_rows"])
    if len(rows) != expected_rows:
        raise ValueError("baseline replicate grid is incomplete")
    methods = sorted(str(value) for value in prereg["methods"])
    scenarios = sorted(str(value["name"]) for value in prereg["scenarios"])
    expected_replicates = int(prereg["replicates_per_cell"])
    indexed: dict[tuple[str, int, str], dict[str, object]] = {}
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("baseline replicate row must be an object")
        key = (str(row["scenario"]), int(row["seed"]), str(row["method"]))
        if key in indexed:
            raise ValueError(f"duplicate baseline replicate: {key}")
        indexed[key] = row
        grouped[(key[0], key[2])].append(row)
    expected_cells = {(scenario, method) for scenario in scenarios for method in methods}
    if set(grouped) != expected_cells or any(
        len(grouped[key]) != expected_replicates for key in expected_cells
    ):
        raise ValueError("baseline scenario-method grid is incomplete")

    cells = {
        scenario: {
            method: aggregate_cell(grouped[(scenario, method)])
            for method in methods
        }
        for scenario in scenarios
    }
    contrasts: dict[str, object] = {}
    for scenario in scenarios:
        contrasts[scenario] = {}
        for comparator in methods:
            if comparator == PRIMARY_METHOD:
                continue
            pair_rows = grouped[(scenario, PRIMARY_METHOD)] + grouped[
                (scenario, comparator)
            ]
            contrasts[scenario][comparator] = paired_contrast(
                pair_rows, comparator=comparator
            )

    retention_name = "retention_and_stochastic_value"
    return {
        "name": "MOSAIC audited paired baseline summary v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "preregistration_sha256": sha256(prereg_path),
        "report_sha256": sha256(report_path),
        "audit_sha256": sha256(audit_path),
        "audit_status": audit.get("status"),
        "replicate_rows": len(rows),
        "scenarios": scenarios,
        "methods": methods,
        "cells": cells,
        "paired_contrasts": contrasts,
        "primary_retention_cell": cells[retention_name][PRIMARY_METHOD],
        "primary_ltt_contrast": contrasts[retention_name]["holm_ltt_grid"],
        "reporting_note": (
            "Rates and pairings are deterministic summaries of the locked audited "
            "comparison. The exact McNemar p-values were added after the locked run "
            "and are descriptive, not a preregistered pass condition."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = summarize(args.prereg, args.report, args.audit)
    atomic_json_dump(payload, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
