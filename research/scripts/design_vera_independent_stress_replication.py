"""Use completed design seeds to lock stress contracts for an independent replication.

This script is exploratory by construction. It searches a declared threshold grid on
seeds 5--12 and emits one deterministic contract per supported dataset. Future runs
must use a disjoint seed block and a separately committed preregistration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from analyze_vera_attacker_ablation import load_candidates
from analyze_vera_balanced_existing import (
    balanced_external_metrics,
    balanced_point_metrics,
    balanced_samples,
    choose,
)
from vera_robust_certificate import balanced_contract_certificates


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREG = ROOT / "prereg_confirmatory_balanced.json"
DEFAULT_HASH = ROOT / "prereg_confirmatory_balanced.sha256"
DEFAULT_RECEIPTS = ROOT / "artifacts" / "confirmatory_balanced_receipts"
DEFAULT_OUTPUT = ROOT / "artifacts" / "vera_independent_stress_design.json"

TARGET_THRESHOLDS = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40)
LEAKAGE_THRESHOLDS = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99)


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


def source_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT.parent,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def prepare_seed_candidates(
    receipt_dir: Path,
    study: dict[str, Any],
    dataset: str,
    seed: int,
) -> list[dict[str, Any]]:
    loaded, labels = load_candidates(receipt_dir, study, dataset, seed)
    indices = np.arange(len(labels[0]), dtype=np.int64)
    prepared: list[dict[str, Any]] = []
    local_alpha = float(study["delta"]) / len(loaded)
    gamma = float(study["deployment_gamma"])
    for candidate in loaded:
        target, leakage, source = balanced_samples(candidate["arrays"], indices)
        target_point, leakage_point = balanced_point_metrics(target, leakage, source)
        certificates = balanced_contract_certificates(
            target,
            leakage,
            source,
            gamma=gamma,
            local_failure_probability=local_alpha,
        )
        target_ucb = max(
            certificate.upper_confidence_bound
            for key, certificate in certificates.items()
            if key.startswith("target::")
        )
        leakage_ucb = max(
            certificate.upper_confidence_bound
            for key, certificate in certificates.items()
            if key.startswith("balanced_leakage::")
        )
        external_target, external_leakage = balanced_external_metrics(candidate["arrays"])
        if external_leakage is None:
            raise RuntimeError(f"unsupported external leakage in design set: {dataset}")
        prepared.append(
            {
                "candidate": candidate["candidate"],
                "method": candidate["method"],
                "validation_max_target_harm": float(target_point),
                "validation_max_balanced_leakage": float(leakage_point),
                "iut_max_target_ucb": float(target_ucb),
                "iut_max_balanced_leakage_ucb": float(leakage_ucb),
                "external_max_target_harm": float(external_target),
                "external_max_balanced_leakage": float(external_leakage),
            }
        )
    if len(prepared) != 12:
        raise RuntimeError(f"expected 12 candidates for {dataset}/seed-{seed}")
    return prepared


def selected_outcome(
    candidates: list[dict[str, Any]],
    *,
    target_threshold: float,
    leakage_threshold: float,
    rule: str,
) -> tuple[bool, bool, str]:
    evaluated: list[dict[str, Any]] = []
    for candidate in candidates:
        record = dict(candidate)
        record["point_feasible"] = (
            record["validation_max_target_harm"] <= target_threshold
            and record["validation_max_balanced_leakage"] <= leakage_threshold
        )
        record["iut_eligible"] = (
            record["iut_max_target_ucb"] <= target_threshold
            and record["iut_max_balanced_leakage_ucb"] <= leakage_threshold
        )
        record["external_contract_satisfied"] = (
            record["external_max_target_harm"] <= target_threshold
            and record["external_max_balanced_leakage"] <= leakage_threshold
        )
        evaluated.append(record)
    eligibility = "point_feasible" if rule == "point" else "iut_eligible"
    selected = choose(evaluated, eligibility)
    if selected is None:
        return False, False, ""
    return True, not bool(selected["external_contract_satisfied"]), str(
        selected["candidate"]
    )


def evaluate_regime(
    by_seed: dict[int, list[dict[str, Any]]],
    target_threshold: float,
    leakage_threshold: float,
) -> dict[str, Any]:
    point_violations = 0
    vera_violations = 0
    point_deployments = 0
    vera_deployments = 0
    oracle_opportunities = 0
    point_only = 0
    vera_only = 0
    seed_records: list[dict[str, Any]] = []
    for seed, candidates in sorted(by_seed.items()):
        point_deployed, point_violation, point_candidate = selected_outcome(
            candidates,
            target_threshold=target_threshold,
            leakage_threshold=leakage_threshold,
            rule="point",
        )
        vera_deployed, vera_violation, vera_candidate = selected_outcome(
            candidates,
            target_threshold=target_threshold,
            leakage_threshold=leakage_threshold,
            rule="vera",
        )
        oracle = any(
            candidate["external_max_target_harm"] <= target_threshold
            and candidate["external_max_balanced_leakage"] <= leakage_threshold
            for candidate in candidates
        )
        point_deployments += int(point_deployed)
        vera_deployments += int(vera_deployed)
        point_violations += int(point_violation)
        vera_violations += int(vera_violation)
        oracle_opportunities += int(oracle)
        point_only += int(point_violation and not vera_violation)
        vera_only += int(vera_violation and not point_violation)
        seed_records.append(
            {
                "seed": seed,
                "point_deployed": point_deployed,
                "point_violation": point_violation,
                "point_candidate": point_candidate,
                "vera_deployed": vera_deployed,
                "vera_violation": vera_violation,
                "vera_candidate": vera_candidate,
                "external_oracle_opportunity": oracle,
            }
        )
    count = len(by_seed)
    safe_vera = vera_deployments - vera_violations
    return {
        "target_threshold": target_threshold,
        "leakage_threshold": leakage_threshold,
        "seed_count": count,
        "point_deployment_count": point_deployments,
        "point_violation_count": point_violations,
        "point_violation_rate": point_violations / count,
        "vera_deployment_count": vera_deployments,
        "vera_violation_count": vera_violations,
        "vera_violation_rate": vera_violations / count,
        "external_oracle_opportunity_count": oracle_opportunities,
        "vera_safe_retention": (
            0.0 if oracle_opportunities == 0 else safe_vera / oracle_opportunities
        ),
        "point_only_violation_count": point_only,
        "vera_only_violation_count": vera_only,
        "seed_records": seed_records,
    }


def select_regime(records: list[dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    minimum_point_failures = math.ceil(0.20 * int(records[0]["seed_count"]))
    eligible = [
        record
        for record in records
        if record["vera_violation_count"] == 0
        and record["point_violation_count"] >= minimum_point_failures
    ]
    retained = [record for record in eligible if record["vera_deployment_count"] > 0]
    pool = retained or eligible
    qualified = bool(pool)
    if not pool:
        pool = [record for record in records if record["vera_violation_count"] == 0]
    if not pool:
        pool = records
    selected = max(
        pool,
        key=lambda record: (
            int(record["point_only_violation_count"]),
            int(record["vera_deployment_count"] > 0),
            int(record["vera_deployment_count"]),
            float(record["vera_safe_retention"]),
            int(record["point_deployment_count"]),
            -float(record["leakage_threshold"]),
            -float(record["target_threshold"]),
        ),
    )
    return selected, qualified


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--hash-file", type=Path, default=DEFAULT_HASH)
    parser.add_argument("--receipt-dir", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prereg_hash = sha256(args.prereg)
    expected_hash = args.hash_file.read_text(encoding="utf-8").split()[0]
    if prereg_hash != expected_hash:
        raise RuntimeError("design-data preregistration hash mismatch")
    prereg = load_json(args.prereg)
    study = prereg["real_study"]
    seeds = [int(seed) for seed in study["seeds"]]
    datasets = {
        dataset: config
        for dataset, config in study["datasets"].items()
        if not config.get("force_abstain_for_unsupported_environment")
    }
    report_datasets: dict[str, Any] = {}
    all_qualified = True
    for dataset in datasets:
        by_seed = {
            seed: prepare_seed_candidates(args.receipt_dir, study, dataset, seed)
            for seed in seeds
        }
        records = [
            evaluate_regime(by_seed, target_threshold, leakage_threshold)
            for target_threshold in TARGET_THRESHOLDS
            for leakage_threshold in LEAKAGE_THRESHOLDS
        ]
        selected, qualified = select_regime(records)
        all_qualified = all_qualified and qualified
        ranked = sorted(
            records,
            key=lambda record: (
                int(record["vera_violation_count"] == 0),
                int(record["point_only_violation_count"]),
                int(record["vera_deployment_count"] > 0),
                int(record["vera_deployment_count"]),
            ),
            reverse=True,
        )
        report_datasets[dataset] = {
            "selection_passed_minimum_design_criteria": qualified,
            "selected_regime": selected,
            "top_10_regimes": ranked[:10],
        }
    report = {
        "name": "VERA independent stress-replication design study",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": all_qualified,
        "analysis_tier": "exploratory_design_only",
        "design_data_prereg_sha256": prereg_hash,
        "design_data_seeds": seeds,
        "future_replication_must_use_disjoint_seeds": True,
        "source_commit": source_commit(),
        "candidate_count_per_dataset_seed": 12,
        "gamma": float(study["deployment_gamma"]),
        "delta": float(study["delta"]),
        "target_threshold_grid": list(TARGET_THRESHOLDS),
        "leakage_threshold_grid": list(LEAKAGE_THRESHOLDS),
        "selection_rule": (
            "Require zero VERA external violations and at least 20% point-selection "
            "violations on design seeds; prefer regimes with a VERA deployment, then "
            "maximize point-only discordance, VERA deployment, safe retention, and "
            "point deployment with deterministic threshold tie-breaks."
        ),
        "datasets": report_datasets,
        "warning": (
            "These results selected future contracts and are not confirmatory evidence. "
            "Only a separately committed, disjoint-seed replication may test them."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
