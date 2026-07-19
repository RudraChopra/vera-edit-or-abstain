#!/usr/bin/env python3
"""Independently audit MOSAIC's locked real-data comparator extension."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit_mosaic_bridge_frontier import diagnostic_from_counts, table_from_counts
from mosaic_bridge import certify_bridge_membership
from mosaic_invariant import (
    adaptive_pre_release_attacker_certificate,
    pre_release_utility_certificate,
)
from mosaic_real import sha256
from mosaic_transform_exact import (
    transform_exact_attacker_confidence_bound,
    transform_exact_utility_confidence_bound,
)
from run_mosaic_bridge_comparator_extension import RULES, validate_lock
from run_mosaic_official_frontier_exact_confirmation import atomic_json_dump


TOLERANCE = 3e-7
DECISION_TOLERANCE = 1e-10


def close(first: object, second: object, tolerance: float = TOLERANCE) -> bool:
    if first is None or second is None:
        return first is second
    return abs(float(first) - float(second)) <= tolerance


def compare_diagnostic(
    stored: dict[str, object], expected: dict[str, object]
) -> list[str]:
    failures: list[str] = []
    if bool(stored["estimable"]) != bool(expected["estimable"]):
        failures.append("diagnostic estimability mismatch")
    if stored.get("missing_strata") != expected.get("missing_strata"):
        failures.append("diagnostic missing-strata mismatch")
    for key in ("worst_privacy_advantage", "worst_conditional_error"):
        if not close(stored.get(key), expected.get(key)):
            failures.append(f"diagnostic {key} mismatch")
    return failures


def exact_risks(
    reference: np.ndarray,
    radii: np.ndarray,
    release: dict[str, object],
    *,
    libraries: object,
    contaminations: object,
) -> tuple[list[float], float]:
    channel = np.asarray(release["release_channel"], dtype=np.float64)
    decoder = tuple(int(value) for value in release["decoder"])
    privacy = [
        transform_exact_attacker_confidence_bound(
            reference[label],
            channel,
            l1_radii=radii[label],
            common_fine_token_channels=libraries[label],
            contamination=float(contaminations[label]),
        ).normalized_advantage
        for label in range(2)
    ]
    utility = [
        transform_exact_utility_confidence_bound(
            reference[label, source],
            channel,
            decoder,
            true_label=label,
            l1_radius=float(radii[label, source]),
            common_fine_token_channels=libraries[label],
            contamination=float(contaminations[label]),
        ).error_probability
        for label in range(2)
        for source in range(2)
    ]
    return [float(value) for value in privacy], float(max(utility))


def capacity_risks(
    reference: np.ndarray,
    radii: np.ndarray,
    release: dict[str, object],
    *,
    libraries: object,
    contaminations: object,
) -> tuple[list[float], float]:
    channel = np.asarray(release["release_channel"], dtype=np.float64)
    decoder = tuple(int(value) for value in release["decoder"])
    privacy = [
        adaptive_pre_release_attacker_certificate(
            reference[label],
            channel,
            l1_radii=radii[label],
            common_fine_token_channels=libraries[label],
            contamination=float(contaminations[label]),
        ).normalized_advantage
        for label in range(2)
    ]
    utility = [
        pre_release_utility_certificate(
            reference[label, source],
            channel,
            decoder,
            true_label=label,
            l1_radius=float(radii[label, source]),
            common_fine_token_channels=libraries[label],
            contamination=float(contaminations[label]),
        ).error_probability
        for label in range(2)
        for source in range(2)
    ]
    return [float(value) for value in privacy], float(max(utility))


def audit_release(
    release: dict[str, object],
    *,
    reference: np.ndarray,
    radii: np.ndarray,
    diagnostic_counts: object,
    libraries: object,
    contaminations: object,
    source_threshold: float,
    utility_thresholds: list[float],
    family: str,
) -> list[str]:
    failures: list[str] = []
    channel = np.asarray(release["release_channel"], dtype=np.float64)
    if channel.ndim != 2 or np.any(channel < -TOLERANCE):
        failures.append("invalid release channel")
        return failures
    if not np.allclose(channel.sum(axis=1), 1.0, atol=TOLERANCE, rtol=0.0):
        failures.append("release channel is not row stochastic")
    if family == "capacity":
        privacy, utility = capacity_risks(
            reference,
            radii,
            release,
            libraries=libraries,
            contaminations=contaminations,
        )
    elif family == "exact":
        privacy, utility = exact_risks(
            reference,
            radii,
            release,
            libraries=libraries,
            contaminations=contaminations,
        )
    else:
        raise ValueError(f"unknown certificate family: {family}")
    if not np.allclose(
        privacy,
        np.asarray(release["selection_source_advantages"], dtype=np.float64),
        atol=TOLERANCE,
        rtol=0.0,
    ):
        failures.append("source-advantage recomputation mismatch")
    if not close(utility, release["selection_worst_conditional_error"]):
        failures.append("utility recomputation mismatch")
    if not close(utility, release["solver_objective"]):
        failures.append("solver objective mismatch")
    if float(release["solver_mip_gap"]) > 1e-10:
        failures.append("solver optimality gap exceeds the exact threshold")
    if float(release["max_constraint_violation"]) > TOLERANCE:
        failures.append("solver constraint violation exceeds tolerance")

    expected_diagnostic = diagnostic_from_counts(
        diagnostic_counts,
        channel,
        tuple(int(value) for value in release["decoder"]),
    )
    failures.extend(compare_diagnostic(release["diagnostic"], expected_diagnostic))
    for threshold in utility_thresholds:
        key = f"{threshold:.2f}"
        stored = release["threshold_decisions"][key]
        deployed = bool(
            max(privacy) <= source_threshold + DECISION_TOLERANCE
            and utility <= threshold + DECISION_TOLERANCE
        )
        safe = bool(
            expected_diagnostic["estimable"]
            and float(expected_diagnostic["worst_privacy_advantage"])
            <= source_threshold + DECISION_TOLERANCE
            and float(expected_diagnostic["worst_conditional_error"])
            <= threshold + DECISION_TOLERANCE
        )
        false_acceptance = bool(
            deployed and expected_diagnostic["estimable"] and not safe
        )
        if stored != {
            "deployed": deployed,
            "diagnostic_safe": safe,
            "false_acceptance": false_acceptance,
        }:
            failures.append(f"threshold decision mismatch at {key}")
    return failures


def expected_selection(
    rows: list[dict[str, object]],
    *,
    rule: str,
    release_key: str,
    source_threshold: float,
    utility_threshold: float,
    force_deploy: bool,
) -> dict[str, object]:
    key = f"{utility_threshold:.2f}"
    available = [row for row in rows if isinstance(row.get(release_key), dict)]
    eligible = (
        available
        if force_deploy
        else [
            row
            for row in available
            if row[release_key]["threshold_decisions"][key]["deployed"] is True
        ]
    )
    if not eligible:
        return {
            "rule": rule,
            "decision": "abstain",
            "candidate": None,
            "utility_threshold": utility_threshold,
            "reason": "no available candidate satisfied the rule",
        }
    selected = min(
        eligible,
        key=lambda row: (
            float(row[release_key]["selection_worst_conditional_error"]),
            str(row["candidate"]),
        ),
    )
    release = selected[release_key]
    diagnostic = release["diagnostic"]
    safe = bool(
        diagnostic["estimable"]
        and float(diagnostic["worst_privacy_advantage"])
        <= source_threshold + DECISION_TOLERANCE
        and float(diagnostic["worst_conditional_error"])
        <= utility_threshold + DECISION_TOLERANCE
    )
    deployed = bool(force_deploy or release["threshold_decisions"][key]["deployed"])
    return {
        "rule": rule,
        "decision": "deploy" if deployed else "abstain",
        "candidate": selected["candidate"],
        "method": selected["method"],
        "strength": selected["strength"],
        "utility_threshold": utility_threshold,
        "selection_worst_conditional_error": release[
            "selection_worst_conditional_error"
        ],
        "selection_source_advantages": release["selection_source_advantages"],
        "diagnostic_estimable": diagnostic["estimable"],
        "diagnostic_worst_source_advantage": diagnostic[
            "worst_privacy_advantage"
        ],
        "diagnostic_worst_conditional_error": diagnostic[
            "worst_conditional_error"
        ],
        "diagnostic_safe": safe,
        "false_acceptance": bool(deployed and diagnostic["estimable"] and not safe),
    }


def mappings_close(first: object, second: object) -> bool:
    if isinstance(first, dict) and isinstance(second, dict):
        return set(first) == set(second) and all(
            mappings_close(first[key], second[key]) for key in first
        )
    if isinstance(first, list) and isinstance(second, list):
        return len(first) == len(second) and all(
            mappings_close(left, right) for left, right in zip(first, second, strict=True)
        )
    if isinstance(first, (float, int)) and isinstance(second, (float, int)):
        if isinstance(first, bool) or isinstance(second, bool):
            return first is second
        return close(first, second)
    return first == second


def audit_one(
    original_path: Path,
    comparator_path: Path,
    *,
    original_prereg_hash: str,
    comparator_lock_hash: str,
) -> dict[str, object]:
    failures: list[str] = []
    original = json.loads(original_path.read_text(encoding="utf-8"))
    comparator = json.loads(comparator_path.read_text(encoding="utf-8"))
    if comparator.get("original_receipt_sha256") != sha256(original_path):
        failures.append("original receipt hash mismatch")
    if comparator.get("original_preregistration_sha256") != original_prereg_hash:
        failures.append("original preregistration hash mismatch")
    if comparator.get("comparator_lock_sha256") != comparator_lock_hash:
        failures.append("comparator lock hash mismatch")
    if comparator.get("rules") != list(RULES):
        failures.append("registered rule list mismatch")
    if (comparator.get("dataset"), comparator.get("seed")) != (
        original.get("dataset"),
        original.get("seed"),
    ):
        failures.append("dataset or seed mismatch")

    protocol = original["protocol"]
    token_count = int(protocol["fine_token_count"])
    table_delta = float(protocol["per_candidate_table_delta"])
    source_threshold = float(protocol["privacy_advantage_threshold"])
    utility_thresholds = [float(value) for value in protocol["utility_thresholds"]]
    original_by_candidate = {row["candidate"]: row for row in original["results"]}
    rows = comparator["results"]
    if len(rows) != len(original_by_candidate):
        failures.append("candidate row count mismatch")

    identity = (np.eye(token_count, dtype=np.float64),)
    for row in rows:
        prefix = str(row.get("candidate"))
        original_row = original_by_candidate.get(prefix)
        if original_row is None:
            failures.append(f"{prefix}: candidate absent from original receipt")
            continue
        if "upstream_error" in row:
            if row["upstream_error"] != original_row.get("optimization_error"):
                failures.append(f"{prefix}: upstream error mismatch")
            continue
        reference, radii, _ = table_from_counts(
            original_row["reference_token_counts"],
            token_count=token_count,
            familywise_delta=table_delta,
        )
        bridge, bridge_radii, bridge_totals = table_from_counts(
            original_row["bridge_token_counts"],
            token_count=token_count,
            familywise_delta=table_delta,
        )
        robust = certify_bridge_membership(
            reference,
            reference_l1_radii=radii,
            bridge_empirical_distributions=bridge,
            bridge_l1_radii=bridge_radii,
        )
        release_specs: list[tuple[str, np.ndarray, object, object, str]] = [
            (
                "mosaic_transform_exact",
                radii,
                robust.transforms_by_label,
                robust.contaminations,
                "exact",
            ),
            (
                "capacity_transfer",
                radii,
                robust.transforms_by_label,
                robust.contaminations,
                "capacity",
            ),
            (
                "validation_plugin",
                np.zeros_like(radii),
                (identity, identity),
                (0.0, 0.0),
                "exact",
            ),
        ]
        if not np.any(bridge_totals == 0):
            plugin = certify_bridge_membership(
                reference,
                reference_l1_radii=np.zeros_like(radii),
                bridge_empirical_distributions=bridge,
                bridge_l1_radii=np.zeros_like(bridge_radii),
            )
            release_specs.append(
                (
                    "bridge_plugin",
                    np.zeros_like(radii),
                    plugin.transforms_by_label,
                    plugin.contaminations,
                    "exact",
                )
            )
        elif row.get("bridge_plugin_error") != "missing bridge source-label stratum":
            failures.append(f"{prefix}: missing-support bridge plug-in mismatch")
        for key, current_radii, libraries, contaminations, family in release_specs:
            if not isinstance(row.get(key), dict):
                failures.append(f"{prefix}: missing {key} release")
                continue
            current = audit_release(
                row[key],
                reference=reference,
                radii=current_radii,
                diagnostic_counts=original_row["diagnostic_token_counts"],
                libraries=libraries,
                contaminations=contaminations,
                source_threshold=source_threshold,
                utility_thresholds=utility_thresholds,
                family=family,
            )
            failures.extend(f"{prefix} {key}: {failure}" for failure in current)

    release_keys = {
        "mosaic_transform_exact": "mosaic_transform_exact",
        "capacity_transfer": "capacity_transfer",
        "bridge_plugin": "bridge_plugin",
        "validation_plugin": "validation_plugin",
        "always_deploy_validation": "validation_plugin",
    }
    stored_selections = comparator["selection_by_rule_and_utility_threshold"]
    for rule, release_key in release_keys.items():
        for threshold in utility_thresholds:
            key = f"{threshold:.2f}"
            expected = expected_selection(
                rows,
                rule=rule,
                release_key=release_key,
                source_threshold=source_threshold,
                utility_threshold=threshold,
                force_deploy=rule == "always_deploy_validation",
            )
            if not mappings_close(stored_selections[rule][key], expected):
                failures.append(f"{rule} selection mismatch at {key}")
    return {
        "dataset": comparator.get("dataset"),
        "seed": comparator.get("seed"),
        "original": str(original_path),
        "comparator": str(comparator_path),
        "candidate_rows": len(rows),
        "failure_count": len(failures),
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--comparator-dir", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    lock, lock_hash = validate_lock(args.lock)
    originals = sorted(args.raw_dir.glob("*.json"))
    comparators = {path.name: path for path in args.comparator_dir.glob("*.json")}
    if len(originals) != int(lock["required_raw_receipt_count"]):
        raise ValueError("raw receipt count differs from the lock")
    if set(path.name for path in originals) != set(comparators):
        raise ValueError("raw and comparator receipt filenames differ")
    rows: list[dict[str, object] | None] = [None] * len(originals)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                audit_one,
                original,
                comparators[original.name],
                original_prereg_hash=str(lock["original_preregistration_sha256"]),
                comparator_lock_hash=lock_hash,
            ): index
            for index, original in enumerate(originals)
        }
        for future in as_completed(futures):
            index = futures[future]
            rows[index] = future.result()
            print(f"audited {rows[index]['comparator']}", flush=True)
    completed = [row for row in rows if row is not None]
    payload = {
        "name": "MOSAIC real bridge comparator independent certificate audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "comparator_lock_sha256": lock_hash,
        "file_count": len(completed),
        "candidate_rows": sum(int(row["candidate_rows"]) for row in completed),
        "failure_count": sum(int(row["failure_count"]) for row in completed),
        "passed": all(int(row["failure_count"]) == 0 for row in completed),
        "files": completed,
    }
    atomic_json_dump(payload, args.output)
    if not payload["passed"]:
        raise AssertionError("comparator audit found failures")


if __name__ == "__main__":
    main()
