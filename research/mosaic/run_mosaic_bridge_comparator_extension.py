#!/usr/bin/env python3
"""Replay locked bridge tables under preregistered deployment comparators."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from audit_mosaic_bridge_frontier import diagnostic_from_counts, table_from_counts
from mosaic_bridge import certify_bridge_membership
from mosaic_optimizer import optimize_invariant_channel
from mosaic_real import sha256
from mosaic_transform_exact_optimizer import optimize_transform_exact_channel
from run_mosaic_official_frontier_exact_confirmation import atomic_json_dump


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
ORIGINAL_PREREG = ROOT / "prereg_mosaic_bridge_v1.json"
RULES = (
    "mosaic_transform_exact",
    "capacity_transfer",
    "bridge_plugin",
    "validation_plugin",
    "always_deploy_validation",
)
POSTHOC_TOLERANCE = 1e-10


def threshold_key(value: float) -> str:
    return f"{float(value):.2f}"


def validate_lock(path: Path) -> tuple[dict[str, object], str]:
    lock_hash = sha256(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.exists() or sidecar.read_text(encoding="utf-8").strip() != lock_hash:
        raise ValueError("comparator lock sidecar mismatch")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("status") != "locked_during_raw_run_before_content_inspection":
        raise ValueError("comparator extension is not locked")
    if sha256(ORIGINAL_PREREG) != lock.get("original_preregistration_sha256"):
        raise ValueError("original bridge preregistration hash mismatch")
    relative_lock = path.resolve().relative_to(REPOSITORY.resolve())
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative_lock.as_posix()}"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    ).stdout
    if committed != path.read_bytes():
        raise ValueError("comparator lock is not the committed lock")
    for relative, expected in lock.get("code_sha256", {}).items():
        if sha256(REPOSITORY / relative) != expected:
            raise ValueError(f"locked comparator code mismatch: {relative}")
    return lock, lock_hash


def diagnostic_decision(
    diagnostic: dict[str, object], *, source_threshold: float, utility_threshold: float
) -> tuple[bool, bool]:
    safe = bool(
        diagnostic["estimable"]
        and float(diagnostic["worst_privacy_advantage"])
        <= source_threshold + POSTHOC_TOLERANCE
        and float(diagnostic["worst_conditional_error"])
        <= utility_threshold + POSTHOC_TOLERANCE
    )
    return safe, bool(diagnostic["estimable"] and not safe)


def serialize_solution(
    solution: object,
    *,
    risk_bound_kind: str,
    diagnostic_counts: object,
    source_threshold: float,
    utility_thresholds: list[float],
) -> dict[str, object]:
    source_advantages = [
        float(value.normalized_advantage) for value in solution.privacy_certificates
    ]
    selection_error = float(solution.certified_worst_conditional_error)
    diagnostic = diagnostic_from_counts(
        diagnostic_counts, solution.release_channel, solution.decoder
    )
    decisions: dict[str, object] = {}
    for threshold in utility_thresholds:
        deployed = bool(
            max(source_advantages) <= source_threshold + POSTHOC_TOLERANCE
            and selection_error <= threshold + POSTHOC_TOLERANCE
        )
        diagnostic_safe, diagnostic_violation = diagnostic_decision(
            diagnostic,
            source_threshold=source_threshold,
            utility_threshold=threshold,
        )
        decisions[threshold_key(threshold)] = {
            "deployed": deployed,
            "diagnostic_safe": diagnostic_safe,
            "false_acceptance": bool(deployed and diagnostic_violation),
        }
    return {
        "method": solution.method,
        "risk_bound_kind": risk_bound_kind,
        "released_token_count": int(solution.release_channel.shape[1]),
        "selection_worst_conditional_error": selection_error,
        "selection_source_advantages": source_advantages,
        "release_channel": solution.release_channel.tolist(),
        "decoder": list(solution.decoder),
        "solver_objective": float(solution.solver_objective),
        "solver_status": solution.solver_status,
        "solver_mip_gap": float(solution.solver_mip_gap),
        "solver_dual_bound": float(solution.solver_dual_bound),
        "max_constraint_violation": float(solution.max_constraint_violation),
        "solved_decoder_assignments": int(solution.solved_decoder_assignments),
        "diagnostic": diagnostic,
        "threshold_decisions": decisions,
    }


def normalize_mosaic_release(release: dict[str, object]) -> dict[str, object]:
    return {
        "method": release["certificate_method"],
        "risk_bound_kind": "simultaneous_confidence_transform_exact",
        "released_token_count": release["released_token_count"],
        "selection_worst_conditional_error": release[
            "certified_worst_conditional_error"
        ],
        "selection_source_advantages": release["certified_privacy_advantages"],
        "release_channel": release["release_channel"],
        "decoder": release["decoder"],
        "solver_objective": release["solver_objective"],
        "solver_status": release["solver_status"],
        "solver_mip_gap": release["solver_mip_gap"],
        "solver_dual_bound": release["solver_dual_bound"],
        "max_constraint_violation": release["max_constraint_violation"],
        "solved_decoder_assignments": release["solved_decoder_assignments"],
        "diagnostic": {
            "estimable": release["diagnostic_estimable"],
            "worst_privacy_advantage": release[
                "diagnostic_worst_privacy_advantage"
            ],
            "worst_conditional_error": release[
                "diagnostic_worst_conditional_error"
            ],
            "missing_strata": release["missing_diagnostic_strata"],
        },
        "threshold_decisions": release["threshold_decisions"],
    }


def serialize_membership(certificate: object) -> dict[str, object]:
    return {
        "method": certificate.method,
        "retained_masses": list(certificate.retained_masses),
        "contaminations": list(certificate.contaminations),
        "transforms_by_label": [
            [transform.tolist() for transform in transforms]
            for transforms in certificate.transforms_by_label
        ],
        "minimum_membership_slacks": [
            float(label.minimum_membership_slack) for label in certificate.labels
        ],
    }


def solve_candidate(
    original_row: dict[str, object], protocol: dict[str, object]
) -> dict[str, object]:
    metadata = {
        key: original_row[key]
        for key in ("candidate", "method", "strength", "provenance")
    }
    if "optimization_error" in original_row:
        return {**metadata, "upstream_error": original_row["optimization_error"]}

    token_count = int(protocol["fine_token_count"])
    table_delta = float(protocol["per_candidate_table_delta"])
    released_count = int(protocol["primary_released_token_count"])
    source_threshold = float(protocol["privacy_advantage_threshold"])
    utility_thresholds = [float(value) for value in protocol["utility_thresholds"]]
    reference, reference_radii, _ = table_from_counts(
        original_row["reference_token_counts"],
        token_count=token_count,
        familywise_delta=table_delta,
    )
    bridge, bridge_radii, bridge_totals = table_from_counts(
        original_row["bridge_token_counts"],
        token_count=token_count,
        familywise_delta=table_delta,
    )
    diagnostic_counts = original_row["diagnostic_token_counts"]
    robust_membership = certify_bridge_membership(
        reference,
        reference_l1_radii=reference_radii,
        bridge_empirical_distributions=bridge,
        bridge_l1_radii=bridge_radii,
    )
    row: dict[str, object] = {
        **metadata,
        "reference_token_counts": original_row["reference_token_counts"],
        "bridge_token_counts": original_row["bridge_token_counts"],
        "diagnostic_token_counts": diagnostic_counts,
        "robust_bridge_membership": serialize_membership(robust_membership),
        "mosaic_transform_exact": normalize_mosaic_release(original_row["release_l2"]),
    }

    capacity = optimize_invariant_channel(
        reference,
        l1_radii=reference_radii,
        common_channels_by_label=robust_membership.transforms_by_label,
        contaminations=robust_membership.contaminations,
        privacy_advantage_thresholds=(source_threshold, source_threshold),
        released_token_count=released_count,
        solver_time_limit_seconds=300.0,
    )
    row["capacity_transfer"] = serialize_solution(
        capacity,
        risk_bound_kind="simultaneous_confidence_generic_capacity",
        diagnostic_counts=diagnostic_counts,
        source_threshold=source_threshold,
        utility_thresholds=utility_thresholds,
    )

    if np.any(bridge_totals == 0):
        row["bridge_plugin_error"] = "missing bridge source-label stratum"
    else:
        zero_radii = np.zeros_like(reference_radii)
        plugin_membership = certify_bridge_membership(
            reference,
            reference_l1_radii=zero_radii,
            bridge_empirical_distributions=bridge,
            bridge_l1_radii=np.zeros_like(bridge_radii),
        )
        row["plugin_bridge_membership"] = serialize_membership(plugin_membership)
        bridge_plugin = optimize_transform_exact_channel(
            reference,
            l1_radii=zero_radii,
            common_channels_by_label=plugin_membership.transforms_by_label,
            contaminations=plugin_membership.contaminations,
            privacy_advantage_thresholds=(source_threshold, source_threshold),
            released_token_count=released_count,
            solver_time_limit_seconds=300.0,
        )
        row["bridge_plugin"] = serialize_solution(
            bridge_plugin,
            risk_bound_kind="point_estimate_bridge_no_intervals",
            diagnostic_counts=diagnostic_counts,
            source_threshold=source_threshold,
            utility_thresholds=utility_thresholds,
        )

    identity = (np.eye(token_count, dtype=np.float64),)
    validation_plugin = optimize_transform_exact_channel(
        reference,
        l1_radii=np.zeros_like(reference_radii),
        common_channels_by_label=(identity, identity),
        contaminations=(0.0, 0.0),
        privacy_advantage_thresholds=(source_threshold, source_threshold),
        released_token_count=released_count,
        solver_time_limit_seconds=300.0,
    )
    row["validation_plugin"] = serialize_solution(
        validation_plugin,
        risk_bound_kind="reference_point_estimate_no_shift_no_intervals",
        diagnostic_counts=diagnostic_counts,
        source_threshold=source_threshold,
        utility_thresholds=utility_thresholds,
    )
    return row


def select_candidate(
    rows: list[dict[str, object]],
    *,
    rule: str,
    release_key: str,
    source_threshold: float,
    utility_threshold: float,
    force_deploy: bool = False,
) -> dict[str, object]:
    key = threshold_key(utility_threshold)
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
    decision = release["threshold_decisions"][key]
    diagnostic = release["diagnostic"]
    diagnostic_safe, diagnostic_violation = diagnostic_decision(
        diagnostic,
        source_threshold=source_threshold,
        utility_threshold=utility_threshold,
    )
    deployed = True if force_deploy else bool(decision["deployed"])
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
        "diagnostic_safe": diagnostic_safe,
        "false_acceptance": bool(deployed and diagnostic_violation),
    }


def replay_one(
    original_path: Path,
    output_path: Path,
    *,
    original_prereg_hash: str,
    comparator_lock_hash: str,
) -> dict[str, object]:
    original = json.loads(original_path.read_text(encoding="utf-8"))
    if original.get("prereg_sha256") != original_prereg_hash:
        raise ValueError(f"{original_path} has the wrong bridge preregistration hash")
    protocol = original["protocol"]
    rows = [solve_candidate(row, protocol) for row in original["results"]]
    thresholds = [float(value) for value in protocol["utility_thresholds"]]
    source_threshold = float(protocol["privacy_advantage_threshold"])
    release_keys = {
        "mosaic_transform_exact": "mosaic_transform_exact",
        "capacity_transfer": "capacity_transfer",
        "bridge_plugin": "bridge_plugin",
        "validation_plugin": "validation_plugin",
    }
    selections: dict[str, object] = {
        rule: {
            threshold_key(threshold): select_candidate(
                rows,
                rule=rule,
                release_key=release_key,
                source_threshold=source_threshold,
                utility_threshold=threshold,
            )
            for threshold in thresholds
        }
        for rule, release_key in release_keys.items()
    }
    selections["always_deploy_validation"] = {
        threshold_key(threshold): select_candidate(
            rows,
            rule="always_deploy_validation",
            release_key="validation_plugin",
            source_threshold=source_threshold,
            utility_threshold=threshold,
            force_deploy=True,
        )
        for threshold in thresholds
    }
    payload: dict[str, object] = {
        "project": "MOSAIC real bridge deployment-rule comparator extension",
        "dataset": original["dataset"],
        "seed": original["seed"],
        "protocol": protocol,
        "original_receipt": str(original_path),
        "original_receipt_sha256": sha256(original_path),
        "original_preregistration_sha256": original_prereg_hash,
        "comparator_lock_sha256": comparator_lock_hash,
        "rules": list(RULES),
        "results": rows,
        "selection_by_rule_and_utility_threshold": selections,
    }
    atomic_json_dump(payload, output_path)
    return {
        "dataset": payload["dataset"],
        "seed": payload["seed"],
        "output": str(output_path),
        "sha256": sha256(output_path),
        "candidate_rows": len(rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    lock, lock_hash = validate_lock(args.lock)
    required_files = int(lock["required_raw_receipt_count"])
    if len(args.inputs) != required_files:
        raise ValueError(f"expected {required_files} raw receipts")
    if len({path.resolve() for path in args.inputs}) != len(args.inputs):
        raise ValueError("raw receipt inputs must be unique")
    if args.manifest.exists():
        raise FileExistsError(f"refusing to overwrite {args.manifest}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object] | None] = [None] * len(args.inputs)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for index, original in enumerate(args.inputs):
            output = args.output_dir / original.name
            if output.exists():
                raise FileExistsError(f"refusing to overwrite {output}")
            future = executor.submit(
                replay_one,
                original,
                output,
                original_prereg_hash=str(lock["original_preregistration_sha256"]),
                comparator_lock_hash=lock_hash,
            )
            futures[future] = index
        for future in as_completed(futures):
            index = futures[future]
            summaries[index] = future.result()
            print(summaries[index]["output"], flush=True)
    files = [summary for summary in summaries if summary is not None]
    manifest = {
        "name": "MOSAIC bridge comparator extension manifest",
        "comparator_lock_sha256": lock_hash,
        "original_preregistration_sha256": lock[
            "original_preregistration_sha256"
        ],
        "rules": list(RULES),
        "files": files,
        "file_count": len(files),
        "candidate_rows": sum(int(file["candidate_rows"]) for file in files),
    }
    atomic_json_dump(manifest, args.manifest)


if __name__ == "__main__":
    main()
