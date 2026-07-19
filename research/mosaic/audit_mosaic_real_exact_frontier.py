#!/usr/bin/env python3
"""Independently replay paired MOSAIC real-frontier receipts and optima."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from mosaic_channel import normalized_attacker_advantage
from mosaic_envelope import weissman_l1_radius
from mosaic_invariant import (
    adaptive_pre_release_attacker_certificate,
    pre_release_utility_certificate,
)
from mosaic_optimizer import optimize_invariant_channel
from mosaic_real import ordered_smoothing_library
from mosaic_transform_exact import (
    transform_exact_attacker_confidence_bound,
    transform_exact_utility_confidence_bound,
)
from mosaic_transform_exact_optimizer import optimize_transform_exact_channel
from run_mosaic_official_frontier_exact_confirmation import select_certified_result


TOLERANCE = 2e-7
VARIANTS = ("capacity_transfer", "transform_exact")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json_dump(payload: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(output)


def probabilities_from_counts(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if counts.shape != (2, 2, 4) or np.any(counts < 0):
        raise ValueError("token counts must have shape (2, 2, 4) and be nonnegative")
    totals = counts.sum(axis=2)
    probabilities = np.full(counts.shape, 0.25, dtype=np.float64)
    for label in range(2):
        for source in range(2):
            if totals[label, source] > 0:
                probabilities[label, source] = counts[label, source] / totals[label, source]
    return probabilities, totals


def replay_external(
    counts: np.ndarray,
    channel: np.ndarray,
    decoder: tuple[int, ...],
) -> dict[str, object]:
    probabilities, totals = probabilities_from_counts(counts)
    missing = [
        [label, source]
        for label in range(2)
        for source in range(2)
        if totals[label, source] == 0
    ]
    if missing:
        return {
            "estimable": False,
            "worst_privacy_advantage": None,
            "worst_conditional_error": None,
            "missing_strata": missing,
        }
    privacy = []
    errors = []
    decoder_array = np.asarray(decoder, dtype=np.int64)
    for label in range(2):
        released = probabilities[label] @ channel
        balanced_accuracy = float(np.max(released, axis=0).sum() / 2.0)
        privacy.append(normalized_attacker_advantage(balanced_accuracy, 2))
        loss = (decoder_array != label).astype(np.float64)
        errors.extend(
            float(probabilities[label, source] @ channel @ loss)
            for source in range(2)
        )
    return {
        "estimable": True,
        "worst_privacy_advantage": max(privacy),
        "worst_conditional_error": max(errors),
        "missing_strata": [],
    }


def close(first: object, second: object) -> bool:
    if first is None or second is None:
        return first is second
    return abs(float(first) - float(second)) <= TOLERANCE


def replay_variant(
    stored: dict[str, object],
    *,
    variant: str,
    probabilities: np.ndarray,
    radii: np.ndarray,
    external_counts: np.ndarray,
    protocol: dict[str, object],
) -> tuple[list[str], dict[str, object]]:
    failures: list[str] = []
    channel = np.asarray(stored["release_channel"], dtype=np.float64)
    decoder = tuple(int(value) for value in stored["decoder"])
    if (
        channel.shape != (4, 2)
        or np.any(channel < -TOLERANCE)
        or np.any(channel > 1.0 + TOLERANCE)
        or not np.allclose(channel.sum(axis=1), 1.0, atol=TOLERANCE, rtol=0.0)
    ):
        failures.append("invalid release channel")
        return failures, {}

    transforms = ordered_smoothing_library(
        4, smoothing=float(protocol["ordered_smoothing"])
    )
    eta = float(protocol["contamination"])
    privacy_threshold = float(protocol["privacy_advantage_threshold"])
    utility_threshold = float(protocol["maximum_worst_conditional_error"])
    if variant == "capacity_transfer":
        privacy = [
            adaptive_pre_release_attacker_certificate(
                probabilities[label],
                channel,
                l1_radii=radii[label],
                common_fine_token_channels=transforms,
                contamination=eta,
            ).normalized_advantage
            for label in range(2)
        ]
        utility = max(
            pre_release_utility_certificate(
                probabilities[label, source],
                channel,
                decoder,
                true_label=label,
                l1_radius=float(radii[label, source]),
                common_fine_token_channels=transforms,
                contamination=eta,
            ).error_probability
            for label in range(2)
            for source in range(2)
        )
        optimized = optimize_invariant_channel(
            probabilities,
            l1_radii=radii,
            common_channels_by_label=(transforms, transforms),
            contaminations=(eta, eta),
            privacy_advantage_thresholds=(privacy_threshold, privacy_threshold),
            released_token_count=2,
            solver_time_limit_seconds=300.0,
        )
    elif variant == "transform_exact":
        privacy = [
            transform_exact_attacker_confidence_bound(
                probabilities[label],
                channel,
                l1_radii=radii[label],
                common_fine_token_channels=transforms,
                contamination=eta,
            ).normalized_advantage
            for label in range(2)
        ]
        utility = max(
            transform_exact_utility_confidence_bound(
                probabilities[label, source],
                channel,
                decoder,
                true_label=label,
                l1_radius=float(radii[label, source]),
                common_fine_token_channels=transforms,
                contamination=eta,
            ).error_probability
            for label in range(2)
            for source in range(2)
        )
        optimized = optimize_transform_exact_channel(
            probabilities,
            l1_radii=radii,
            common_channels_by_label=(transforms, transforms),
            contaminations=(eta, eta),
            privacy_advantage_thresholds=(privacy_threshold, privacy_threshold),
            released_token_count=2,
            solver_time_limit_seconds=300.0,
        )
    else:
        raise ValueError(f"unknown certificate variant: {variant}")

    if not np.allclose(
        privacy,
        np.asarray(stored["certified_privacy_advantages"], dtype=np.float64),
        atol=TOLERANCE,
        rtol=0.0,
    ):
        failures.append("privacy certificate mismatch")
    if not close(utility, stored["certified_worst_conditional_error"]):
        failures.append("utility certificate mismatch")
    if not close(optimized.certified_worst_conditional_error, utility):
        failures.append("global optimum mismatch")
    if not close(stored["solver_objective"], utility):
        failures.append("stored solver objective mismatch")
    if float(stored["solver_mip_gap"]) > 1e-10:
        failures.append("non-global solver gap")
    if float(stored["max_constraint_violation"]) > TOLERANCE:
        failures.append("solver constraint violation")

    deployed = bool(
        max(privacy) <= privacy_threshold + 1e-10
        and utility <= utility_threshold + 1e-10
    )
    if deployed is not bool(stored["deployed"]):
        failures.append("deployment decision mismatch")

    external = replay_external(external_counts, channel, decoder)
    if external["estimable"] is not bool(stored["external_estimable"]):
        failures.append("external estimability mismatch")
    if not close(
        external["worst_privacy_advantage"],
        stored["external_worst_privacy_advantage"],
    ):
        failures.append("external privacy mismatch")
    if not close(
        external["worst_conditional_error"],
        stored["external_worst_conditional_error"],
    ):
        failures.append("external utility mismatch")
    if external["missing_strata"] != stored["missing_external_strata"]:
        failures.append("external missing-strata mismatch")
    external_safe = bool(
        external["estimable"]
        and float(external["worst_privacy_advantage"]) <= privacy_threshold + 1e-10
        and float(external["worst_conditional_error"]) <= utility_threshold + 1e-10
    )
    if external_safe is not bool(stored["external_safe"]):
        failures.append("external safety mismatch")
    false_acceptance = bool(deployed and external["estimable"] and not external_safe)
    if false_acceptance is not bool(stored["false_acceptance"]):
        failures.append("false-acceptance mismatch")
    return failures, {
        "certified_error": utility,
        "deployed": deployed,
        "external_estimable": external["estimable"],
        "external_safe": external_safe,
        "false_acceptance": false_acceptance,
    }


def replay_candidate(
    result: dict[str, object], protocol: dict[str, object]
) -> tuple[list[str], dict[str, object]]:
    failures: list[str] = []
    certification_counts = np.asarray(
        result["certification_token_counts"], dtype=np.int64
    )
    probabilities, totals = probabilities_from_counts(certification_counts)
    delta = float(protocol["per_candidate_delta"])
    radii = np.asarray(
        [
            [
                2.0
                if totals[label, source] == 0
                else weissman_l1_radius(int(totals[label, source]), 4, delta / 4.0)
                for source in range(2)
            ]
            for label in range(2)
        ],
        dtype=np.float64,
    )
    if not np.allclose(
        radii,
        np.asarray(result["l1_radii"], dtype=np.float64),
        atol=TOLERANCE,
        rtol=0.0,
    ):
        failures.append("L1 radius mismatch")

    replay: dict[str, object] = {"candidate": result["candidate"]}
    external_counts = np.asarray(result["external_token_counts"], dtype=np.int64)
    for variant in VARIANTS:
        stored = result.get(variant)
        if not isinstance(stored, dict):
            failures.append(f"missing {variant} result")
            continue
        variant_failures, variant_replay = replay_variant(
            stored,
            variant=variant,
            probabilities=probabilities,
            radii=radii,
            external_counts=external_counts,
            protocol=protocol,
        )
        failures.extend(f"{variant}: {value}" for value in variant_failures)
        replay[variant] = variant_replay
    if all(variant in replay for variant in VARIANTS):
        fallback = float(replay["capacity_transfer"]["certified_error"])
        exact = float(replay["transform_exact"]["certified_error"])
        if exact > fallback + TOLERANCE:
            failures.append("pointwise transform-exact dominance violation")
        replay["objective_improvement"] = fallback - exact
    return failures, replay


def audit_file(path: Path) -> tuple[list[str], dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    results = payload.get("results", [])
    registered_count = int(payload["protocol"]["frontier_candidate_count"])
    if payload.get("smoke") is True:
        if not 0 < len(results) <= registered_count:
            failures.append("invalid reduced smoke frontier size")
    elif len(results) != registered_count:
        failures.append("frontier candidate count mismatch")
    replayed = []
    for result in results:
        if "optimization_error" in result:
            failures.append(f"{result['candidate']}: optimization failed")
            continue
        candidate_failures, replay = replay_candidate(result, payload["protocol"])
        failures.extend(f"{result['candidate']}: {value}" for value in candidate_failures)
        replayed.append(replay)
    expected_selection = {
        variant: select_certified_result(results, variant) for variant in VARIANTS
    }
    if payload.get("selection") != expected_selection:
        failures.append("frontier selection mismatch")
    return failures, {
        "path": str(path),
        "sha256": sha256(path),
        "dataset": payload.get("dataset"),
        "seed": payload.get("seed"),
        "prereg_sha256": payload.get("prereg_sha256"),
        "candidate_rows_replayed": len(replayed),
        "optimization_replays": 2 * len(replayed),
        "strict_objective_improvements": sum(
            float(value.get("objective_improvement", 0.0)) > 1e-10
            for value in replayed
        ),
        "pointwise_dominance": all(
            float(value.get("objective_improvement", 0.0)) >= -TOLERANCE
            for value in replayed
        ),
        "selection": expected_selection,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    failures: list[str] = []
    files: list[dict[str, object] | None] = [None] * len(args.inputs)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(audit_file, path): (index, path)
            for index, path in enumerate(args.inputs)
        }
        for future in as_completed(futures):
            index, path = futures[future]
            file_failures, summary = future.result()
            failures.extend(f"{path.name}: {value}" for value in file_failures)
            files[index] = summary
    completed_files = [value for value in files if value is not None]
    prereg_hashes = {value["prereg_sha256"] for value in completed_files}
    if len(prereg_hashes) != 1:
        failures.append("receipts do not share one preregistration hash")
    report: dict[str, object] = {
        "name": "MOSAIC paired exact real-frontier independent replay",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": not failures,
        "tolerance": TOLERANCE,
        "files_replayed": len(completed_files),
        "candidate_rows_replayed": sum(
            int(value["candidate_rows_replayed"]) for value in completed_files
        ),
        "optimization_replays": sum(
            int(value["optimization_replays"]) for value in completed_files
        ),
        "strict_objective_improvements": sum(
            int(value["strict_objective_improvements"])
            for value in completed_files
        ),
        "pointwise_dominance": all(
            bool(value["pointwise_dominance"]) for value in completed_files
        ),
        "failures": failures,
        "files": completed_files,
    }
    if args.output:
        atomic_json_dump(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
