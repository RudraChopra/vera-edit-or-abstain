#!/usr/bin/env python3
"""Run every locked powered Qwen temporal-confirmation job."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from mosaic_bridge import certify_bridge_membership
from mosaic_real import (
    SPLIT_EXTERNAL,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    build_token_table,
    evaluate_external_channel,
    fit_score_tokenizer,
    load_frozen_store,
)
from mosaic_transform_exact_optimizer import optimize_transform_exact_channel
from run_mosaic_bridge_frontier import stratified_bridge_diagnostic_split
from run_mosaic_qwen_temporal_confirmation import (
    nonconstant,
    sampled_release_tokens,
    training_decoder,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE = (
    ROOT / "research/data/civilcomments_qwen25_powered_confirmation"
)
DEFAULT_PREREG = (
    ROOT / "research/mosaic/prereg_mosaic_qwen_powered_confirmation_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT / "research/artifacts/mosaic_qwen_powered_confirmation_v1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def validate_lock(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.read_text(encoding="utf-8").split()[0] != sha256(path):
        raise ValueError("Qwen powered lock sidecar mismatch")
    prereg = load(path)
    if prereg.get("status") != "locked_before_model_and_outcomes":
        raise RuntimeError("Qwen powered preregistration is not locked")
    for relative, expected in prereg["code_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"locked source mismatch: {relative}")
    for local in (path, sidecar):
        relative = local.relative_to(ROOT)
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative.as_posix()}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if committed != local.read_bytes():
            raise RuntimeError(f"{relative} is not the committed lock")
    return prereg


def expected_balanced_accuracy(
    tokens: np.ndarray,
    labels: np.ndarray,
    channel: np.ndarray,
    decoder: tuple[int, ...],
) -> float:
    decoder_array = np.asarray(decoder)
    correct = []
    for label in (0, 1):
        current = tokens[labels == label]
        reward = (decoder_array == label).astype(np.float64)
        correct.append(float(np.mean(channel[current] @ reward)))
    return float(np.mean(correct))


def diagnostic_counts(
    labels: np.ndarray,
    sources: np.ndarray,
) -> list[list[int]]:
    return [
        [
            int(np.sum((labels == label) & (sources == source)))
            for source in (0, 1)
        ]
        for label in (0, 1)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    prereg = validate_lock(args.prereg)
    manifest_path = args.store / "manifest.json"
    manifest = load(manifest_path)
    if manifest["preregistration_sha256"] != sha256(args.prereg):
        raise RuntimeError("Qwen store was extracted under another lock")
    expected_model = prereg["model"]
    for field, expected in (
        ("model", expected_model["model_id"]),
        ("model_revision", expected_model["revision"]),
        ("representation", "layer28_mean"),
        ("hidden_layer", 28),
        ("pooling", "mean"),
    ):
        if manifest[field] != expected:
            raise RuntimeError(f"store field {field} differs from the lock")

    store = load_frozen_store(args.store)
    construction = np.flatnonzero(store.split == SPLIT_TRAIN)
    reference = np.flatnonzero(store.split == SPLIT_VALIDATION)
    target_pool = np.flatnonzero(store.split == SPLIT_EXTERNAL)
    token_count = int(prereg["fine_token_count"])
    seeds = tuple(int(value) for value in prereg["seeds"])
    thresholds = tuple(float(value) for value in prereg["utility_thresholds"])
    privacy_threshold = float(prereg["privacy_advantage_threshold"])
    primary_threshold = float(prereg["primary_utility_threshold"])
    per_table_delta = float(prereg["familywise_delta"]) / (
        2.0 * len(seeds)
    )
    operational_draws = int(
        prereg["operational_draws_per_primary_release"]
    )
    args.output.mkdir(parents=True)
    receipts: list[dict[str, Any]] = []

    for seed in seeds:
        receipt: dict[str, Any] = {
            "seed": seed,
            "status": "started",
            "preregistration_sha256": sha256(args.prereg),
            "store_manifest_sha256": sha256(manifest_path),
            "candidate": "Qwen2.5-1.5B-Instruct::layer28_mean::K=2",
        }
        try:
            bridge, diagnostic = stratified_bridge_diagnostic_split(
                target_pool,
                store.target,
                store.source,
                seed=seed,
            )
            tokenizer = fit_score_tokenizer(
                store.features[construction],
                store.target[construction],
                token_count=token_count,
                seed=seed,
            )
            construction_tokens = tokenizer.encode(
                store.features[construction]
            )
            reference_tokens = tokenizer.encode(store.features[reference])
            bridge_tokens = tokenizer.encode(store.features[bridge])
            diagnostic_tokens = tokenizer.encode(store.features[diagnostic])
            reference_table = build_token_table(
                reference_tokens,
                store.target[reference],
                store.source[reference],
                token_count=token_count,
                familywise_delta=per_table_delta,
            )
            bridge_table = build_token_table(
                bridge_tokens,
                store.target[bridge],
                store.source[bridge],
                token_count=token_count,
                familywise_delta=per_table_delta,
            )
            bridge_certificate = certify_bridge_membership(
                reference_table.probabilities,
                reference_l1_radii=reference_table.l1_radii,
                bridge_empirical_distributions=bridge_table.probabilities,
                bridge_l1_radii=bridge_table.l1_radii,
            )
            solution = optimize_transform_exact_channel(
                reference_table.probabilities,
                l1_radii=reference_table.l1_radii,
                common_channels_by_label=(
                    bridge_certificate.transforms_by_label
                ),
                contaminations=bridge_certificate.contaminations,
                privacy_advantage_thresholds=(
                    privacy_threshold,
                    privacy_threshold,
                ),
                released_token_count=int(prereg["released_token_count"]),
                solver_time_limit_seconds=float(
                    prereg["solver_time_limit_seconds"]
                ),
                attacker_constraint_generation=bool(
                    prereg["attacker_constraint_generation"]
                ),
            )
            diagnostic_risk = evaluate_external_channel(
                diagnostic_tokens,
                store.target[diagnostic],
                store.source[diagnostic],
                solution.release_channel,
                solution.decoder,
            )
            raw_decoder = training_decoder(
                construction_tokens,
                store.target[construction],
                token_count,
            )
            raw_risk = evaluate_external_channel(
                diagnostic_tokens,
                store.target[diagnostic],
                store.source[diagnostic],
                np.eye(token_count),
                raw_decoder,
            )
            source_bounds = tuple(
                float(value.normalized_advantage)
                for value in solution.privacy_certificates
            )
            certified_error = float(
                solution.certified_worst_conditional_error
            )
            is_nonconstant = nonconstant(solution.release_channel)
            complete_support = bool(
                np.all(reference_table.counts.sum(axis=2) > 0)
                and np.all(bridge_table.counts.sum(axis=2) > 0)
                and diagnostic_risk.estimable
            )
            threshold_decisions = {
                f"{threshold:.2f}": bool(
                    complete_support
                    and is_nonconstant
                    and max(source_bounds)
                    <= privacy_threshold + 1e-10
                    and certified_error <= threshold + 1e-10
                )
                for threshold in thresholds
            }
            primary_release = threshold_decisions[
                f"{primary_threshold:.2f}"
            ]
            heldout_violation = bool(
                primary_release
                and (
                    diagnostic_risk.worst_privacy_advantage is None
                    or diagnostic_risk.worst_conditional_error is None
                    or diagnostic_risk.worst_privacy_advantage
                    > privacy_threshold
                    or diagnostic_risk.worst_conditional_error
                    > primary_threshold
                )
            )
            operational = []
            if primary_release:
                identity = np.eye(solution.released_token_count)
                for draw in range(operational_draws):
                    rng = np.random.default_rng(seed * 100_000 + draw)
                    released_tokens = sampled_release_tokens(
                        diagnostic_tokens,
                        solution.release_channel,
                        rng,
                    )
                    risk = evaluate_external_channel(
                        released_tokens,
                        store.target[diagnostic],
                        store.source[diagnostic],
                        identity,
                        solution.decoder,
                    )
                    violation = bool(
                        risk.worst_privacy_advantage is None
                        or risk.worst_conditional_error is None
                        or risk.worst_privacy_advantage
                        > privacy_threshold
                        or risk.worst_conditional_error
                        > primary_threshold
                    )
                    operational.append(
                        {
                            "draw": draw,
                            "source_advantage": (
                                risk.worst_privacy_advantage
                            ),
                            "worst_conditional_error": (
                                risk.worst_conditional_error
                            ),
                            "violation": violation,
                        }
                    )
            receipt.update(
                {
                    "status": "complete",
                    "tokenizer_thresholds": list(tokenizer.thresholds),
                    "reference_stratum_counts": (
                        reference_table.counts.sum(axis=2).tolist()
                    ),
                    "bridge_stratum_counts": (
                        bridge_table.counts.sum(axis=2).tolist()
                    ),
                    "diagnostic_stratum_counts": diagnostic_counts(
                        store.target[diagnostic],
                        store.source[diagnostic],
                    ),
                    "retained_masses": list(
                        bridge_certificate.retained_masses
                    ),
                    "contaminations": list(
                        bridge_certificate.contaminations
                    ),
                    "release_channel": solution.release_channel.tolist(),
                    "decoder": list(solution.decoder),
                    "nonconstant_release": is_nonconstant,
                    "certified_source_advantage_upper": list(source_bounds),
                    "certified_worst_conditional_error_upper": (
                        certified_error
                    ),
                    "threshold_decisions": threshold_decisions,
                    "primary_release": primary_release,
                    "heldout_source_advantage": (
                        diagnostic_risk.worst_privacy_advantage
                    ),
                    "heldout_worst_conditional_error": (
                        diagnostic_risk.worst_conditional_error
                    ),
                    "heldout_primary_violation": heldout_violation,
                    "raw_fine_token_decoder": list(raw_decoder),
                    "raw_fine_token_source_advantage": (
                        raw_risk.worst_privacy_advantage
                    ),
                    "raw_fine_token_worst_conditional_error": (
                        raw_risk.worst_conditional_error
                    ),
                    "released_expected_balanced_accuracy": (
                        expected_balanced_accuracy(
                            diagnostic_tokens,
                            store.target[diagnostic],
                            solution.release_channel,
                            solution.decoder,
                        )
                    ),
                    "operational_replays": operational,
                    "operational_violation_count": int(
                        sum(
                            bool(value["violation"])
                            for value in operational
                        )
                    ),
                    "solver_status": solution.solver_status,
                    "solver_mip_gap": solution.solver_mip_gap,
                    "solver_max_constraint_violation": (
                        solution.max_constraint_violation
                    ),
                }
            )
        except Exception as error:
            receipt.update(
                {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        receipt_path = args.output / f"seed-{seed}.json"
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        receipts.append(receipt)

    completed = [row for row in receipts if row["status"] == "complete"]
    primary = [row for row in completed if row.get("primary_release")]
    heldout_violations = sum(
        bool(row["heldout_primary_violation"]) for row in primary
    )
    operational_trials = sum(
        len(row["operational_replays"]) for row in primary
    )
    operational_violations = sum(
        int(row["operational_violation_count"]) for row in primary
    )
    minimum_releases = int(
        prereg["main_paper_inclusion_gate"]["minimum_primary_releases"]
    )
    inclusion_gate = bool(
        len(completed) == len(seeds)
        and len(primary) >= minimum_releases
        and heldout_violations == 0
        and operational_violations == 0
    )
    summary = {
        "name": "MOSAIC Qwen2.5 powered temporal confirmation v1",
        "status": (
            "complete"
            if len(completed) == len(seeds)
            else "complete_with_errors"
        ),
        "confirmatory_evidence": True,
        "claim_boundary": prereg["claim_boundary"],
        "preregistration_sha256": sha256(args.prereg),
        "store_manifest_sha256": sha256(manifest_path),
        "registered_jobs": len(seeds),
        "completed_jobs": len(completed),
        "error_jobs": len(seeds) - len(completed),
        "primary_releases": len(primary),
        "primary_abstentions": len(seeds) - len(primary),
        "heldout_primary_violations": heldout_violations,
        "operational_primary_trials": operational_trials,
        "operational_primary_violations": operational_violations,
        "main_paper_inclusion_gate_pass": inclusion_gate,
        "receipt_sha256": {
            f"seed-{seed}.json": sha256(
                args.output / f"seed-{seed}.json"
            )
            for seed in seeds
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if len(completed) != len(seeds):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
