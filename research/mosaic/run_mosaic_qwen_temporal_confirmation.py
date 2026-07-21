#!/usr/bin/env python3
"""Run every locked MOSAIC Qwen temporal-confirmation job."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

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


REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_STORE = Path(
    "/Volumes/Backups/FARO/artifacts/civilcomments_qwen25_temporal_confirmation"
)
DEFAULT_PREREG = (
    REPOSITORY / "research/mosaic/prereg_mosaic_qwen_temporal_confirmation_v1.json"
)
DEFAULT_OUTPUT = REPOSITORY / "research/artifacts/mosaic_qwen_temporal_confirmation_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nonconstant(channel: np.ndarray) -> bool:
    return not np.allclose(channel, channel[0:1], atol=1e-8, rtol=0.0)


def training_decoder(tokens: np.ndarray, target: np.ndarray, token_count: int) -> tuple[int, ...]:
    decoder = []
    for token in range(token_count):
        labels = target[tokens == token]
        decoder.append(int(labels.mean() >= 0.5) if len(labels) else 0)
    return tuple(decoder)


def sampled_release_tokens(
    fine_tokens: np.ndarray, channel: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    cumulative = np.cumsum(channel[np.asarray(fine_tokens, dtype=np.int64)], axis=1)
    cumulative[:, -1] = 1.0
    uniforms = rng.random(len(fine_tokens))
    return np.sum(uniforms[:, None] > cumulative, axis=1).astype(np.int16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    prereg = json.loads(args.prereg.read_text(encoding="utf-8"))
    if prereg.get("status") != "locked_confirmation_authorized":
        raise RuntimeError("confirmation preregistration is not authorized")
    for relative, expected in prereg["code_sha256"].items():
        if sha256(REPOSITORY / relative) != expected:
            raise RuntimeError(f"locked source mismatch: {relative}")
    pilot_path = REPOSITORY / prereg["pilot_artifact"]
    if sha256(pilot_path) != prereg["pilot_artifact_sha256"]:
        raise RuntimeError("pilot artifact differs from the lock")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if not pilot["go_to_locked_confirmation"]:
        raise RuntimeError("pilot did not authorize confirmation")
    if pilot["selected_candidate"]["candidate"] != prereg["selected_candidate"]["key"]:
        raise RuntimeError("locked candidate does not match the pilot selection")

    manifest_path = args.store / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["preregistration_sha256"] != sha256(args.prereg):
        raise RuntimeError("representation store was built under another lock")
    selected = prereg["selected_candidate"]
    for key, expected in (
        ("representation", selected["representation"]),
        ("hidden_layer", selected["hidden_layer"]),
        ("pooling", selected["pooling"]),
        ("model_revision", prereg["model"]["revision"]),
    ):
        if manifest[key] != expected:
            raise RuntimeError(f"store field {key} differs from the lock")

    store = load_frozen_store(args.store)
    construction = np.flatnonzero(store.split == SPLIT_TRAIN)
    reference = np.flatnonzero(store.split == SPLIT_VALIDATION)
    target_pool = np.flatnonzero(store.split == SPLIT_EXTERNAL)
    token_count = int(selected["token_count"])
    seeds = tuple(int(value) for value in prereg["seeds"])
    thresholds = tuple(float(value) for value in prereg["utility_thresholds"])
    privacy_threshold = float(prereg["privacy_advantage_threshold"])
    primary_threshold = float(prereg["primary_utility_threshold"])
    per_table_delta = float(prereg["familywise_delta"]) / (2.0 * len(seeds))
    operational_draws = int(prereg["operational_draws_per_primary_release"])
    args.output.mkdir(parents=True)
    receipts: list[dict[str, object]] = []

    for seed in seeds:
        receipt: dict[str, object] = {
            "seed": seed,
            "status": "started",
            "preregistration_sha256": sha256(args.prereg),
            "store_manifest_sha256": sha256(manifest_path),
            "candidate": selected["key"],
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
            construction_tokens = tokenizer.encode(store.features[construction])
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
                common_channels_by_label=bridge_certificate.transforms_by_label,
                contaminations=bridge_certificate.contaminations,
                privacy_advantage_thresholds=(privacy_threshold, privacy_threshold),
                released_token_count=int(prereg["released_token_count"]),
                solver_time_limit_seconds=float(prereg["solver_time_limit_seconds"]),
                attacker_constraint_generation=bool(prereg["attacker_constraint_generation"]),
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
            certified_error = float(solution.certified_worst_conditional_error)
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
                    and max(source_bounds) <= privacy_threshold + 1e-10
                    and certified_error <= threshold + 1e-10
                )
                for threshold in thresholds
            }
            primary_release = threshold_decisions[f"{primary_threshold:.2f}"]
            heldout_violation = bool(
                primary_release
                and (
                    diagnostic_risk.worst_privacy_advantage is None
                    or diagnostic_risk.worst_conditional_error is None
                    or diagnostic_risk.worst_privacy_advantage > privacy_threshold
                    or diagnostic_risk.worst_conditional_error > primary_threshold
                )
            )
            operational = []
            if primary_release:
                identity = np.eye(solution.released_token_count)
                for draw in range(operational_draws):
                    rng = np.random.default_rng(seed * 100_000 + draw)
                    released_tokens = sampled_release_tokens(
                        diagnostic_tokens, solution.release_channel, rng
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
                        or risk.worst_privacy_advantage > privacy_threshold
                        or risk.worst_conditional_error > primary_threshold
                    )
                    operational.append(
                        {
                            "draw": draw,
                            "source_advantage": risk.worst_privacy_advantage,
                            "worst_conditional_error": risk.worst_conditional_error,
                            "violation": violation,
                        }
                    )
            receipt.update(
                {
                    "status": "complete",
                    "tokenizer_thresholds": list(tokenizer.thresholds),
                    "reference_stratum_counts": reference_table.counts.sum(axis=2).tolist(),
                    "bridge_stratum_counts": bridge_table.counts.sum(axis=2).tolist(),
                    "diagnostic_stratum_counts": [
                        [
                            int(
                                np.sum(
                                    (store.target[diagnostic] == yy)
                                    & (store.source[diagnostic] == ss)
                                )
                            )
                            for ss in (0, 1)
                        ]
                        for yy in (0, 1)
                    ],
                    "retained_masses": list(bridge_certificate.retained_masses),
                    "contaminations": list(bridge_certificate.contaminations),
                    "release_channel": solution.release_channel.tolist(),
                    "decoder": list(solution.decoder),
                    "nonconstant_release": is_nonconstant,
                    "certified_source_advantage_upper": list(source_bounds),
                    "certified_worst_conditional_error_upper": certified_error,
                    "threshold_decisions": threshold_decisions,
                    "primary_release": primary_release,
                    "heldout_source_advantage": diagnostic_risk.worst_privacy_advantage,
                    "heldout_worst_conditional_error": diagnostic_risk.worst_conditional_error,
                    "heldout_primary_violation": heldout_violation,
                    "raw_fine_token_decoder": list(raw_decoder),
                    "raw_fine_token_source_advantage": raw_risk.worst_privacy_advantage,
                    "raw_fine_token_worst_conditional_error": raw_risk.worst_conditional_error,
                    "operational_replays": operational,
                    "operational_violation_count": int(
                        sum(bool(value["violation"]) for value in operational)
                    ),
                    "solver_status": solution.solver_status,
                    "solver_mip_gap": solution.solver_mip_gap,
                    "solver_max_constraint_violation": solution.max_constraint_violation,
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
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        receipts.append(receipt)

    completed = [value for value in receipts if value["status"] == "complete"]
    primary = [value for value in completed if value.get("primary_release")]
    heldout_violations = sum(bool(value["heldout_primary_violation"]) for value in primary)
    operational_trials = sum(len(value["operational_replays"]) for value in primary)
    operational_violations = sum(int(value["operational_violation_count"]) for value in primary)
    inclusion_gate = bool(
        len(completed) == len(seeds)
        and len(primary) >= int(prereg["main_paper_inclusion_gate"]["minimum_primary_releases"])
        and heldout_violations == 0
        and operational_violations == 0
    )
    summary = {
        "name": "MOSAIC Qwen2.5 temporal confirmation v1",
        "status": "complete" if len(completed) == len(seeds) else "complete_with_errors",
        "confirmatory_evidence": True,
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
            f"seed-{seed}.json": sha256(args.output / f"seed-{seed}.json") for seed in seeds
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if len(completed) != len(seeds):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
