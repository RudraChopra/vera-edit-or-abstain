#!/usr/bin/env python3
"""Run paired conservative and transform-exact MOSAIC real-data certificates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from mosaic_optimizer import optimize_invariant_channel
from mosaic_transform_exact_optimizer import optimize_transform_exact_channel
from mosaic_real import (
    BIOS_CLINICAL_PROFESSIONS,
    SPLIT_EXTERNAL,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    balanced_stratum_sample,
    build_token_table,
    evaluate_external_channel,
    fit_score_tokenizer,
    load_frozen_store,
    ordered_smoothing_library,
    sha256,
)
from official_eraser_adapters import EditedCandidate
from run_official_eraser_frontier import (
    dispatch_candidates,
    preprocess,
    random_cap,
    split_eraser_train_construction,
)
from run_mosaic_real_pilot import DATASETS


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
METHODS = ("inlp", "leace", "rlace", "taco", "mance")
FRONTIER_CANDIDATE_COUNT = 13  # identity plus 12 official eraser strengths


def atomic_json_dump(payload: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(output)


def target_values(native: np.ndarray, target_mode: str) -> np.ndarray:
    if target_mode == "native_binary":
        values = np.asarray(native, dtype=np.int64)
    elif target_mode == "bios_clinical_binary":
        values = np.isin(native, tuple(BIOS_CLINICAL_PROFESSIONS)).astype(np.int64)
    else:
        raise ValueError(f"unknown target mode: {target_mode}")
    if set(np.unique(values)) != {0, 1}:
        raise ValueError("frontier pilot requires a binary task")
    return values


def materialize(features: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.asarray(features[indices], dtype=np.float32).copy()


def identity_candidate(
    train: np.ndarray,
    construction: np.ndarray,
    deployment: np.ndarray,
) -> EditedCandidate:
    """Return the unedited representation as an explicit frontier member."""
    return EditedCandidate(
        method="Identity",
        strength="unedited",
        train=train.copy(),
        validation=construction.copy(),
        external=deployment.copy(),
        provenance={
            "official_entrypoint": "none (unedited control)",
            "transformation": "identity",
        },
    )


def select_certified_result(
    results: list[dict[str, object]], variant: str
) -> dict[str, object]:
    """Select the lowest-error candidate certified by one registered variant."""
    eligible = [
        result
        for result in results
        if isinstance(result.get(variant), dict)
        and result[variant].get("deployed") is True
        and isinstance(
            result[variant].get("certified_worst_conditional_error"), (int, float)
        )
    ]
    if not eligible:
        return {
            "decision": "abstain",
            "candidate": None,
            "reason": "no candidate satisfied the registered privacy and utility contract",
        }
    selected = min(
        eligible,
        key=lambda result: (
            float(result[variant]["certified_worst_conditional_error"]),
            str(result["candidate"]),
        ),
    )
    return {
        "decision": "deploy",
        "candidate": selected["candidate"],
        "method": selected["method"],
        "strength": selected["strength"],
        "certificate_variant": variant,
        "certified_worst_conditional_error": selected[variant][
            "certified_worst_conditional_error"
        ],
        "certified_privacy_advantages": selected[variant][
            "certified_privacy_advantages"
        ],
        "external_estimable": selected[variant]["external_estimable"],
        "external_worst_privacy_advantage": selected[variant][
            "external_worst_privacy_advantage"
        ],
        "external_worst_conditional_error": selected[variant][
            "external_worst_conditional_error"
        ],
        "external_safe": selected[variant]["external_safe"],
        "false_acceptance": selected[variant]["false_acceptance"],
    }


def variant_result(
    solution: object,
    *,
    external_tokens: np.ndarray,
    y_external: np.ndarray,
    s_external: np.ndarray,
    privacy_threshold: float,
    utility_threshold: float,
) -> dict[str, object]:
    """Serialize one channel solution and its untouched external diagnostic."""
    external_risk = evaluate_external_channel(
        external_tokens,
        y_external,
        s_external,
        solution.release_channel,
        solution.decoder,
    )
    privacy_advantages = [
        value.normalized_advantage for value in solution.privacy_certificates
    ]
    deploy = bool(
        max(privacy_advantages) <= privacy_threshold + 1e-10
        and solution.certified_worst_conditional_error <= utility_threshold + 1e-10
    )
    external_safe = bool(
        external_risk.estimable
        and external_risk.worst_privacy_advantage is not None
        and external_risk.worst_conditional_error is not None
        and external_risk.worst_privacy_advantage <= privacy_threshold + 1e-10
        and external_risk.worst_conditional_error <= utility_threshold + 1e-10
    )
    return {
        "certificate_method": solution.method,
        "certified_worst_conditional_error": (
            solution.certified_worst_conditional_error
        ),
        "certified_privacy_advantages": privacy_advantages,
        "release_channel": solution.release_channel.tolist(),
        "decoder": solution.decoder,
        "solver_objective": solution.solver_objective,
        "solver_status": solution.solver_status,
        "solver_mip_gap": solution.solver_mip_gap,
        "solver_dual_bound": solution.solver_dual_bound,
        "max_constraint_violation": solution.max_constraint_violation,
        "deployed": deploy,
        "external_estimable": external_risk.estimable,
        "external_worst_privacy_advantage": external_risk.worst_privacy_advantage,
        "external_worst_conditional_error": external_risk.worst_conditional_error,
        "external_safe": external_safe,
        "false_acceptance": bool(
            deploy and external_risk.estimable and not external_safe
        ),
        "missing_external_strata": external_risk.missing_strata,
    }


def candidate_result(
    candidate: EditedCandidate,
    *,
    y_construction: np.ndarray,
    y_certification: np.ndarray,
    y_external: np.ndarray,
    s_certification: np.ndarray,
    s_external: np.ndarray,
    certification_count: int,
    seed: int,
    delta: float,
    contamination: float,
    privacy_threshold: float,
    utility_threshold: float,
    smoothing: float,
) -> dict[str, object]:
    certification = candidate.external[:certification_count]
    external = candidate.external[certification_count:]
    tokenizer = fit_score_tokenizer(
        candidate.validation,
        y_construction,
        token_count=4,
        seed=seed,
    )
    certification_tokens = tokenizer.encode(certification)
    external_tokens = tokenizer.encode(external)
    table = build_token_table(
        certification_tokens,
        y_certification,
        s_certification,
        token_count=4,
        familywise_delta=delta / FRONTIER_CANDIDATE_COUNT,
    )
    transforms = ordered_smoothing_library(4, smoothing=smoothing)
    capacity_transfer = optimize_invariant_channel(
        table.probabilities,
        l1_radii=table.l1_radii,
        common_channels_by_label=(transforms, transforms),
        contaminations=(contamination, contamination),
        privacy_advantage_thresholds=(privacy_threshold, privacy_threshold),
        released_token_count=2,
        solver_time_limit_seconds=300.0,
    )
    transform_exact = optimize_transform_exact_channel(
        table.probabilities,
        l1_radii=table.l1_radii,
        common_channels_by_label=(transforms, transforms),
        contaminations=(contamination, contamination),
        privacy_advantage_thresholds=(privacy_threshold, privacy_threshold),
        released_token_count=2,
        solver_time_limit_seconds=300.0,
    )
    if (
        transform_exact.certified_worst_conditional_error
        > capacity_transfer.certified_worst_conditional_error + 2e-7
    ):
        raise RuntimeError("TRANSFORM_EXACT_DOMINANCE_VIOLATION")
    external_table = build_token_table(
        external_tokens,
        y_external,
        s_external,
        token_count=4,
        familywise_delta=0.5,
    )
    return {
        "candidate": candidate.key,
        "method": candidate.method,
        "strength": candidate.strength,
        "provenance": candidate.provenance,
        "tokenizer_thresholds": tokenizer.thresholds,
        "certification_token_counts": table.counts.tolist(),
        "certification_stratum_counts": table.counts.sum(axis=2).tolist(),
        "l1_radii": table.l1_radii.tolist(),
        "external_token_counts": external_table.counts.tolist(),
        "external_stratum_counts": external_table.counts.sum(axis=2).tolist(),
        "capacity_transfer": variant_result(
            capacity_transfer,
            external_tokens=external_tokens,
            y_external=y_external,
            s_external=s_external,
            privacy_threshold=privacy_threshold,
            utility_threshold=utility_threshold,
        ),
        "transform_exact": variant_result(
            transform_exact,
            external_tokens=external_tokens,
            y_external=y_external,
            s_external=s_external,
            privacy_threshold=privacy_threshold,
            utility_threshold=utility_threshold,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirmation-prereg", type=Path)
    return parser.parse_args()


def validate_confirmation_prereg(
    prereg_path: Path,
    *,
    dataset: str,
    seed: int,
    methods: list[str],
    smoke: bool,
) -> tuple[dict[str, object], str]:
    if smoke or tuple(methods) != METHODS:
        raise ValueError("confirmation requires the complete non-smoke official frontier")
    prereg_sha = sha256(prereg_path)
    sidecar = prereg_path.with_suffix(prereg_path.suffix + ".sha256")
    if not sidecar.exists() or sidecar.read_text(encoding="utf-8").strip() != prereg_sha:
        raise ValueError("confirmation preregistration sidecar does not match")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if prereg.get("status") != "locked_before_confirmatory_outcomes":
        raise ValueError("confirmation preregistration is not locked")
    if dataset not in prereg.get("datasets", {}):
        raise ValueError("dataset is outside the locked confirmation")
    if seed not in prereg.get("confirmation_seeds", []):
        raise ValueError("seed is outside the locked confirmation")
    relative_runner = str(Path(__file__).resolve().relative_to(REPOSITORY))
    expected_runner_sha = prereg.get("code_sha256", {}).get(relative_runner)
    if expected_runner_sha != sha256(Path(__file__)):
        raise ValueError("confirmation runner does not match the preregistered code hash")
    protocol = prereg.get("protocol", {})
    expected = {
        "frontier_candidate_count": FRONTIER_CANDIDATE_COUNT,
        "familywise_delta": 0.05,
        "per_candidate_delta": 0.05 / FRONTIER_CANDIDATE_COUNT,
        "contamination": 0.05,
        "privacy_advantage_threshold": 0.35,
        "maximum_worst_conditional_error": 0.49,
        "ordered_smoothing": 0.10,
        "certificate_variants": ["capacity_transfer", "transform_exact"],
    }
    if protocol != expected:
        raise ValueError("runner constants differ from the locked protocol")
    return prereg, prereg_sha


def main() -> None:
    args = parse_args()
    prereg = None
    prereg_sha = None
    if args.confirmation_prereg is not None:
        prereg, prereg_sha = validate_confirmation_prereg(
            args.confirmation_prereg,
            dataset=args.dataset,
            seed=args.seed,
            methods=args.methods,
            smoke=args.smoke,
        )
        if args.output is None:
            raise ValueError("confirmation requires an explicit output path")
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite confirmation output: {args.output}")
    config = DATASETS[args.dataset]
    path = Path(config["path"])
    store = load_frozen_store(path, target_mode=str(config["target_mode"]))
    y = store.target
    s = store.source
    g = s
    rng = np.random.default_rng(100_003 * args.seed + 2027)
    train_indices, construction_indices = split_eraser_train_construction(
        np.flatnonzero(store.split == SPLIT_TRAIN), y, s, g, rng
    )
    train_indices = random_cap(train_indices, 8000, rng)
    construction_indices = random_cap(construction_indices, 2000, rng)
    certification_indices = balanced_stratum_sample(
        np.flatnonzero(store.split == SPLIT_VALIDATION),
        y,
        s,
        maximum_total=8000,
        seed=args.seed * 100 + 2,
    )
    try:
        external_indices = balanced_stratum_sample(
            np.flatnonzero(store.split == SPLIT_EXTERNAL),
            y,
            s,
            maximum_total=8000,
            seed=args.seed * 100 + 3,
        )
    except ValueError:
        external_indices = np.flatnonzero(store.split == SPLIT_EXTERNAL).astype(np.int64)
        external_indices = random_cap(external_indices, 8000, rng)

    train = materialize(store.features, train_indices)
    construction = materialize(store.features, construction_indices)
    certification = materialize(store.features, certification_indices)
    external = materialize(store.features, external_indices)
    (train, construction, certification, external), preprocessing = preprocess(
        train,
        construction,
        certification,
        external,
        dimension=128,
        seed=args.seed,
    )
    deployment = np.concatenate((certification, external), axis=0)
    y_deployment = np.concatenate((y[certification_indices], y[external_indices]))
    s_deployment = np.concatenate((s[certification_indices], s[external_indices]))
    identity = identity_candidate(train, construction, deployment)
    results = [
        candidate_result(
            identity,
            y_construction=y[construction_indices],
            y_certification=y[certification_indices],
            y_external=y[external_indices],
            s_certification=s[certification_indices],
            s_external=s[external_indices],
            certification_count=len(certification_indices),
            seed=args.seed,
            delta=0.05,
            contamination=0.05,
            privacy_threshold=0.35,
            utility_threshold=0.49,
            smoothing=0.10,
        )
    ]
    for method in args.methods:
        print(f"running {args.dataset} {method}", flush=True)
        candidates = dispatch_candidates(
            method,
            train,
            construction,
            deployment,
            y[train_indices],
            y[construction_indices],
            y_deployment,
            s[train_indices],
            s[construction_indices],
            s_deployment,
            seed=args.seed,
            smoke=args.smoke,
        )
        for candidate in candidates:
            try:
                results.append(
                    candidate_result(
                        candidate,
                        y_construction=y[construction_indices],
                        y_certification=y[certification_indices],
                        y_external=y[external_indices],
                        s_certification=s[certification_indices],
                        s_external=s[external_indices],
                        certification_count=len(certification_indices),
                        seed=args.seed,
                        delta=0.05,
                        contamination=0.05,
                        privacy_threshold=0.35,
                        utility_threshold=0.49,
                        smoothing=0.10,
                    )
                )
            except RuntimeError as error:
                results.append(
                    {
                        "candidate": candidate.key,
                        "method": candidate.method,
                        "strength": candidate.strength,
                        "provenance": candidate.provenance,
                        "capacity_transfer": None,
                        "transform_exact": None,
                        "optimization_error": str(error),
                    }
                )
    if len(results) != FRONTIER_CANDIDATE_COUNT and not args.smoke:
        raise RuntimeError(
            f"expected {FRONTIER_CANDIDATE_COUNT} frontier candidates, got {len(results)}"
        )
    output = args.output or (
        REPOSITORY
        / "research"
        / "artifacts"
        / f"mosaic_official_frontier_exact_{args.dataset.lower().replace('-', '_')}_seed{args.seed}.json"
    )
    payload = {
        "name": (
            "MOSAIC paired exact-certificate real-feature confirmation"
            if prereg is not None
            else "MOSAIC paired exact-certificate exploratory pilot"
        ),
        "status": (
            "confirmatory_locked_protocol"
            if prereg is not None
            else "exploratory_not_confirmatory"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prereg_sha256": prereg_sha,
        "dataset": args.dataset,
        "seed": args.seed,
        "smoke": args.smoke,
        "store_manifest_sha256": sha256(path / "manifest.json"),
        "preprocessing": preprocessing,
        "protocol": {
            "frontier_candidate_count": FRONTIER_CANDIDATE_COUNT,
            "familywise_delta": 0.05,
            "per_candidate_delta": 0.05 / FRONTIER_CANDIDATE_COUNT,
            "contamination": 0.05,
            "privacy_advantage_threshold": 0.35,
            "maximum_worst_conditional_error": 0.49,
            "ordered_smoothing": 0.10,
            "certificate_variants": ["capacity_transfer", "transform_exact"],
        },
        "fit_counts": {
            "eraser_train": len(train_indices),
            "tokenizer_construction": len(construction_indices),
            "certification": len(certification_indices),
            "external": len(external_indices),
        },
        "selection": {
            variant: select_certified_result(results, variant)
            for variant in ("capacity_transfer", "transform_exact")
        },
        "results": results,
    }
    atomic_json_dump(payload, output)
    print(json.dumps({"output": str(output), "candidates": len(results)}, indent=2))


if __name__ == "__main__":
    main()
