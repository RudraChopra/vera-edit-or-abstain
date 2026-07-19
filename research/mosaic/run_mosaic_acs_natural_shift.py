#!/usr/bin/env python3
"""Run MOSAIC across fresh ACS tasks with whole-PUMA geographic holdouts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np

from mosaic_real import (
    SPLIT_EXTERNAL,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    balanced_stratum_sample,
    build_token_table,
    fit_score_tokenizer,
    load_frozen_store,
    sha256,
)
from mosaic_strict_certification import (
    certify_bridge_membership_strict,
    optimize_transform_exact_channel_strict,
)
from replay_mosaic_bridge_strict import serialize_bridge, serialize_release
from run_mosaic_official_frontier_exact_confirmation import (
    FRONTIER_CANDIDATE_COUNT,
    METHODS,
    atomic_json_dump,
    identity_candidate,
    materialize,
)
from run_official_eraser_frontier import (
    dispatch_candidates,
    preprocess,
    random_cap,
    split_eraser_train_construction,
)


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
TASKS = ("income", "employment", "public_coverage")
REFERENCE_STATE = "CA"
TARGET_STATES = ("WA", "IL", "NY", "FL")
CONFIRMATION_SEEDS = (1400, 1401, 1402, 1403, 1404)
FINE_TOKEN_COUNTS = (4, 8)
RELEASED_TOKEN_COUNT = 2
FAMILYWISE_DELTA = 0.05
SOURCE_ADVANTAGE_THRESHOLD = 0.35
UTILITY_THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.49)
PRIMARY_UTILITY_THRESHOLD = 0.40
BRIDGE_ENVIRONMENT_FRACTION = 2.0 / 3.0
MAXIMUM_REFERENCE_ROWS = 24_000
MAXIMUM_BRIDGE_ROWS = 24_000
MAXIMUM_DIAGNOSTIC_ROWS = 24_000
OPERATIONAL_DRAWS = 100


def threshold_key(value: float) -> str:
    return f"{float(value):.2f}"


def expected_protocol() -> dict[str, object]:
    return {
        "reference_state": REFERENCE_STATE,
        "target_states": list(TARGET_STATES),
        "tasks": list(TASKS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "fine_token_counts": list(FINE_TOKEN_COUNTS),
        "released_token_count": RELEASED_TOKEN_COUNT,
        "frontier_candidate_count": FRONTIER_CANDIDATE_COUNT,
        "familywise_delta": FAMILYWISE_DELTA,
        "per_candidate_table_delta": FAMILYWISE_DELTA
        / (len(FINE_TOKEN_COUNTS) * 2.0 * FRONTIER_CANDIDATE_COUNT),
        "source_advantage_threshold": SOURCE_ADVANTAGE_THRESHOLD,
        "utility_thresholds": list(UTILITY_THRESHOLDS),
        "primary_utility_threshold": PRIMARY_UTILITY_THRESHOLD,
        "bridge_environment_fraction": BRIDGE_ENVIRONMENT_FRACTION,
        "maximum_reference_rows": MAXIMUM_REFERENCE_ROWS,
        "maximum_bridge_rows": MAXIMUM_BRIDGE_ROWS,
        "maximum_diagnostic_rows": MAXIMUM_DIAGNOSTIC_ROWS,
        "operational_draws": OPERATIONAL_DRAWS,
        "selection_rule": "minimum certified error, then lexicographic candidate key",
    }


def puma_bridge_diagnostic_split(
    indices: Sequence[int], environments: np.ndarray, *, seed: int
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], tuple[int, ...]]:
    """Hold out entire target-state PUMAs without consulting labels or sources."""

    selected = np.asarray(indices, dtype=np.int64)
    represented = np.unique(environments[selected]).astype(np.int64)
    if len(represented) < 3:
        raise ValueError("natural-shift evaluation requires at least three PUMAs")
    order = np.random.default_rng(seed).permutation(represented)
    bridge_count = int(np.floor(BRIDGE_ENVIRONMENT_FRACTION * len(order)))
    bridge_count = min(max(1, bridge_count), len(order) - 1)
    bridge_environments = tuple(sorted(int(value) for value in order[:bridge_count]))
    diagnostic_environments = tuple(sorted(int(value) for value in order[bridge_count:]))
    bridge = selected[np.isin(environments[selected], bridge_environments)]
    diagnostic = selected[np.isin(environments[selected], diagnostic_environments)]
    if len(bridge) == 0 or len(diagnostic) == 0:
        raise RuntimeError("PUMA partition produced an empty side")
    return bridge, diagnostic, bridge_environments, diagnostic_environments


def _all_strata(values: np.ndarray, target: np.ndarray, source: np.ndarray) -> bool:
    return {
        (int(target[index]), int(source[index])) for index in np.asarray(values)
    } == {(0, 0), (0, 1), (1, 0), (1, 1)}


def _table_payload(table: object) -> dict[str, object]:
    return {
        "token_counts": table.counts.tolist(),
        "stratum_counts": table.counts.sum(axis=2).tolist(),
        "l1_radii": table.l1_radii.tolist(),
    }


def _select(
    rows: list[dict[str, object]], *, rule: str, utility_threshold: float
) -> dict[str, object]:
    key = threshold_key(utility_threshold)
    release_key = f"{rule}_release"
    eligible = [
        row
        for row in rows
        if isinstance(row.get(release_key), dict)
        and bool(row[release_key]["threshold_decisions"][key]["deployed"])
    ]
    if not eligible:
        return {
            "decision": "abstain",
            "candidate": None,
            "rule": rule,
            "utility_threshold": utility_threshold,
            "reason": "no candidate satisfied the registered strict contract",
        }
    selected = min(
        eligible,
        key=lambda row: (
            float(row[release_key]["certified_worst_conditional_error_upper"]),
            str(row["candidate"]),
        ),
    )
    release = selected[release_key]
    decision = release["threshold_decisions"][key]
    return {
        "decision": "deploy",
        "candidate": selected["candidate"],
        "method": selected["method"],
        "strength": selected["strength"],
        "rule": rule,
        "utility_threshold": utility_threshold,
        "certified_worst_conditional_error_upper": release[
            "certified_worst_conditional_error_upper"
        ],
        "certified_source_advantage_upper": release[
            "certified_source_advantage_upper"
        ],
        "diagnostic_estimable": release["diagnostic"]["estimable"],
        "diagnostic_worst_source_advantage": release["diagnostic"][
            "worst_privacy_advantage"
        ],
        "diagnostic_worst_conditional_error": release["diagnostic"][
            "worst_conditional_error"
        ],
        "diagnostic_safe": decision["diagnostic_safe"],
        "false_acceptance": decision["false_acceptance"],
    }


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "q025": float(np.quantile(array, 0.025)),
        "median": float(np.median(array)),
        "q975": float(np.quantile(array, 0.975)),
        "maximum": float(np.max(array)),
    }


def operational_replay(
    diagnostic_counts: Sequence[Sequence[Sequence[int]]],
    release_channel: Sequence[Sequence[float]],
    decoder: Sequence[int],
    *,
    seed: int,
    draws: int = OPERATIONAL_DRAWS,
) -> dict[str, object]:
    """Sample one public token per held-out person and run the fixed decoder."""

    counts = np.asarray(diagnostic_counts, dtype=np.int64)
    channel = np.asarray(release_channel, dtype=np.float64)
    decoder_array = np.asarray(decoder, dtype=np.int64)
    if counts.ndim != 3 or channel.shape[0] != counts.shape[2]:
        raise ValueError("diagnostic counts and channel disagree")
    rng = np.random.default_rng(seed)
    privacy_values: list[float] = []
    utility_values: list[float] = []
    for _ in range(draws):
        released = np.zeros((counts.shape[0], counts.shape[1], channel.shape[1]), dtype=np.int64)
        for label in range(counts.shape[0]):
            for source in range(counts.shape[1]):
                for token in range(counts.shape[2]):
                    released[label, source] += rng.multinomial(
                        int(counts[label, source, token]), channel[token]
                    )
        label_privacy = []
        stratum_errors = []
        for label in range(released.shape[0]):
            normalized = released[label] / released[label].sum(axis=1, keepdims=True)
            attacker_ba = float(np.max(normalized, axis=0).sum() / released.shape[1])
            label_privacy.append(2.0 * attacker_ba - 1.0)
            for source in range(released.shape[1]):
                wrong = decoder_array != label
                stratum_errors.append(
                    float(released[label, source, wrong].sum() / released[label, source].sum())
                )
        privacy_values.append(max(label_privacy))
        utility_values.append(max(stratum_errors))
    return {
        "draws": draws,
        "one_public_token_per_person": True,
        "source_advantage": _quantiles(privacy_values),
        "worst_conditional_error": _quantiles(utility_values),
        "primary_contract_violations": sum(
            privacy > SOURCE_ADVANTAGE_THRESHOLD or error > PRIMARY_UTILITY_THRESHOLD
            for privacy, error in zip(privacy_values, utility_values, strict=True)
        ),
    }


def validate_preregistration(
    path: Path, *, task: str, target_state: str, seed: int, store: Path
) -> tuple[dict[str, object], str]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    digest = sha256(path)
    if not sidecar.exists() or sidecar.read_text(encoding="utf-8").strip() != digest:
        raise ValueError("preregistration sidecar mismatch")
    prereg = json.loads(path.read_text(encoding="utf-8"))
    if prereg.get("status") != "locked_before_fresh_state_downloads":
        raise ValueError("natural-shift preregistration is not locked")
    if prereg.get("protocol") != expected_protocol():
        raise ValueError("runner protocol differs from the locked protocol")
    if task not in prereg["protocol"]["tasks"]:
        raise ValueError("task is outside the locked study")
    if target_state not in prereg["protocol"]["target_states"]:
        raise ValueError("state is outside the locked study")
    if seed not in prereg["protocol"]["confirmation_seeds"]:
        raise ValueError("seed is outside the locked study")
    for relative, expected in prereg.get("code_sha256", {}).items():
        if sha256(REPOSITORY / relative) != expected:
            raise ValueError(f"locked code mismatch: {relative}")
    for local in (path, sidecar):
        relative = local.resolve().relative_to(REPOSITORY.resolve())
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative.as_posix()}"],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
        ).stdout
        if committed != local.read_bytes():
            raise ValueError(f"{relative} is not the committed lock")
    manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("task") != task or manifest.get("states") != {
        "reference": REFERENCE_STATE,
        "external": target_state,
    }:
        raise ValueError("frozen store does not match the registered job")
    return prereg, digest


def run_job(
    *,
    task: str,
    target_state: str,
    seed: int,
    store_path: Path,
) -> dict[str, object]:
    store = load_frozen_store(store_path, target_mode="native_binary")
    environment = np.load(store_path / "environment.npy", mmap_mode="r")
    y = store.target
    s = store.source
    rng = np.random.default_rng(100_003 * seed + 2027)
    train_indices, construction_indices = split_eraser_train_construction(
        np.flatnonzero(store.split == SPLIT_TRAIN), y, s, s, rng
    )
    train_indices = random_cap(train_indices, 8000, rng)
    construction_indices = random_cap(construction_indices, 4000, rng)
    reference_indices = balanced_stratum_sample(
        np.flatnonzero(store.split == SPLIT_VALIDATION),
        y,
        s,
        maximum_total=MAXIMUM_REFERENCE_ROWS,
        seed=seed * 100 + 2,
    )
    raw_bridge, raw_diagnostic, bridge_pumas, diagnostic_pumas = puma_bridge_diagnostic_split(
        np.flatnonzero(store.split == SPLIT_EXTERNAL),
        environment,
        seed=seed * 100 + 3,
    )
    bridge_indices = balanced_stratum_sample(
        raw_bridge,
        y,
        s,
        maximum_total=MAXIMUM_BRIDGE_ROWS,
        seed=seed * 100 + 4,
    )
    diagnostic_indices = balanced_stratum_sample(
        raw_diagnostic,
        y,
        s,
        maximum_total=MAXIMUM_DIAGNOSTIC_ROWS,
        seed=seed * 100 + 5,
    )
    if not all(
        _all_strata(values, y, s)
        for values in (reference_indices, bridge_indices, diagnostic_indices)
    ):
        raise ValueError("registered source-label support is incomplete")

    train = materialize(store.features, train_indices)
    construction = materialize(store.features, construction_indices)
    reference = materialize(store.features, reference_indices)
    external_order = np.concatenate((bridge_indices, diagnostic_indices))
    external = materialize(store.features, external_order)
    (train, construction, reference, external), preprocessing = preprocess(
        train,
        construction,
        reference,
        external,
        dimension=128,
        seed=seed,
    )
    deployment = np.concatenate((reference, external), axis=0)
    y_deployment = np.concatenate((y[reference_indices], y[external_order]))
    s_deployment = np.concatenate((s[reference_indices], s[external_order]))
    candidates = [identity_candidate(train, construction, deployment)]
    for method in METHODS:
        print(f"running {task} CA->{target_state} seed={seed} {method}", flush=True)
        candidates.extend(
            dispatch_candidates(
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
                seed=seed,
                smoke=False,
            )
        )
    if len(candidates) != FRONTIER_CANDIDATE_COUNT:
        raise RuntimeError("official frontier candidate count changed")

    table_delta = float(expected_protocol()["per_candidate_table_delta"])
    alphabets: dict[str, object] = {}
    for token_count in FINE_TOKEN_COUNTS:
        rows: list[dict[str, object]] = []
        ephemeral_counts: dict[str, np.ndarray] = {}
        for candidate in candidates:
            row: dict[str, object] = {
                "candidate": candidate.key,
                "method": candidate.method,
                "strength": candidate.strength,
                "provenance": candidate.provenance,
            }
            try:
                tokenizer = fit_score_tokenizer(
                    candidate.validation,
                    y[construction_indices],
                    token_count=token_count,
                    seed=seed,
                )
                candidate_reference = candidate.external[: len(reference_indices)]
                bridge_end = len(reference_indices) + len(bridge_indices)
                candidate_bridge = candidate.external[len(reference_indices) : bridge_end]
                candidate_diagnostic = candidate.external[bridge_end:]
                reference_tokens = tokenizer.encode(candidate_reference)
                bridge_tokens = tokenizer.encode(candidate_bridge)
                diagnostic_tokens = tokenizer.encode(candidate_diagnostic)
                reference_table = build_token_table(
                    reference_tokens,
                    y[reference_indices],
                    s[reference_indices],
                    token_count=token_count,
                    familywise_delta=table_delta,
                )
                bridge_table = build_token_table(
                    bridge_tokens,
                    y[bridge_indices],
                    s[bridge_indices],
                    token_count=token_count,
                    familywise_delta=table_delta,
                )
                diagnostic_table = build_token_table(
                    diagnostic_tokens,
                    y[diagnostic_indices],
                    s[diagnostic_indices],
                    token_count=token_count,
                    familywise_delta=0.5,
                )
                membership = certify_bridge_membership_strict(
                    reference_table.probabilities,
                    reference_l1_radii=reference_table.l1_radii,
                    bridge_empirical_distributions=bridge_table.probabilities,
                    bridge_l1_radii=bridge_table.l1_radii,
                )
                mosaic = optimize_transform_exact_channel_strict(
                    reference_table.probabilities,
                    l1_radii=reference_table.l1_radii,
                    common_channels_by_label=membership.transforms_by_label,
                    contaminations=membership.contaminations,
                    source_advantage_thresholds=(
                        SOURCE_ADVANTAGE_THRESHOLD,
                        SOURCE_ADVANTAGE_THRESHOLD,
                    ),
                    released_token_count=RELEASED_TOKEN_COUNT,
                    solver_time_limit_seconds=300.0,
                )
                identity = np.eye(token_count, dtype=np.float64)
                direct = optimize_transform_exact_channel_strict(
                    bridge_table.probabilities,
                    l1_radii=bridge_table.l1_radii,
                    common_channels_by_label=((identity,), (identity,)),
                    contaminations=(0.0, 0.0),
                    source_advantage_thresholds=(
                        SOURCE_ADVANTAGE_THRESHOLD,
                        SOURCE_ADVANTAGE_THRESHOLD,
                    ),
                    released_token_count=RELEASED_TOKEN_COUNT,
                    solver_time_limit_seconds=300.0,
                )
                row.update(
                    {
                        "tokenizer_thresholds": list(tokenizer.thresholds),
                        "reference_table": _table_payload(reference_table),
                        "bridge_table": _table_payload(bridge_table),
                        "diagnostic_table": _table_payload(diagnostic_table),
                        "bridge_membership": serialize_bridge(membership),
                        "mosaic_release": serialize_release(
                            mosaic,
                            diagnostic_counts=diagnostic_table.counts,
                            source_threshold=SOURCE_ADVANTAGE_THRESHOLD,
                            utility_thresholds=list(UTILITY_THRESHOLDS),
                        ),
                        "direct_release": serialize_release(
                            direct,
                            diagnostic_counts=diagnostic_table.counts,
                            source_threshold=SOURCE_ADVANTAGE_THRESHOLD,
                            utility_thresholds=list(UTILITY_THRESHOLDS),
                        ),
                    }
                )
                ephemeral_counts[candidate.key] = diagnostic_table.counts
            except (RuntimeError, ValueError) as error:
                row["optimization_error"] = str(error)
            rows.append(row)
        selections = {
            rule: {
                threshold_key(threshold): _select(
                    rows, rule=rule, utility_threshold=threshold
                )
                for threshold in UTILITY_THRESHOLDS
            }
            for rule in ("mosaic", "direct")
        }
        primary_key = threshold_key(PRIMARY_UTILITY_THRESHOLD)
        for rule in ("mosaic", "direct"):
            selection = selections[rule][primary_key]
            if selection["decision"] == "deploy":
                selected_row = next(
                    row for row in rows if row["candidate"] == selection["candidate"]
                )
                release = selected_row[f"{rule}_release"]
                selection["operational_replay"] = operational_replay(
                    ephemeral_counts[str(selection["candidate"])],
                    release["release_channel"],
                    release["decoder"],
                    seed=seed * 10_000 + token_count * 100 + (0 if rule == "mosaic" else 1),
                )
        alphabets[str(token_count)] = {
            "rows": rows,
            "selection_by_rule_and_threshold": selections,
            "primary_selection": {
                rule: selections[rule][primary_key] for rule in ("mosaic", "direct")
            },
        }

    return {
        "project": "MOSAIC natural multi-environment ACS confirmation",
        "task": task,
        "reference_state": REFERENCE_STATE,
        "target_state": target_state,
        "seed": seed,
        "protocol": expected_protocol(),
        "store_manifest_sha256": sha256(store_path / "manifest.json"),
        "preprocessing": preprocessing,
        "sample_counts": {
            "train": len(train_indices),
            "construction": len(construction_indices),
            "reference": len(reference_indices),
            "bridge": len(bridge_indices),
            "diagnostic": len(diagnostic_indices),
        },
        "puma_partition": {
            "bridge": list(bridge_pumas),
            "diagnostic": list(diagnostic_pumas),
            "disjoint": set(bridge_pumas).isdisjoint(diagnostic_pumas),
        },
        "alphabets": alphabets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--target-state", choices=TARGET_STATES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prereg", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {args.output}")
    prereg_sha = None
    if args.prereg is not None:
        _, prereg_sha = validate_preregistration(
            args.prereg,
            task=args.task,
            target_state=args.target_state,
            seed=args.seed,
            store=args.store,
        )
    payload = run_job(
        task=args.task,
        target_state=args.target_state,
        seed=args.seed,
        store_path=args.store,
    )
    payload["preregistration_sha256"] = prereg_sha
    payload["claim_boundary"] = (
        "Whole target-state PUMAs are held out before tokenization and optimization. "
        "The diagnostic measures natural geographic transfer for the registered "
        "states and tasks; it is not a claim about all states or later time periods."
    )
    atomic_json_dump(payload, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
