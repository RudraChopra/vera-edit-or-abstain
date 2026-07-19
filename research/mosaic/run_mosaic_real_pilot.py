#!/usr/bin/env python3
"""Run exploratory MOSAIC pilots on five frozen real-feature stores."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

from mosaic_optimizer import optimize_invariant_channel, optimize_population_external_channel
from mosaic_real import (
    SPLIT_EXTERNAL,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    balanced_stratum_sample,
    build_token_table,
    evaluate_external_channel,
    fit_score_tokenizer,
    load_frozen_store,
    minimum_contamination_fraction,
    ordered_smoothing_library,
    random_cap_sample,
    sha256,
)


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
DEFAULT_OUTPUT = REPOSITORY / "research" / "artifacts" / "mosaic_real_pilot_v1.json"
DATASETS = {
    "ACSIncome-CA-TX": {
        "path": Path("/Volumes/Backups/FARO/artifacts/acs_income_ca_tx_numpy_store"),
        "target_mode": "native_binary",
        "modality": "tabular geographic shift",
    },
    "Waterbirds": {
        "path": Path("/Volumes/Backups/FARO/artifacts/waterbirds_official_numpy_store"),
        "target_mode": "native_binary",
        "modality": "vision",
    },
    "Camelyon17-WILDS": {
        "path": Path("/Volumes/Backups/FARO/artifacts/camelyon17_resnet18_torch_center_numpy_store"),
        "target_mode": "native_binary",
        "modality": "medical vision",
    },
    "CivilComments-WILDS": {
        "path": Path("/Volumes/Backups/FARO/artifacts/civilcomments_lexical_numpy_store"),
        "target_mode": "native_binary",
        "modality": "text",
    },
    "BiasBios-Clinical": {
        "path": Path("/Volumes/Backups/FARO/artifacts/bios_rlace_numpy_store"),
        "target_mode": "bios_clinical_binary",
        "modality": "text",
    },
    "GaitPDB": {
        "path": Path("/Volumes/Backups/FARO/artifacts/gaitpdb_numpy_store"),
        "target_mode": "native_binary",
        "modality": "time series",
    },
}


def atomic_json_dump(payload: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(output)


def selected_indices(store: Any, split: int, cap: int, seed: int) -> np.ndarray:
    return balanced_stratum_sample(
        np.flatnonzero(store.split == split),
        store.target,
        store.source,
        maximum_total=cap,
        seed=seed,
    )


def risk_payload(risk: Any) -> dict[str, object]:
    return {
        "privacy_advantage_by_label": risk.privacy_advantage_by_label,
        "worst_privacy_advantage": risk.worst_privacy_advantage,
        "conditional_error_by_label_source": risk.conditional_error_by_label_source,
        "worst_conditional_error": risk.worst_conditional_error,
        "missing_strata": risk.missing_strata,
        "estimable": risk.estimable,
    }


def run_dataset(
    name: str,
    config: dict[str, object],
    *,
    seed: int,
    cap: int,
    token_count: int,
    released_count: int,
    delta: float,
    privacy_thresholds: tuple[float, ...],
    contaminations: tuple[float, ...],
    smoothing: float,
) -> dict[str, object]:
    store_path = Path(config["path"])
    store = load_frozen_store(store_path, target_mode=str(config["target_mode"]))
    fit_indices = random_cap_sample(
        np.flatnonzero(store.split == SPLIT_TRAIN),
        maximum_total=cap,
        seed=seed * 100 + 1,
    )
    certification_indices = selected_indices(store, SPLIT_VALIDATION, cap, seed * 100 + 2)
    try:
        external_indices = selected_indices(store, SPLIT_EXTERNAL, cap, seed * 100 + 3)
    except ValueError:
        external_indices = np.flatnonzero(store.split == SPLIT_EXTERNAL).astype(np.int64)
        if len(external_indices) > cap:
            rng = np.random.default_rng(seed * 100 + 3)
            external_indices = np.sort(rng.choice(external_indices, size=cap, replace=False))
    tokenizer = fit_score_tokenizer(
        store.features[fit_indices],
        store.target[fit_indices],
        token_count=token_count,
        seed=seed,
    )
    certification_tokens = tokenizer.encode(store.features[certification_indices])
    external_tokens = tokenizer.encode(store.features[external_indices])
    table = build_token_table(
        certification_tokens,
        store.target[certification_indices],
        store.source[certification_indices],
        token_count=token_count,
        familywise_delta=delta,
    )
    external_table = build_token_table(
        external_tokens,
        store.target[external_indices],
        store.source[external_indices],
        token_count=token_count,
        familywise_delta=0.5,
    )
    transforms = ordered_smoothing_library(token_count, smoothing=smoothing)
    libraries = tuple(transforms for _ in range(2))
    scenarios = []
    for contamination in contaminations:
        for privacy_threshold in privacy_thresholds:
            record: dict[str, object] = {
                "contamination": contamination,
                "privacy_advantage_threshold": privacy_threshold,
            }
            try:
                solution = optimize_invariant_channel(
                    table.probabilities,
                    l1_radii=table.l1_radii,
                    common_channels_by_label=libraries,
                    contaminations=[contamination, contamination],
                    privacy_advantage_thresholds=[privacy_threshold, privacy_threshold],
                    released_token_count=released_count,
                    solver_time_limit_seconds=120.0,
                )
                external_risk = evaluate_external_channel(
                    external_tokens,
                    store.target[external_indices],
                    store.source[external_indices],
                    solution.release_channel,
                    solution.decoder,
                )
                record.update(
                    {
                        "mosaic_feasible": True,
                        "certified_worst_conditional_error": solution.certified_worst_conditional_error,
                        "certified_privacy_advantages": [
                            value.normalized_advantage for value in solution.privacy_certificates
                        ],
                        "release_channel": solution.release_channel.tolist(),
                        "decoder": solution.decoder,
                        "solver_mip_gap": solution.solver_mip_gap,
                        "max_constraint_violation": solution.max_constraint_violation,
                        "external_risk": risk_payload(external_risk),
                    }
                )
            except RuntimeError as error:
                record.update({"mosaic_feasible": False, "abstention_reason": str(error)})
            try:
                plugin = optimize_population_external_channel(
                    table.probabilities,
                    common_channels_by_label=libraries,
                    contaminations=[contamination, contamination],
                    privacy_advantage_thresholds=[privacy_threshold, privacy_threshold],
                    released_token_count=released_count,
                    solver_time_limit_seconds=120.0,
                )
                plugin_risk = evaluate_external_channel(
                    external_tokens,
                    store.target[external_indices],
                    store.source[external_indices],
                    plugin.release_channel,
                    plugin.decoder,
                )
                record["plugin"] = {
                    "feasible": True,
                    "validation_worst_conditional_error": plugin.exact_worst_conditional_error,
                    "release_channel": plugin.release_channel.tolist(),
                    "decoder": plugin.decoder,
                    "external_risk": risk_payload(plugin_risk),
                }
            except RuntimeError as error:
                record["plugin"] = {"feasible": False, "reason": str(error)}
            scenarios.append(record)
    eta_fit = []
    for label in range(2):
        if any(external_table.counts[label, source].sum() == 0 for source in range(2)):
            eta_fit.append(None)
        else:
            eta_fit.append(
                minimum_contamination_fraction(
                    table.probabilities[label],
                    external_table.probabilities[label],
                    transforms,
                )
            )
    return {
        "dataset": name,
        "modality": config["modality"],
        "target_mode": config["target_mode"],
        "store_manifest_sha256": sha256(store_path / "manifest.json"),
        "fit_examples": len(fit_indices),
        "certification_examples": len(certification_indices),
        "external_examples": len(external_indices),
        "certification_stratum_counts": table.counts.sum(axis=2).tolist(),
        "external_stratum_counts": external_table.counts.sum(axis=2).tolist(),
        "l1_radii": table.l1_radii.tolist(),
        "tokenizer_thresholds": tokenizer.thresholds,
        "plug_in_minimum_contamination_by_label": eta_fit,
        "external_support_complete": all(value is not None for value in eta_fit),
        "scenarios": scenarios,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cap", type=int, default=8000)
    parser.add_argument("--datasets", nargs="*", choices=tuple(DATASETS), default=list(DATASETS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = {
        "seed": args.seed,
        "cap_per_split": args.cap,
        "token_count": 4,
        "released_token_count": 2,
        "delta": 0.05,
        "privacy_advantage_thresholds": [0.25, 0.35, 0.45],
        "contaminations": [0.0, 0.05, 0.10],
        "ordered_smoothing": 0.10,
    }
    results = []
    for name in args.datasets:
        print(f"running {name}", flush=True)
        results.append(
            run_dataset(
                name,
                DATASETS[name],
                seed=args.seed,
                cap=args.cap,
                token_count=protocol["token_count"],
                released_count=protocol["released_token_count"],
                delta=protocol["delta"],
                privacy_thresholds=tuple(protocol["privacy_advantage_thresholds"]),
                contaminations=tuple(protocol["contaminations"]),
                smoothing=protocol["ordered_smoothing"],
            )
        )
    payload = {
        "name": "MOSAIC real-feature exploratory pilot v1",
        "status": "exploratory_not_confirmatory",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "results": results,
        "scope": (
            "External benchmark risks and shift-membership LPs are exploratory. "
            "They do not establish that a population belongs to the structured "
            "shift class. Camelyon external support is expected to be incomplete."
        ),
    }
    atomic_json_dump(payload, args.output)
    print(json.dumps({"output": str(args.output), "datasets": len(results)}, indent=2))


if __name__ == "__main__":
    main()
