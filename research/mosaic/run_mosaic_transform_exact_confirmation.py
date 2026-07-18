#!/usr/bin/env python3
"""Run the locked transform-exact MOSAIC refinement confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
from scipy.stats import beta

from run_mosaic_synthetic_pilot import Scenario
from run_mosaic_transform_exact_pilot import (
    METHODS,
    RefinementResult,
    aggregate,
    run_refinement_replicate,
)


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
DEFAULT_PREREG = ROOT / "prereg_mosaic_transform_exact_v1.json"
DEFAULT_SIDECAR = ROOT / "prereg_mosaic_transform_exact_v1.sha256"
DEFAULT_OUTPUT = REPOSITORY / "research/artifacts/mosaic_transform_exact_confirmation_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def verify_lock(prereg: Path, sidecar: Path) -> tuple[dict[str, Any], str]:
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    actual = sha256(prereg)
    if actual != expected:
        raise RuntimeError("preregistration sidecar mismatch")
    config = load_json(prereg)
    for group in ("code_sha256", "pilot_artifact_sha256"):
        for relative, expected_hash in config[group].items():
            if sha256(REPOSITORY / relative) != expected_hash:
                raise RuntimeError(f"locked hash mismatch: {relative}")
    return config, actual


def scenario_from_config(config: dict[str, Any], population: dict[str, Any]) -> Scenario:
    laws = np.asarray(population["laws"], dtype=np.float64)
    transforms = tuple(
        np.asarray(transform, dtype=np.float64)
        for transform in population["common_transform_extremes"]
    )
    contamination = float(config["contamination"])
    privacy = float(config["privacy_threshold"])
    return Scenario(
        name=str(config["name"]),
        population=laws,
        libraries=tuple(transforms for _ in range(laws.shape[0])),
        contaminations=tuple(contamination for _ in range(laws.shape[0])),
        privacy_thresholds=tuple(privacy for _ in range(laws.shape[0])),
        utility_threshold=float(config["utility_threshold"]),
        released_token_count=int(population["released_token_count"]),
    )


def cp_interval(successes: int, trials: int) -> tuple[float, float]:
    lower = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(
        beta.ppf(0.975, successes + 1, trials - successes)
    )
    return lower, upper


def add_intervals(cells: list[dict[str, object]]) -> None:
    for cell in cells:
        trials = int(cell["replicates"])
        for field, prefix in (
            ("deployments", "deployment"),
            ("false_acceptances", "false_acceptance"),
            ("safe_deployments", "safe_deployment"),
        ):
            lower, upper = cp_interval(int(cell[field]), trials)
            cell[f"{prefix}_cp95_lower"] = lower
            cell[f"{prefix}_cp95_upper"] = upper


def evaluate_gates(
    rows: list[RefinementResult], cells: list[dict[str, object]], config: dict[str, Any]
) -> dict[str, object]:
    expected_cells = sum(
        len(scenario["sample_sizes_per_stratum"])
        for scenario in config["scenarios"]
    )
    replicates = int(config["replicates_per_cell"])
    complete = len(rows) == expected_cells * replicates * len(METHODS)
    coverage = all(
        float(cell["false_acceptance_rate"]) <= float(config["delta"]) + 1e-12
        and int(cell["failures_on_confidence_event"]) == 0
        for cell in cells
    )
    index = {
        (row.scenario, row.sample_size_per_stratum, row.seed, row.method): row
        for row in rows
    }
    paired = []
    for key, exact in index.items():
        if key[3] != "transform_exact":
            continue
        transfer = index[(key[0], key[1], key[2], "capacity_transfer")]
        paired.append(
            exact.certified_worst_conditional_error
            <= transfer.certified_worst_conditional_error + 2e-7
        )
    dominance = len(paired) == expected_cells * replicates and all(paired)
    cell_index = {
        (str(cell["scenario"]), int(cell["sample_size_per_stratum"]), str(cell["method"])): cell
        for cell in cells
    }

    def retention_gate(n: int, name: str) -> tuple[bool, float, float]:
        exact = float(
            cell_index[("retention_and_exactness_value", n, "transform_exact")][
                "safe_deployment_rate"
            ]
        )
        transfer = float(
            cell_index[("retention_and_exactness_value", n, "capacity_transfer")][
                "safe_deployment_rate"
            ]
        )
        gate = config["pass_conditions"][name]
        passed = exact >= float(gate["minimum_transform_exact_safe_deployment_rate"]) and (
            exact - transfer >= float(gate["minimum_safe_deployment_margin_over_capacity_transfer"])
        )
        return passed, exact, transfer

    n125_pass, n125_exact, n125_transfer = retention_gate(125, "retention_n125")
    n250_pass, n250_exact, n250_transfer = retention_gate(250, "retention_n250")
    all_pass = bool(complete and coverage and dominance and n125_pass and n250_pass)
    return {
        "complete_execution": complete,
        "coverage": coverage,
        "pointwise_dominance": dominance,
        "retention_n125": n125_pass,
        "retention_n250": n250_pass,
        "all_pass_before_independent_audit": all_pass,
        "retention_n125_transform_exact": n125_exact,
        "retention_n125_capacity_transfer": n125_transfer,
        "retention_n250_transform_exact": n250_exact,
        "retention_n250_capacity_transfer": n250_transfer,
    }


def atomic_json(payload: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, prereg_hash = verify_lock(args.prereg, args.sidecar)
    if args.verify_only:
        print(json.dumps({"verified": True, "preregistration_sha256": prereg_hash}, indent=2))
        return
    if args.output.exists():
        raise FileExistsError("refusing to overwrite confirmation output")
    replicates = int(config["replicates_per_cell"])
    all_rows: list[RefinementResult] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for scenario_index, scenario_config in enumerate(config["scenarios"]):
            scenario = scenario_from_config(scenario_config, config["population"])
            for n_value in scenario_config["sample_sizes_per_stratum"]:
                n = int(n_value)
                seeds = [
                    int(config["seed_base"]) + scenario_index * 10_000_000 + n * 10_000 + replicate
                    for replicate in range(replicates)
                ]
                nested = list(
                    executor.map(
                        run_refinement_replicate,
                        [(seed, n, scenario, float(config["delta"])) for seed in seeds],
                    )
                )
                all_rows.extend(result for pair in nested for result in pair)
    cells = aggregate(all_rows)
    add_intervals(cells)
    gates = evaluate_gates(all_rows, cells, config)
    report: dict[str, object] = {
        "name": str(config["confirmation_name"]),
        "status": "complete_confirmatory_result",
        "preregistration": str(args.prereg),
        "preregistration_sha256": prereg_hash,
        "code_sha256": config["code_sha256"],
        "pilot_artifact_sha256": config["pilot_artifact_sha256"],
        "methods": list(METHODS),
        "replicates_per_cell": replicates,
        "scenarios": config["scenarios"],
        "cells": cells,
        "pass_conditions": gates,
        "replicate_results": [asdict(row) for row in all_rows],
        "claim_boundary": config["claim_boundary"],
    }
    atomic_json(report, args.output)
    summary = {
        key: value for key, value in report.items() if key != "replicate_results"
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not gates["all_pass_before_independent_audit"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
