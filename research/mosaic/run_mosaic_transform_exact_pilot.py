#!/usr/bin/env python3
"""Pilot transform-exact MOSAIC against the capacity-transfer fallback."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Sequence

import numpy as np

from mosaic_optimizer import optimize_invariant_channel
from mosaic_transform_exact_optimizer import optimize_transform_exact_channel
from run_mosaic_synthetic_pilot import (
    Scenario,
    Selection,
    confidence_event,
    empirical_table,
    exact_risks,
    simultaneous_radii,
    witness_scenario,
)


METHODS = ("capacity_transfer", "transform_exact")


@dataclass(frozen=True)
class RefinementResult:
    seed: int
    scenario: str
    sample_size_per_stratum: int
    method: str
    deployed: bool
    exact_safe: bool
    false_acceptance: bool
    exact_worst_privacy_advantage: float
    exact_worst_conditional_error: float
    certified_worst_conditional_error: float
    certified_privacy_advantages: tuple[float, ...]
    release_channel: tuple[tuple[float, ...], ...]
    decoder: tuple[int, ...]
    confidence_event: bool
    failure_on_confidence_event: bool
    empirical_table: tuple[tuple[tuple[float, ...], ...], ...]
    l1_radii: tuple[tuple[float, ...], ...]
    solver_status: str
    solver_gap: float
    solver_dual_bound: float
    max_constraint_violation: float


def _selection(channel: np.ndarray, decoder: Sequence[int], criterion: float) -> Selection:
    return Selection(
        channel=np.asarray(channel, dtype=np.float64),
        decoder=tuple(int(value) for value in decoder),
        criterion=float(criterion),
    )


def result_for_solution(
    *,
    seed: int,
    n: int,
    scenario: Scenario,
    method: str,
    empirical: np.ndarray,
    radii: np.ndarray,
    event: bool,
    solution: object,
) -> RefinementResult:
    selection = _selection(
        solution.release_channel,
        solution.decoder,
        solution.certified_worst_conditional_error,
    )
    privacy_by_label, utility = exact_risks(scenario, selection)
    safe = bool(
        all(
            value <= scenario.privacy_thresholds[label] + 1e-9
            for label, value in enumerate(privacy_by_label)
        )
        and utility <= scenario.utility_threshold + 1e-9
    )
    privacy_certificates = tuple(
        float(certificate.normalized_advantage)
        for certificate in solution.privacy_certificates
    )
    deployed = bool(
        all(
            value <= scenario.privacy_thresholds[label] + 1e-10
            for label, value in enumerate(privacy_certificates)
        )
        and selection.criterion <= scenario.utility_threshold + 1e-10
    )
    false_acceptance = bool(deployed and not safe)
    return RefinementResult(
        seed=seed,
        scenario=scenario.name,
        sample_size_per_stratum=n,
        method=method,
        deployed=deployed,
        exact_safe=safe,
        false_acceptance=false_acceptance,
        exact_worst_privacy_advantage=float(max(privacy_by_label)),
        exact_worst_conditional_error=float(utility),
        certified_worst_conditional_error=selection.criterion,
        certified_privacy_advantages=privacy_certificates,
        release_channel=tuple(
            tuple(float(value) for value in row) for row in selection.channel
        ),
        decoder=selection.decoder,
        confidence_event=event,
        failure_on_confidence_event=bool(false_acceptance and event),
        empirical_table=tuple(
            tuple(tuple(float(value) for value in row) for row in label)
            for label in empirical
        ),
        l1_radii=tuple(tuple(float(value) for value in row) for row in radii),
        solver_status=str(solution.solver_status),
        solver_gap=float(solution.solver_mip_gap),
        solver_dual_bound=float(solution.solver_dual_bound),
        max_constraint_violation=float(solution.max_constraint_violation),
    )


def run_refinement_replicate(
    payload: tuple[int, int, Scenario, float]
) -> tuple[RefinementResult, RefinementResult]:
    seed, n, scenario, delta = payload
    rng = np.random.default_rng(seed)
    empirical = empirical_table(rng, scenario.population, n)
    radii = simultaneous_radii(
        n,
        label_count=empirical.shape[0],
        source_count=empirical.shape[1],
        fine_count=empirical.shape[2],
        delta=delta,
    )
    event = confidence_event(empirical, scenario.population, radii)
    arguments = dict(
        l1_radii=radii,
        common_channels_by_label=scenario.libraries,
        contaminations=scenario.contaminations,
        privacy_advantage_thresholds=scenario.privacy_thresholds,
        released_token_count=scenario.released_token_count,
    )
    transfer = optimize_invariant_channel(empirical, **arguments)
    exact = optimize_transform_exact_channel(empirical, **arguments)
    return (
        result_for_solution(
            seed=seed,
            n=n,
            scenario=scenario,
            method="capacity_transfer",
            empirical=empirical,
            radii=radii,
            event=event,
            solution=transfer,
        ),
        result_for_solution(
            seed=seed,
            n=n,
            scenario=scenario,
            method="transform_exact",
            empirical=empirical,
            radii=radii,
            event=event,
            solution=exact,
        ),
    )


def aggregate(results: Sequence[RefinementResult]) -> list[dict[str, object]]:
    cells = []
    keys = sorted(
        {
            (result.scenario, result.sample_size_per_stratum, result.method)
            for result in results
        }
    )
    for scenario, n, method in keys:
        subset = [
            result
            for result in results
            if (result.scenario, result.sample_size_per_stratum, result.method)
            == (scenario, n, method)
        ]
        count = len(subset)
        deployments = sum(result.deployed for result in subset)
        false_acceptances = sum(result.false_acceptance for result in subset)
        safe_deployments = sum(result.deployed and result.exact_safe for result in subset)
        cells.append(
            {
                "scenario": scenario,
                "sample_size_per_stratum": n,
                "method": method,
                "replicates": count,
                "deployments": deployments,
                "deployment_rate": deployments / count,
                "false_acceptances": false_acceptances,
                "false_acceptance_rate": false_acceptances / count,
                "safe_deployments": safe_deployments,
                "safe_deployment_rate": safe_deployments / count,
                "confidence_event_count": sum(result.confidence_event for result in subset),
                "failures_on_confidence_event": sum(
                    result.failure_on_confidence_event for result in subset
                ),
                "mean_certified_worst_error": float(
                    np.mean(
                        [result.certified_worst_conditional_error for result in subset]
                    )
                ),
            }
        )
    return cells


def atomic_json_dump(payload: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=output.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(output)


def scenario_payload(scenario: Scenario) -> dict[str, object]:
    return {
        "name": scenario.name,
        "population": scenario.population.tolist(),
        "libraries": [
            [transform.tolist() for transform in library]
            for library in scenario.libraries
        ],
        "contaminations": list(scenario.contaminations),
        "privacy_thresholds": list(scenario.privacy_thresholds),
        "utility_threshold": scenario.utility_threshold,
        "released_token_count": scenario.released_token_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=200)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=(125, 250, 500))
    parser.add_argument("--privacy-threshold", type=float, default=0.35)
    parser.add_argument("--utility-threshold", type=float, default=0.45)
    parser.add_argument("--contamination", type=float, default=0.10)
    parser.add_argument("--scenario-name", default="transform_exact_pilot")
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/artifacts/mosaic_transform_exact_pilot_v1.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.replicates <= 0 or args.workers <= 0:
        raise ValueError("replicates and workers must be positive")
    scenario = witness_scenario(
        privacy_threshold=args.privacy_threshold,
        utility_threshold=args.utility_threshold,
        contamination=args.contamination,
    )
    scenario = Scenario(
        name=str(args.scenario_name),
        population=scenario.population,
        libraries=scenario.libraries,
        contaminations=scenario.contaminations,
        privacy_thresholds=scenario.privacy_thresholds,
        utility_threshold=scenario.utility_threshold,
        released_token_count=scenario.released_token_count,
    )
    payloads = [
        (seed, int(n), scenario, float(args.delta))
        for n in args.sample_sizes
        for seed in range(args.seed_start, args.seed_start + args.replicates)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        nested = list(executor.map(run_refinement_replicate, payloads))
    results = [result for pair in nested for result in pair]
    report: dict[str, object] = {
        "name": "MOSAIC transform-exact disclosed pilot",
        "status": "pilot_not_confirmatory",
        "methods": list(METHODS),
        "scenario": scenario_payload(scenario),
        "replicates_per_cell": args.replicates,
        "sample_sizes_per_stratum": list(args.sample_sizes),
        "seed_start": args.seed_start,
        "delta": args.delta,
        "cells": aggregate(results),
        "replicate_results": [asdict(result) for result in results],
    }
    atomic_json_dump(report, args.output)
    summary = {
        key: value for key, value in report.items() if key != "replicate_results"
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
