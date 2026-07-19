#!/usr/bin/env python3
"""Run a hash-locked bridge model-misspecification confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Sequence

import numpy as np
from scipy.stats import beta

from mosaic_bridge import certify_bridge_membership
from run_mosaic_synthetic_pilot import simultaneous_radii


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
DEFAULT_PREREG = ROOT / "prereg_mosaic_bridge_misspecification_v1.json"
DEFAULT_OUTPUT = (
    REPOSITORY / "research/artifacts/mosaic_bridge_misspecification_v1.json"
)
REFERENCE = np.asarray(
    [
        [[0.80, 0.20], [0.60, 0.40]],
        [[0.20, 0.80], [0.40, 0.60]],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class ShiftScenario:
    name: str
    target: np.ndarray
    description: str


@dataclass(frozen=True)
class BridgeReplicate:
    scenario: str
    sample_size_per_stratum: int
    seed: int
    reference_empirical: tuple[tuple[tuple[float, ...], ...], ...]
    target_empirical: tuple[tuple[tuple[float, ...], ...], ...]
    reference_radii: tuple[tuple[float, ...], ...]
    target_radii: tuple[tuple[float, ...], ...]
    retained_masses: tuple[float, ...]
    minimum_retained_mass: float
    population_retained_masses: tuple[float, ...]
    population_minimum_retained_mass: float
    population_centered_retained_masses: tuple[float, ...]
    population_centered_minimum_retained_mass: float
    model_valid: bool
    accepted_membership: bool
    false_acceptance: bool
    joint_confidence_event: bool
    failure_on_confidence_event: bool


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scenario_registry() -> tuple[ShiftScenario, ...]:
    common = np.asarray([[0.90, 0.10], [0.10, 0.90]], dtype=np.float64)
    common_target = REFERENCE @ common

    residual = np.zeros_like(REFERENCE)
    residual[:, 0, 0] = 1.0
    residual[:, 1, 1] = 1.0
    understated = 0.75 * common_target + 0.25 * residual

    transform_zero = np.asarray([[1.0, 0.0], [0.5, 0.5]], dtype=np.float64)
    transform_one = np.asarray([[0.5, 0.5], [0.0, 1.0]], dtype=np.float64)
    source_specific = np.empty_like(REFERENCE)
    source_specific[:, 0] = REFERENCE[:, 0] @ transform_zero
    source_specific[:, 1] = REFERENCE[:, 1] @ transform_one
    return (
        ShiftScenario(
            "compatible_common_transform",
            common_target,
            "One source-blind Markov transform and no differential contamination.",
        ),
        ShiftScenario(
            "underdeclared_contamination",
            understated,
            "True differential contamination is 0.25 while the contract permits 0.20.",
        ),
        ShiftScenario(
            "source_specific_transform",
            source_specific,
            "The two sources undergo different transformations before release.",
        ),
    )


def sample_tensor(
    rng: np.random.Generator, population: np.ndarray, n: int
) -> np.ndarray:
    return np.stack(
        [
            np.stack(
                [
                    rng.multinomial(n, population[label, source]) / n
                    for source in range(population.shape[1])
                ]
            )
            for label in range(population.shape[0])
        ]
    )


def exact_population_retained_masses(target: np.ndarray) -> tuple[float, ...]:
    zeros = np.zeros(REFERENCE.shape[:2], dtype=np.float64)
    certificate = certify_bridge_membership(
        REFERENCE,
        reference_l1_radii=zeros,
        bridge_empirical_distributions=target,
        bridge_l1_radii=zeros,
        numerical_margin=1e-12,
    )
    return certificate.retained_masses


def population_centered_retained_masses(
    target: np.ndarray, radii: np.ndarray, *, numerical_margin: float
) -> tuple[float, ...]:
    certificate = certify_bridge_membership(
        REFERENCE,
        reference_l1_radii=radii,
        bridge_empirical_distributions=target,
        bridge_l1_radii=radii,
        numerical_margin=numerical_margin,
    )
    return certificate.retained_masses


def confidence_event(
    empirical: np.ndarray, population: np.ndarray, radii: np.ndarray
) -> bool:
    return bool(
        np.all(np.abs(empirical - population).sum(axis=2) <= radii + 1e-12)
    )


def as_nested_tuple(array: np.ndarray) -> tuple[tuple[tuple[float, ...], ...], ...]:
    return tuple(
        tuple(tuple(float(value) for value in row) for row in label)
        for label in array
    )


def as_matrix_tuple(array: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in array)


def run_replicate(
    payload: tuple[
        int,
        int,
        ShiftScenario,
        float,
        float,
        float,
        tuple[float, ...],
        tuple[float, ...],
    ]
) -> BridgeReplicate:
    (
        seed,
        n,
        scenario,
        delta,
        minimum_retained_mass,
        numerical_margin,
        population_retained,
        centered_retained,
    ) = payload
    rng = np.random.default_rng(seed)
    reference_empirical = sample_tensor(rng, REFERENCE, n)
    target_empirical = sample_tensor(rng, scenario.target, n)
    radii = simultaneous_radii(
        n,
        label_count=REFERENCE.shape[0],
        source_count=REFERENCE.shape[1],
        fine_count=REFERENCE.shape[2],
        delta=delta / 2.0,
    )
    joint_event = bool(
        confidence_event(reference_empirical, REFERENCE, radii)
        and confidence_event(target_empirical, scenario.target, radii)
    )
    certificate = certify_bridge_membership(
        reference_empirical,
        reference_l1_radii=radii,
        bridge_empirical_distributions=target_empirical,
        bridge_l1_radii=radii,
        numerical_margin=numerical_margin,
    )
    retained = certificate.retained_masses
    empirical_minimum = min(retained)
    population_minimum = min(population_retained)
    model_valid = bool(population_minimum >= minimum_retained_mass - 1e-10)
    accepted = bool(empirical_minimum >= minimum_retained_mass)
    false_acceptance = bool(accepted and not model_valid)
    return BridgeReplicate(
        scenario=scenario.name,
        sample_size_per_stratum=n,
        seed=seed,
        reference_empirical=as_nested_tuple(reference_empirical),
        target_empirical=as_nested_tuple(target_empirical),
        reference_radii=as_matrix_tuple(radii),
        target_radii=as_matrix_tuple(radii),
        retained_masses=tuple(float(value) for value in retained),
        minimum_retained_mass=float(empirical_minimum),
        population_retained_masses=population_retained,
        population_minimum_retained_mass=float(population_minimum),
        population_centered_retained_masses=centered_retained,
        population_centered_minimum_retained_mass=float(min(centered_retained)),
        model_valid=model_valid,
        accepted_membership=accepted,
        false_acceptance=false_acceptance,
        joint_confidence_event=joint_event,
        failure_on_confidence_event=bool(false_acceptance and joint_event),
    )


def cp_interval(successes: int, trials: int) -> tuple[float, float]:
    lower = 0.0 if successes == 0 else float(
        beta.ppf(0.025, successes, trials - successes + 1)
    )
    upper = 1.0 if successes == trials else float(
        beta.ppf(0.975, successes + 1, trials - successes)
    )
    return lower, upper


def aggregate(rows: Sequence[BridgeReplicate]) -> list[dict[str, object]]:
    keys = sorted({(row.scenario, row.sample_size_per_stratum) for row in rows})
    cells: list[dict[str, object]] = []
    for scenario, n in keys:
        subset = [
            row
            for row in rows
            if row.scenario == scenario and row.sample_size_per_stratum == n
        ]
        count = len(subset)
        accepts = sum(row.accepted_membership for row in subset)
        false_acceptances = sum(row.false_acceptance for row in subset)
        accept_lower, accept_upper = cp_interval(accepts, count)
        false_lower, false_upper = cp_interval(false_acceptances, count)
        values = np.asarray(
            [row.minimum_retained_mass for row in subset], dtype=np.float64
        )
        cells.append(
            {
                "scenario": scenario,
                "sample_size_per_stratum": n,
                "replicates": count,
                "model_valid": subset[0].model_valid,
                "population_minimum_retained_mass": subset[
                    0
                ].population_minimum_retained_mass,
                "population_centered_minimum_retained_mass": subset[
                    0
                ].population_centered_minimum_retained_mass,
                "acceptances": accepts,
                "acceptance_rate": accepts / count,
                "acceptance_cp95_lower": accept_lower,
                "acceptance_cp95_upper": accept_upper,
                "false_acceptances": false_acceptances,
                "false_acceptance_rate": false_acceptances / count,
                "false_acceptance_cp95_lower": false_lower,
                "false_acceptance_cp95_upper": false_upper,
                "joint_confidence_events": sum(
                    row.joint_confidence_event for row in subset
                ),
                "failures_on_confidence_event": sum(
                    row.failure_on_confidence_event for row in subset
                ),
                "mean_minimum_retained_mass": float(np.mean(values)),
                "q05_minimum_retained_mass": float(np.quantile(values, 0.05)),
                "q95_minimum_retained_mass": float(np.quantile(values, 0.95)),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    prereg_hash = sha256(args.prereg)
    sidecar = args.prereg.with_suffix(args.prereg.suffix + ".sha256")
    if sidecar.read_text(encoding="utf-8").strip() != prereg_hash:
        raise AssertionError("misspecification preregistration sidecar mismatch")
    prereg: dict[str, Any] = json.loads(args.prereg.read_text(encoding="utf-8"))
    for relative, expected_hash in prereg["code_sha256"].items():
        if sha256(REPOSITORY / relative) != expected_hash:
            raise AssertionError(f"locked code hash mismatch: {relative}")

    scenarios = scenario_registry()
    config_by_name = {str(value["name"]): value for value in prereg["scenarios"]}
    if set(config_by_name) != {scenario.name for scenario in scenarios}:
        raise AssertionError("scenario registry differs from the lock")
    delta = float(prereg["familywise_delta"])
    minimum_retained = float(prereg["minimum_retained_mass"])
    numerical_margin = float(prereg["numerical_margin"])
    replicate_count = int(prereg["replicates_per_cell"])
    seed_base = int(prereg["seed_base"])
    sample_sizes = tuple(int(value) for value in prereg["sample_sizes_per_stratum"])

    payloads = []
    scenario_summaries = []
    for scenario_index, scenario in enumerate(scenarios):
        population_retained = exact_population_retained_masses(scenario.target)
        locked_population = tuple(
            float(value) for value in config_by_name[scenario.name]["population_retained_masses"]
        )
        if not np.allclose(population_retained, locked_population, atol=5e-10, rtol=0.0):
            raise AssertionError("population retained mass differs from the lock")
        scenario_summaries.append(
            {
                "name": scenario.name,
                "description": scenario.description,
                "population_retained_masses": list(population_retained),
                "model_valid": min(population_retained) >= minimum_retained - 1e-10,
            }
        )
        for sample_index, n in enumerate(sample_sizes):
            radii = simultaneous_radii(
                n,
                label_count=REFERENCE.shape[0],
                source_count=REFERENCE.shape[1],
                fine_count=REFERENCE.shape[2],
                delta=delta / 2.0,
            )
            centered = population_centered_retained_masses(
                scenario.target, radii, numerical_margin=numerical_margin
            )
            for replicate in range(replicate_count):
                seed = (
                    seed_base
                    + scenario_index * 100_000_000
                    + sample_index * 1_000_000
                    + replicate
                )
                payloads.append(
                    (
                        seed,
                        n,
                        scenario,
                        delta,
                        minimum_retained,
                        numerical_margin,
                        population_retained,
                        centered,
                    )
                )

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(run_replicate, payloads))
    cells = aggregate(rows)
    invalid_cells = [cell for cell in cells if not bool(cell["model_valid"])]
    pass_conditions = {
        "complete_rows": len(rows) == int(prereg["pass_conditions"]["complete_rows"]),
        "maximum_invalid_false_acceptance_rate": max(
            float(cell["false_acceptance_rate"]) for cell in invalid_cells
        )
        <= float(
            prereg["pass_conditions"]["maximum_invalid_false_acceptance_rate"]
        ),
        "confidence_event_failures": sum(
            int(cell["failures_on_confidence_event"]) for cell in cells
        )
        == int(prereg["pass_conditions"]["confidence_event_failures"]),
    }
    report: dict[str, object] = {
        "name": "MOSAIC bridge model-misspecification confirmation v1",
        "status": "complete",
        "preregistration_sha256": prereg_hash,
        "code_sha256": prereg["code_sha256"],
        "familywise_delta": delta,
        "minimum_retained_mass": minimum_retained,
        "maximum_declared_contamination": 1.0 - minimum_retained,
        "replicates_per_cell": replicate_count,
        "sample_sizes_per_stratum": list(sample_sizes),
        "scenarios": scenario_summaries,
        "cells": cells,
        "pass_conditions": pass_conditions,
        "pass": all(pass_conditions.values()),
        "replicate_results": [asdict(row) for row in rows],
        "claim_boundary": prereg["claim_boundary"],
    }
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "replicate_results"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
