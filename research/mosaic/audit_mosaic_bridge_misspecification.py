#!/usr/bin/env python3
"""Deterministically replay the locked bridge misspecification study."""

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

from run_mosaic_bridge_misspecification import (
    DEFAULT_PREREG,
    REFERENCE,
    BridgeReplicate,
    ShiftScenario,
    aggregate,
    confidence_event,
    exact_population_retained_masses,
    population_centered_retained_masses,
    sample_tensor,
    scenario_registry,
)
from mosaic_bridge import certify_bridge_membership
from run_mosaic_synthetic_pilot import simultaneous_radii


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
DEFAULT_REPORT = (
    REPOSITORY / "research/artifacts/mosaic_bridge_misspecification_v1.json"
)
DEFAULT_OUTPUT = (
    REPOSITORY / "research/artifacts/mosaic_bridge_misspecification_audit_v1.json"
)
TOLERANCE = 5e-9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def replay(
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
    seed, n, scenario, delta, threshold, margin, population_retained, centered = payload
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
    certificate = certify_bridge_membership(
        reference_empirical,
        reference_l1_radii=radii,
        bridge_empirical_distributions=target_empirical,
        bridge_l1_radii=radii,
        numerical_margin=margin,
    )
    retained = certificate.retained_masses
    empirical_minimum = min(retained)
    population_minimum = min(population_retained)
    model_valid = population_minimum >= threshold - 1e-10
    accepted = empirical_minimum >= threshold
    event = bool(
        confidence_event(reference_empirical, REFERENCE, radii)
        and confidence_event(target_empirical, scenario.target, radii)
    )
    false_acceptance = bool(accepted and not model_valid)
    nested = lambda array: tuple(
        tuple(tuple(float(value) for value in row) for row in label)
        for label in array
    )
    matrix = lambda array: tuple(
        tuple(float(value) for value in row) for row in array
    )
    return BridgeReplicate(
        scenario=scenario.name,
        sample_size_per_stratum=n,
        seed=seed,
        reference_empirical=nested(reference_empirical),
        target_empirical=nested(target_empirical),
        reference_radii=matrix(radii),
        target_radii=matrix(radii),
        retained_masses=tuple(float(value) for value in retained),
        minimum_retained_mass=float(empirical_minimum),
        population_retained_masses=population_retained,
        population_minimum_retained_mass=float(population_minimum),
        population_centered_retained_masses=centered,
        population_centered_minimum_retained_mass=float(min(centered)),
        model_valid=bool(model_valid),
        accepted_membership=bool(accepted),
        false_acceptance=false_acceptance,
        joint_confidence_event=event,
        failure_on_confidence_event=bool(false_acceptance and event),
    )


def assert_close(observed: object, expected: object, *, path: str) -> None:
    if isinstance(expected, (bool, str)) or expected is None:
        if observed != expected:
            raise AssertionError(f"{path} mismatch")
    elif isinstance(expected, (tuple, list)):
        np.testing.assert_allclose(
            np.asarray(observed, dtype=np.float64),
            np.asarray(expected, dtype=np.float64),
            atol=TOLERANCE,
            rtol=0.0,
            err_msg=path,
        )
    elif abs(float(observed) - float(expected)) > TOLERANCE:
        raise AssertionError(f"{path} mismatch")


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
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
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
    if args.prereg.with_suffix(args.prereg.suffix + ".sha256").read_text(
        encoding="utf-8"
    ).strip() != prereg_hash:
        raise AssertionError("preregistration sidecar mismatch")
    prereg = load_json(args.prereg)
    report = load_json(args.report)
    for relative, expected_hash in prereg["code_sha256"].items():
        if sha256(REPOSITORY / relative) != expected_hash:
            raise AssertionError(f"locked code hash mismatch: {relative}")
    if report["preregistration_sha256"] != prereg_hash:
        raise AssertionError("report uses the wrong preregistration")
    if report["code_sha256"] != prereg["code_sha256"]:
        raise AssertionError("report code lock mismatch")

    delta = float(prereg["familywise_delta"])
    threshold = float(prereg["minimum_retained_mass"])
    margin = float(prereg["numerical_margin"])
    replicate_count = int(prereg["replicates_per_cell"])
    seed_base = int(prereg["seed_base"])
    sample_sizes = tuple(int(value) for value in prereg["sample_sizes_per_stratum"])
    scenarios = scenario_registry()
    payloads = []
    for scenario_index, scenario in enumerate(scenarios):
        population_retained = exact_population_retained_masses(scenario.target)
        for sample_index, n in enumerate(sample_sizes):
            radii = simultaneous_radii(
                n,
                label_count=REFERENCE.shape[0],
                source_count=REFERENCE.shape[1],
                fine_count=REFERENCE.shape[2],
                delta=delta / 2.0,
            )
            centered = population_centered_retained_masses(
                scenario.target, radii, numerical_margin=margin
            )
            payloads.extend(
                (
                    seed_base
                    + scenario_index * 100_000_000
                    + sample_index * 1_000_000
                    + replicate,
                    n,
                    scenario,
                    delta,
                    threshold,
                    margin,
                    population_retained,
                    centered,
                )
                for replicate in range(replicate_count)
            )
    observed = {
        (
            str(row["scenario"]),
            int(row["sample_size_per_stratum"]),
            int(row["seed"]),
        ): row
        for row in report["replicate_results"]
    }
    if len(observed) != len(payloads):
        raise AssertionError("replicate grid is incomplete or duplicated")
    replayed: list[BridgeReplicate] = []
    field_checks = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for expected in executor.map(replay, payloads):
            key = (expected.scenario, expected.sample_size_per_stratum, expected.seed)
            stored = observed[key]
            expected_dict = asdict(expected)
            if set(stored) != set(expected_dict):
                raise AssertionError("replicate row has unexpected fields")
            for field, value in expected_dict.items():
                assert_close(stored[field], value, path=f"{key}.{field}")
                field_checks += 1
            replayed.append(expected)

    expected_cells = {
        (str(cell["scenario"]), int(cell["sample_size_per_stratum"])): cell
        for cell in aggregate(replayed)
    }
    observed_cells = {
        (str(cell["scenario"]), int(cell["sample_size_per_stratum"])): cell
        for cell in report["cells"]
    }
    if set(expected_cells) != set(observed_cells):
        raise AssertionError("aggregate cell grid mismatch")
    aggregate_checks = 0
    for key, expected in expected_cells.items():
        stored = observed_cells[key]
        if set(stored) != set(expected):
            raise AssertionError("aggregate cell has unexpected fields")
        for field, value in expected.items():
            assert_close(stored[field], value, path=f"{key}.{field}")
            aggregate_checks += 1

    invalid_cells = [cell for cell in expected_cells.values() if not cell["model_valid"]]
    maximum_false_rate = max(
        float(cell["false_acceptance_rate"]) for cell in invalid_cells
    )
    confidence_failures = sum(
        int(cell["failures_on_confidence_event"]) for cell in expected_cells.values()
    )
    pass_conditions = {
        "complete_rows": len(replayed) == int(prereg["pass_conditions"]["complete_rows"]),
        "maximum_invalid_false_acceptance_rate": maximum_false_rate
        <= float(prereg["pass_conditions"]["maximum_invalid_false_acceptance_rate"]),
        "confidence_event_failures": confidence_failures
        == int(prereg["pass_conditions"]["confidence_event_failures"]),
    }
    if report["pass_conditions"] != pass_conditions:
        raise AssertionError("stored pass conditions mismatch replay")
    payload: dict[str, object] = {
        "name": "MOSAIC bridge misspecification deterministic replay v1",
        "status": "development_replay_not_independent_human_review",
        "preregistration_sha256": prereg_hash,
        "report_sha256": sha256(args.report),
        "tables_replayed": len(replayed),
        "replicate_field_checks": field_checks,
        "aggregate_field_checks": aggregate_checks,
        "maximum_invalid_false_acceptance_rate": maximum_false_rate,
        "confidence_event_failures": confidence_failures,
        "pass_conditions": pass_conditions,
        "pass": all(pass_conditions.values()),
    }
    atomic_json_dump(payload, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
