#!/usr/bin/env python3
"""Run paired structure-aware baselines for the MOSAIC synthetic witness."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from typing import Sequence

import numpy as np

from mosaic_invariant import (
    adaptive_pre_release_attacker_certificate,
    pre_release_utility_certificate,
)
from mosaic_optimizer import optimize_invariant_channel
from run_mosaic_synthetic_confirmation import add_intervals, atomic_json_dump
from run_mosaic_synthetic_pilot import (
    ReplicateResult,
    Scenario,
    Selection,
    aggregate,
    confidence_event,
    deterministic_selection,
    empirical_table,
    result_for_selection,
    selection_from_mosaic_solution,
    simultaneous_radii,
    witness_scenario,
)


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
DEFAULT_PREREG = ROOT / "prereg_mosaic_baseline_extension_v1.json"
DEFAULT_OUTPUT = REPOSITORY / "research/artifacts/mosaic_baseline_extension_v1.json"
GRID = tuple(float(value) for value in np.linspace(0.0, 1.0, 5))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_family(
    scenario: Scenario,
) -> tuple[tuple[np.ndarray, tuple[int, ...]], ...]:
    fine_count = scenario.population.shape[2]
    label_count = scenario.population.shape[0]
    released_count = scenario.released_token_count
    channels = tuple(
        np.asarray(
            [[probability, 1.0 - probability] for probability in probabilities],
            dtype=np.float64,
        )
        for probabilities in product(GRID, repeat=fine_count)
    )
    decoders = tuple(product(range(label_count), repeat=released_count))
    return tuple((channel, tuple(decoder)) for channel in channels for decoder in decoders)


def empirical_external_error(
    empirical: np.ndarray,
    channel: np.ndarray,
    decoder: Sequence[int],
    scenario: Scenario,
) -> float:
    decoder_array = np.asarray(decoder, dtype=np.int64)
    worst = 0.0
    for label in range(empirical.shape[0]):
        loss = (decoder_array != label).astype(np.float64)
        row_loss = channel @ loss
        residual = float(np.max(row_loss))
        eta = float(scenario.contaminations[label])
        for source in range(empirical.shape[1]):
            for transform in scenario.libraries[label]:
                common = float(empirical[label, source] @ transform @ row_loss)
                worst = max(worst, (1.0 - eta) * common + eta * residual)
    return worst


def candidate_safety_p_value(
    empirical: np.ndarray,
    channel: np.ndarray,
    decoder: Sequence[int],
    scenario: Scenario,
    *,
    n: int,
) -> float:
    """Composite p-value for the null that any registered contract fails."""

    label_count, source_count, _ = empirical.shape
    released_count = channel.shape[1]
    assignments = tuple(product(range(source_count), repeat=released_count))
    component_p_values: list[float] = []
    chance = 1.0 / source_count
    for label in range(label_count):
        eta = float(scenario.contaminations[label])
        retained = 1.0 - eta
        ba_threshold = chance + (1.0 - chance) * float(
            scenario.privacy_thresholds[label]
        )
        for transform in scenario.libraries[label]:
            for assignment in assignments:
                assignment_array = np.asarray(assignment, dtype=np.int64)
                common = 0.0
                residual = 0.0
                for source in range(source_count):
                    correct = (assignment_array == source).astype(np.float64)
                    fine_score = channel @ correct
                    common += float(empirical[label, source] @ transform @ fine_score)
                    residual += float(np.max(fine_score))
                common /= source_count
                residual /= source_count
                if retained <= 0.0:
                    p_value = 0.0 if residual <= ba_threshold else 1.0
                else:
                    null_boundary = (ba_threshold - eta * residual) / retained
                    gap = max(0.0, null_boundary - common)
                    p_value = float(np.exp(-2.0 * source_count * n * gap * gap))
                component_p_values.append(min(1.0, max(0.0, p_value)))

        decoder_array = np.asarray(decoder, dtype=np.int64)
        loss = (decoder_array != label).astype(np.float64)
        fine_loss = channel @ loss
        residual_loss = float(np.max(fine_loss))
        for source in range(source_count):
            for transform in scenario.libraries[label]:
                common_loss = float(
                    empirical[label, source] @ transform @ fine_loss
                )
                if retained <= 0.0:
                    p_value = (
                        0.0
                        if residual_loss <= scenario.utility_threshold
                        else 1.0
                    )
                else:
                    null_boundary = (
                        scenario.utility_threshold - eta * residual_loss
                    ) / retained
                    gap = max(0.0, null_boundary - common_loss)
                    p_value = float(np.exp(-2.0 * n * gap * gap))
                component_p_values.append(min(1.0, max(0.0, p_value)))
    # The unsafe null is a union over component nulls.  The maximum component
    # p-value is valid for that union because it dominates the p-value of every
    # component that could make the candidate unsafe.
    return max(component_p_values, default=1.0)


def holm_rejections(p_values: Sequence[float], *, delta: float) -> tuple[int, ...]:
    values = tuple(float(value) for value in p_values)
    if not 0.0 < delta < 1.0 or any(
        not np.isfinite(value) or value < 0.0 or value > 1.0 for value in values
    ):
        raise ValueError("invalid p-values or familywise level")
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    rejected: list[int] = []
    family_size = len(values)
    for rank, index in enumerate(order):
        if values[index] <= delta / (family_size - rank):
            rejected.append(index)
        else:
            break
    return tuple(rejected)


def holm_ltt_selection(
    empirical: np.ndarray,
    scenario: Scenario,
    *,
    n: int,
    delta: float,
) -> tuple[Selection, bool, float, int]:
    family = candidate_family(scenario)
    p_values = tuple(
        candidate_safety_p_value(
            empirical, channel, decoder, scenario, n=n
        )
        for channel, decoder in family
    )
    rejected = holm_rejections(p_values, delta=delta)
    if rejected:
        selected_index = min(
            rejected,
            key=lambda index: (
                empirical_external_error(
                    empirical, family[index][0], family[index][1], scenario
                ),
                index,
            ),
        )
        channel, decoder = family[selected_index]
        criterion = empirical_external_error(empirical, channel, decoder, scenario)
        return Selection(channel, decoder, criterion), True, p_values[selected_index], len(rejected)
    # Return a deterministic placeholder only for exact grading; it is never deployed.
    channel, decoder = family[0]
    return Selection(channel, decoder, 1.0), False, min(p_values), 0


def table_region_grid_selection(
    empirical: np.ndarray,
    radii: np.ndarray,
    scenario: Scenario,
) -> Selection:
    best: Selection | None = None
    best_index: int | None = None
    for index, (channel, decoder) in enumerate(candidate_family(scenario)):
        source_certificates = tuple(
            adaptive_pre_release_attacker_certificate(
                empirical[label],
                channel,
                l1_radii=radii[label],
                common_fine_token_channels=scenario.libraries[label],
                contamination=scenario.contaminations[label],
            )
            for label in range(empirical.shape[0])
        )
        if any(
            certificate.normalized_advantage
            > scenario.privacy_thresholds[label]
            for label, certificate in enumerate(source_certificates)
        ):
            continue
        error = max(
            pre_release_utility_certificate(
                empirical[label, source],
                channel,
                decoder,
                true_label=label,
                l1_radius=float(radii[label, source]),
                common_fine_token_channels=scenario.libraries[label],
                contamination=scenario.contaminations[label],
            ).error_probability
            for label in range(empirical.shape[0])
            for source in range(empirical.shape[1])
        )
        candidate = Selection(channel, decoder, float(error))
        if best is None or (candidate.criterion, index) < (best.criterion, best_index):
            best = candidate
            best_index = index
    if best is None:
        raise AssertionError("the finite grid must include a source-feasible constant channel")
    return best


def run_replicate(
    payload: tuple[int, int, Scenario, float]
) -> tuple[list[ReplicateResult], dict[str, object]]:
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

    mosaic_solution = optimize_invariant_channel(
        empirical,
        l1_radii=radii,
        common_channels_by_label=scenario.libraries,
        contaminations=scenario.contaminations,
        privacy_advantage_thresholds=scenario.privacy_thresholds,
        released_token_count=scenario.released_token_count,
    )
    mosaic = selection_from_mosaic_solution(mosaic_solution)
    grid = table_region_grid_selection(empirical, radii, scenario)
    deterministic = deterministic_selection(empirical, radii, scenario)
    holm, holm_deployed, holm_p, holm_rejections_count = holm_ltt_selection(
        empirical, scenario, n=n, delta=delta
    )
    results = [
        result_for_selection(
            seed=seed,
            n=n,
            method="mosaic_continuum",
            scenario=scenario,
            selection=mosaic,
            deployed=mosaic.criterion <= scenario.utility_threshold,
            event=event,
        ),
        result_for_selection(
            seed=seed,
            n=n,
            method="table_region_grid",
            scenario=scenario,
            selection=grid,
            deployed=grid.criterion <= scenario.utility_threshold,
            event=event,
        ),
        result_for_selection(
            seed=seed,
            n=n,
            method="holm_ltt_grid",
            scenario=scenario,
            selection=holm,
            deployed=holm_deployed,
            event=None,
        ),
        result_for_selection(
            seed=seed,
            n=n,
            method="fare_style_deterministic",
            scenario=scenario,
            selection=deterministic,
            deployed=deterministic.criterion <= scenario.utility_threshold,
            event=event,
        ),
    ]
    diagnostic = {
        "seed": seed,
        "scenario": scenario.name,
        "sample_size_per_stratum": n,
        "holm_selected_p_value": holm_p,
        "holm_rejections": holm_rejections_count,
        "confidence_event": event,
    }
    return results, diagnostic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sidecar = args.prereg.with_suffix(args.prereg.suffix + ".sha256")
    prereg_hash = sha256(args.prereg)
    if sidecar.read_text(encoding="utf-8").strip() != prereg_hash:
        raise ValueError("baseline-extension preregistration sidecar mismatch")
    config = json.loads(args.prereg.read_text(encoding="utf-8"))
    for relative, expected in config["code_sha256"].items():
        if sha256(REPOSITORY / relative) != expected:
            raise ValueError(f"locked code hash mismatch: {relative}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    scenarios = (
        (
            replace(
                witness_scenario(
                privacy_threshold=0.30,
                utility_threshold=0.40,
                contamination=0.20,
                ),
                name="hard_safety_boundary",
            ),
            125,
        ),
        (
            replace(
                witness_scenario(
                privacy_threshold=0.35,
                utility_threshold=0.45,
                contamination=0.10,
                ),
                name="retention_and_stochastic_value",
            ),
            250,
        ),
    )
    replicate_count = int(config["replicates_per_cell"])
    seed_base = int(config["seed_base"])
    delta = float(config["familywise_delta"])
    payloads = [
        (
            seed_base + scenario_index * 10_000_000 + replicate,
            n,
            scenario,
            delta,
        )
        for scenario_index, (scenario, n) in enumerate(scenarios)
        for replicate in range(replicate_count)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        nested = list(executor.map(run_replicate, payloads))
    results = [result for group, _ in nested for result in group]
    diagnostics = [diagnostic for _, diagnostic in nested]
    cells = aggregate(results)
    for cell in cells:
        matching = [
            scenario.name
            for scenario, n in scenarios
            if n == int(cell["sample_size_per_stratum"])
        ]
        cell["scenario"] = matching[0]
    add_intervals(cells)
    report: dict[str, object] = {
        "name": "MOSAIC paired structure-aware baseline extension v1",
        "status": "complete",
        "preregistration_sha256": prereg_hash,
        "code_sha256": config["code_sha256"],
        "replicates_per_cell": replicate_count,
        "familywise_delta": delta,
        "candidate_channels": len(GRID) ** 3,
        "candidate_channel_decoder_pairs": len(GRID) ** 3 * 4,
        "cells": cells,
        "diagnostics": diagnostics,
        "replicate_results": [asdict(result) for result in results],
        "scope": config["claim_boundary"],
    }
    atomic_json_dump(report, args.output)
    print(json.dumps({key: value for key, value in report.items() if key not in {"diagnostics", "replicate_results"}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
