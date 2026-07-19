#!/usr/bin/env python3
"""Measure MOSAIC scaling with larger alphabets and source counts."""

from __future__ import annotations

import argparse
import json
from math import ceil, log
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from mosaic_envelope import weissman_l1_radius
from mosaic_transform_exact_optimizer import optimize_transform_exact_channel


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "research/artifacts/mosaic_scaling_study_v1.json"
ALPHABETS = (4, 8, 16, 32, 64)
SOURCE_COUNTS = (2, 3, 4)
SEEDS = (4100, 4101, 4102, 4103, 4104)
LABEL_COUNT = 2
RELEASED_COUNT = 2
SAMPLE_SIZE = 2_000
FAMILY_FAILURE_PROBABILITY = 0.05
PRIVACY_THRESHOLD = 0.35
UTILITY_THRESHOLD = 0.40
UTILITY_FRONTIER = (0.10, 0.12, 0.15, 0.20, 0.40)
CONTAMINATION = 0.05


def population_table(token_count: int, source_count: int) -> np.ndarray:
    """Return a task-informative table with a removable source-specific spike."""

    if token_count < 4 or token_count % 2:
        raise ValueError("token_count must be even and at least four")
    if source_count < 2:
        raise ValueError("source_count must be at least two")
    half = token_count // 2
    table = np.empty((LABEL_COUNT, source_count, token_count), dtype=np.float64)
    for label in range(LABEL_COUNT):
        block = np.arange(label * half, (label + 1) * half)
        for source in range(source_count):
            row = np.full(token_count, 0.01 / token_count)
            row[block] += 0.84 / half
            row[label * half + source % half] += 0.15
            table[label, source] = row / row.sum()
    return table


def required_sample_size(
    token_count: int,
    source_count: int,
    *,
    target_radius: float,
) -> int:
    delta = FAMILY_FAILURE_PROBABILITY / (LABEL_COUNT * source_count)
    log_prefactor = token_count * log(2.0) + np.log1p(-(2.0 ** (1 - token_count)))
    return int(ceil(2.0 * (log_prefactor - log(delta)) / target_radius**2))


def sampled_table(
    population: np.ndarray, sample_size: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    counts = np.asarray(
        [
            [rng.multinomial(sample_size, row) for row in label_rows]
            for label_rows in population
        ],
        dtype=np.float64,
    )
    return counts / sample_size


def solve_cell(token_count: int, source_count: int, seed: int) -> dict[str, Any]:
    delta = FAMILY_FAILURE_PROBABILITY / (LABEL_COUNT * source_count)
    radius = weissman_l1_radius(SAMPLE_SIZE, token_count, delta)
    empirical = sampled_table(
        population_table(token_count, source_count), SAMPLE_SIZE, seed
    )
    started = perf_counter()
    solution = optimize_transform_exact_channel(
        empirical,
        l1_radii=np.full((LABEL_COUNT, source_count), radius),
        common_channels_by_label=(
            (np.eye(token_count),),
            (np.eye(token_count),),
        ),
        contaminations=(CONTAMINATION, CONTAMINATION),
        privacy_advantage_thresholds=(PRIVACY_THRESHOLD, PRIVACY_THRESHOLD),
        released_token_count=RELEASED_COUNT,
        attacker_constraint_generation=source_count >= 3,
    )
    elapsed = perf_counter() - started
    privacy = max(
        certificate.normalized_advantage
        for certificate in solution.privacy_certificates
    )
    error = solution.certified_worst_conditional_error
    return {
        "token_count": token_count,
        "source_count": source_count,
        "seed": seed,
        "sample_size_per_stratum": SAMPLE_SIZE,
        "l1_radius": radius,
        "required_n_for_k4_radius": required_sample_size(
            token_count,
            source_count,
            target_radius=weissman_l1_radius(SAMPLE_SIZE, 4, delta),
        ),
        "wall_clock_seconds": elapsed,
        "certified_source_advantage": privacy,
        "privacy_slack": PRIVACY_THRESHOLD - privacy,
        "certified_worst_error": error,
        "utility_slack": UTILITY_THRESHOLD - error,
        "deploy": error <= UTILITY_THRESHOLD + 1e-10,
        "active_attacker_assignments": solution.active_attacker_assignments,
        "full_attacker_assignments": source_count**RELEASED_COUNT,
        "constraint_generation_iterations": solution.constraint_generation_iterations,
        "optimizer_method": solution.method,
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for token_count in ALPHABETS:
        for source_count in SOURCE_COUNTS:
            cell = [
                row
                for row in rows
                if row["token_count"] == token_count
                and row["source_count"] == source_count
            ]
            summary.append(
                {
                    "token_count": token_count,
                    "source_count": source_count,
                    "replicates": len(cell),
                    "retention_rate": float(np.mean([row["deploy"] for row in cell])),
                    "retention_by_utility_threshold": {
                        f"{threshold:.2f}": float(
                            np.mean(
                                [
                                    row["certified_worst_error"] <= threshold
                                    for row in cell
                                ]
                            )
                        )
                        for threshold in UTILITY_FRONTIER
                    },
                    "l1_radius": float(np.mean([row["l1_radius"] for row in cell])),
                    "required_n_for_k4_radius": int(
                        max(row["required_n_for_k4_radius"] for row in cell)
                    ),
                    "wall_clock_seconds_median": float(
                        np.median([row["wall_clock_seconds"] for row in cell])
                    ),
                    "wall_clock_seconds_max": float(
                        np.max([row["wall_clock_seconds"] for row in cell])
                    ),
                    "privacy_slack_median": float(
                        np.median([row["privacy_slack"] for row in cell])
                    ),
                    "utility_slack_median": float(
                        np.median([row["utility_slack"] for row in cell])
                    ),
                    "active_attacker_assignments_median": float(
                        np.median([row["active_attacker_assignments"] for row in cell])
                    ),
                    "full_attacker_assignments": source_count**RELEASED_COUNT,
                }
            )
    return summary


def hard_constraint_generation_point() -> dict[str, Any]:
    token_count = 16
    source_count = 4
    released_count = 4
    delta = FAMILY_FAILURE_PROBABILITY / (LABEL_COUNT * source_count)
    radius = weissman_l1_radius(SAMPLE_SIZE, token_count, delta)
    empirical = sampled_table(
        population_table(token_count, source_count), SAMPLE_SIZE, 4199
    )
    started = perf_counter()
    solution = optimize_transform_exact_channel(
        empirical,
        l1_radii=np.full((LABEL_COUNT, source_count), radius),
        common_channels_by_label=(
            (np.eye(token_count),),
            (np.eye(token_count),),
        ),
        contaminations=(CONTAMINATION, CONTAMINATION),
        privacy_advantage_thresholds=(PRIVACY_THRESHOLD, PRIVACY_THRESHOLD),
        released_token_count=released_count,
        decoder_candidates=((0, 0, 1, 1),),
        attacker_constraint_generation=True,
    )
    return {
        "token_count": token_count,
        "source_count": source_count,
        "released_token_count": released_count,
        "full_attacker_assignments": source_count**released_count,
        "active_attacker_assignments": solution.active_attacker_assignments,
        "constraint_generation_iterations": solution.constraint_generation_iterations,
        "wall_clock_seconds": perf_counter() - started,
        "certified_worst_error": solution.certified_worst_conditional_error,
        "maximum_certified_source_advantage": max(
            certificate.normalized_advantage
            for certificate in solution.privacy_certificates
        ),
        "method": solution.method,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    rows = [
        solve_cell(token_count, source_count, seed)
        for token_count in ALPHABETS
        for source_count in SOURCE_COUNTS
        for seed in SEEDS
    ]
    summary = summarize(rows)
    hard_point = hard_constraint_generation_point()
    payload = {
        "name": "MOSAIC finite-alphabet scaling study v1",
        "status": "post-review deterministic protocol; not preregistered",
        "claim_boundary": (
            "Synthetic tables isolate computational and multinomial-radius scaling. "
            "They do not establish natural-data prevalence or end-to-end feature quality."
        ),
        "settings": {
            "token_counts": ALPHABETS,
            "source_counts": SOURCE_COUNTS,
            "seeds": SEEDS,
            "label_count": LABEL_COUNT,
            "released_token_count": RELEASED_COUNT,
            "sample_size_per_stratum": SAMPLE_SIZE,
            "family_failure_probability": FAMILY_FAILURE_PROBABILITY,
            "privacy_threshold": PRIVACY_THRESHOLD,
            "utility_threshold": UTILITY_THRESHOLD,
            "utility_frontier": UTILITY_FRONTIER,
            "contamination": CONTAMINATION,
        },
        "summary": summary,
        "hard_constraint_generation_point": hard_point,
        "rows": rows,
        "pass": {
            "all_75_jobs_solved": len(rows) == 75,
            "all_cells_represented": len(summary) == 15,
            "largest_alphabet_solved": any(
                row["token_count"] == 64 for row in rows
            ),
            "four_source_case_solved": any(
                row["source_count"] == 4 for row in rows
            ),
            "constraint_generation_reduces_active_family": (
                hard_point["active_attacker_assignments"]
                < hard_point["full_attacker_assignments"]
            ),
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(arguments.output), "pass": payload["pass"]}, indent=2))


if __name__ == "__main__":
    main()
