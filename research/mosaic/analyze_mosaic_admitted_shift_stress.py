#!/usr/bin/env python3
"""Construct exact target laws inside learned MOSAIC bridge classes.

The direct target-table comparator covers its sampled target table. This audit
asks a different question: after that comparator deploys, can a target law move
within the candidate's learned bridge class and break the direct contract? The
stress law is explicit and is evaluated against the candidate-matched strict
MOSAIC channel on the same fine-token population.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import product
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
from scipy.stats import beta

from mosaic_channel import normalized_attacker_advantage
from mosaic_exact import exact_external_attacker_risk
from mosaic_real import sha256


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
DEFAULT_DIRECT = REPOSITORY / "research/artifacts/mosaic_direct_target_receipts_v1"
DEFAULT_STRICT = REPOSITORY / "research/artifacts/mosaic_bridge_strict_v2_receipts_v1"
DEFAULT_SPEC = ROOT / "MOSAIC_ADMITTED_SHIFT_STRESS_SPEC.md"
DEFAULT_OUTPUT = REPOSITORY / "research/artifacts/mosaic_admitted_shift_stress_v1.json"
UTILITY_THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.49)
PRIMARY_UTILITY_THRESHOLD = 0.40
PRIVACY_THRESHOLD = 0.35
TOLERANCE = 2e-7
TIE_TOLERANCE = 1e-12


def threshold_key(value: float) -> str:
    return f"{float(value):.2f}"


def probability_tensor(counts: object) -> np.ndarray:
    values = np.asarray(counts, dtype=np.float64)
    if values.ndim != 3 or np.any(values < 0.0):
        raise ValueError("token counts must be a nonnegative label-source-token tensor")
    totals = values.sum(axis=2, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("stress analysis requires complete reference support")
    return values / totals


def population_metrics(
    laws: np.ndarray, channel: np.ndarray, decoder: tuple[int, ...]
) -> tuple[float, float]:
    labels, sources, _ = laws.shape
    privacy: list[float] = []
    utility: list[float] = []
    decoder_array = np.asarray(decoder, dtype=np.int64)
    for label in range(labels):
        released = laws[label] @ channel
        balanced_accuracy = float(np.max(released, axis=0).sum() / sources)
        privacy.append(normalized_attacker_advantage(balanced_accuracy, sources))
        loss = (decoder_array != label).astype(np.float64)
        utility.extend(
            float(laws[label, source] @ channel @ loss)
            for source in range(sources)
        )
    return max(privacy), max(utility)


def canonical_worst_assignment(
    reference: np.ndarray,
    transform: np.ndarray,
    channel: np.ndarray,
    contamination: float,
    expected_balanced_accuracy: float,
) -> tuple[int, ...]:
    """Resolve exact-attacker ties independently of floating-point loop order."""

    source_count = reference.shape[0]
    released_count = channel.shape[1]
    common_released = reference @ transform @ channel
    matches: list[tuple[int, ...]] = []
    for assignment in product(range(source_count), repeat=released_count):
        assignment_array = np.asarray(assignment, dtype=np.int64)
        common_score = 0.0
        residual_score = 0.0
        for source in range(source_count):
            correct_outputs = (assignment_array == source).astype(np.float64)
            common_score += float(common_released[source] @ correct_outputs)
            residual_score += float(np.max(channel @ correct_outputs))
        balanced_accuracy = (
            (1.0 - contamination) * common_score + contamination * residual_score
        ) / source_count
        if abs(balanced_accuracy - expected_balanced_accuracy) <= TIE_TOLERANCE:
            matches.append(tuple(int(value) for value in assignment))
    if not matches:
        raise RuntimeError("could not recover a maximizing attacker assignment")
    return min(matches)


def worst_admitted_law(
    reference: np.ndarray,
    bridge_membership: dict[str, Any],
    direct_channel: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]], float]:
    labels, sources, tokens = reference.shape
    laws = np.zeros_like(reference)
    details: list[dict[str, Any]] = []
    maximum_membership_error = 0.0
    for label in range(labels):
        membership = bridge_membership["labels"][label]
        transform = np.asarray(membership["transform"], dtype=np.float64)
        retained = float(membership["retained_mass"])
        contamination = float(membership["contamination"])
        exact = exact_external_attacker_risk(
            reference[label],
            direct_channel,
            (transform,),
            contamination=contamination,
        )
        assignment = np.asarray(
            canonical_worst_assignment(
                reference[label],
                transform,
                direct_channel,
                contamination,
                exact.balanced_accuracy,
            ),
            dtype=np.int64,
        )
        residuals = np.zeros((sources, tokens), dtype=np.float64)
        for source in range(sources):
            correct_outputs = (assignment == source).astype(np.float64)
            residual_token = int(np.argmax(direct_channel @ correct_outputs))
            residuals[source, residual_token] = 1.0
            common = reference[label, source] @ transform
            laws[label, source] = retained * common + contamination * residuals[source]
            reconstructed = retained * common + contamination * residuals[source]
            maximum_membership_error = max(
                maximum_membership_error,
                float(np.max(np.abs(laws[label, source] - reconstructed))),
            )
        details.append(
            {
                "label": label,
                "retained_mass": retained,
                "contamination": contamination,
                "maximizing_assignment": assignment.tolist(),
                "tie_rule": "lexicographically_first_within_1e-12_of_exact_maximum",
                "residual_laws": residuals.tolist(),
                "exact_direct_worst_normalized_advantage": exact.normalized_advantage,
            }
        )
    if np.any(laws < -TOLERANCE) or not np.allclose(
        laws.sum(axis=2), 1.0, atol=TOLERANCE, rtol=0.0
    ):
        raise RuntimeError("constructed admitted law is not a probability tensor")
    return laws, details, maximum_membership_error


def exact_interval(successes: int, trials: int, level: float = 0.95) -> list[float]:
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("invalid binomial count")
    alpha = 1.0 - level
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return [lower, upper]


def strict_release_values(release: dict[str, Any]) -> tuple[float, list[float]]:
    return (
        float(release["certified_worst_conditional_error_upper"]),
        [float(value) for value in release["certified_source_advantage_upper"]],
    )


def analyze_threshold(
    direct_paths: list[Path],
    strict_directory: Path,
    utility_threshold: float,
    *,
    include_jobs: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    key = threshold_key(utility_threshold)
    aggregate: Counter[str] = Counter()
    datasets: dict[str, Counter[str]] = {}
    jobs: list[dict[str, Any]] = []
    structural_radii: list[float] = []
    direct_region_excesses: list[float] = []

    for direct_path in direct_paths:
        direct = json.loads(direct_path.read_text(encoding="utf-8"))
        selection = direct["selection_by_utility_threshold"][key]
        if selection.get("decision") != "deploy":
            continue
        dataset = str(direct["dataset"])
        seed = int(direct["seed"])
        candidate = str(selection["candidate"])
        strict_path = strict_directory / f"{dataset}__seed{seed}.json"
        strict = json.loads(strict_path.read_text(encoding="utf-8"))
        original_path = REPOSITORY / str(direct["original_receipt"])
        original = json.loads(original_path.read_text(encoding="utf-8"))
        direct_row = next(row for row in direct["results"] if row["candidate"] == candidate)
        original_row = next(row for row in original["results"] if row["candidate"] == candidate)
        strict_row = next(row for row in strict["results"] if row["candidate"] == candidate)

        direct_release = direct_row["direct_target"]
        strict_release = strict_row["release_l2"]
        direct_channel = np.asarray(direct_release["release_channel"], dtype=np.float64)
        direct_decoder = tuple(int(value) for value in direct_release["decoder"])
        reference = probability_tensor(original_row["reference_token_counts"])
        admitted, label_details, membership_error = worst_admitted_law(
            reference,
            strict_row["bridge_membership"],
            direct_channel,
        )
        direct_privacy, direct_utility = population_metrics(
            admitted, direct_channel, direct_decoder
        )
        exact_privacy = max(
            float(value["exact_direct_worst_normalized_advantage"])
            for value in label_details
        )
        if abs(direct_privacy - exact_privacy) > TOLERANCE:
            raise RuntimeError("constructed direct stress law misses exact worst privacy")

        direct_certified_privacy = max(
            float(value) for value in direct_release["selection_source_advantages"]
        )
        direct_certified_utility = float(
            direct_release["selection_worst_conditional_error"]
        )
        if direct_certified_privacy > PRIVACY_THRESHOLD + TOLERANCE:
            raise RuntimeError("stored direct deployment fails its selection privacy bound")
        if direct_certified_utility > utility_threshold + TOLERANCE:
            raise RuntimeError("stored direct deployment fails its selection utility bound")

        direct_violation = bool(
            direct_privacy > PRIVACY_THRESHOLD + TOLERANCE
            or direct_utility > utility_threshold + TOLERANCE
        )
        strict_error, strict_privacy_bounds = strict_release_values(strict_release)
        mosaic_deployed = bool(
            max(strict_privacy_bounds) <= PRIVACY_THRESHOLD + TOLERANCE
            and strict_error <= utility_threshold + TOLERANCE
        )
        mosaic_privacy = None
        mosaic_utility = None
        mosaic_violation = False
        if mosaic_deployed:
            mosaic_channel = np.asarray(strict_release["release_channel"], dtype=np.float64)
            mosaic_decoder = tuple(int(value) for value in strict_release["decoder"])
            mosaic_privacy, mosaic_utility = population_metrics(
                admitted, mosaic_channel, mosaic_decoder
            )
            mosaic_violation = bool(
                mosaic_privacy > PRIVACY_THRESHOLD + TOLERANCE
                or mosaic_utility > utility_threshold + TOLERANCE
            )
            if mosaic_violation:
                raise RuntimeError("strict MOSAIC deployment violates its admitted class")

        bridge_probabilities = probability_tensor(original_row["bridge_token_counts"])
        bridge_radii = np.asarray(original_row["bridge_l1_radii"], dtype=np.float64)
        distances = np.abs(admitted - bridge_probabilities).sum(axis=2)
        excess = float(np.max(distances - bridge_radii))
        outside_direct_region = bool(excess > TOLERANCE)
        max_contamination = max(
            float(value) for value in strict_row["bridge_membership"]["contaminations"]
        )
        structural_radii.append(max_contamination)
        direct_region_excesses.append(excess)

        aggregate["direct_deployments"] += 1
        aggregate["direct_contract_violations"] += int(direct_violation)
        aggregate["direct_privacy_violations"] += int(
            direct_privacy > PRIVACY_THRESHOLD + TOLERANCE
        )
        aggregate["direct_utility_violations"] += int(
            direct_utility > utility_threshold + TOLERANCE
        )
        aggregate["stress_laws_outside_direct_region"] += int(outside_direct_region)
        aggregate["mosaic_deployments"] += int(mosaic_deployed)
        aggregate["mosaic_abstentions"] += int(not mosaic_deployed)
        aggregate["mosaic_contract_violations"] += int(mosaic_violation)
        current = datasets.setdefault(dataset, Counter())
        current["direct_deployments"] += 1
        current["direct_contract_violations"] += int(direct_violation)
        current["mosaic_deployments"] += int(mosaic_deployed)
        current["mosaic_abstentions"] += int(not mosaic_deployed)
        current["mosaic_contract_violations"] += int(mosaic_violation)

        if include_jobs:
            jobs.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "candidate": candidate,
                    "direct_receipt": str(direct_path.relative_to(REPOSITORY)),
                    "direct_receipt_sha256": sha256(direct_path),
                    "strict_receipt": str(strict_path.relative_to(REPOSITORY)),
                    "strict_receipt_sha256": sha256(strict_path),
                    "max_membership_reconstruction_error": membership_error,
                    "max_structural_tv_radius": max_contamination,
                    "max_l1_distance_beyond_direct_region": excess,
                    "stress_law_outside_direct_region": outside_direct_region,
                    "direct": {
                        "certified_target_privacy": direct_certified_privacy,
                        "certified_target_utility": direct_certified_utility,
                        "stress_privacy": direct_privacy,
                        "stress_utility": direct_utility,
                        "contract_violation": direct_violation,
                        "release_channel": direct_channel.tolist(),
                        "decoder": list(direct_decoder),
                    },
                    "mosaic": {
                        "decision": "deploy" if mosaic_deployed else "abstain",
                        "certified_privacy": max(strict_privacy_bounds),
                        "certified_utility": strict_error,
                        "stress_privacy": mosaic_privacy,
                        "stress_utility": mosaic_utility,
                        "contract_violation": mosaic_violation,
                    },
                    "labels": label_details,
                    "admitted_target_laws": admitted.tolist(),
                }
            )

    trials = int(aggregate["direct_deployments"])
    violations = int(aggregate["direct_contract_violations"])
    summary: dict[str, Any] = {
        "utility_threshold": utility_threshold,
        **{key: int(value) for key, value in aggregate.items()},
        "direct_violation_rate": 0.0 if trials == 0 else violations / trials,
        "direct_violation_rate_exact_95_interval": (
            None if trials == 0 else exact_interval(violations, trials)
        ),
        "dataset_rows": {
            dataset: {key: int(value) for key, value in counts.items()}
            for dataset, counts in sorted(datasets.items())
        },
        "median_max_structural_tv_radius": (
            None if not structural_radii else float(np.median(structural_radii))
        ),
        "median_l1_distance_beyond_direct_region": (
            None if not direct_region_excesses else float(np.median(direct_region_excesses))
        ),
    }
    return summary, jobs


def atomic_json_dump(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=output.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-dir", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument("--strict-dir", type=Path, default=DEFAULT_STRICT)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    direct_paths = sorted(args.direct_dir.glob("*.json"))
    if len(direct_paths) != 100:
        raise ValueError("admitted-shift stress requires all 100 direct receipts")
    if len(list(args.strict_dir.glob("*.json"))) != 100:
        raise ValueError("admitted-shift stress requires all 100 strict-v2 receipts")
    if not args.spec.is_file():
        raise FileNotFoundError(args.spec)

    frontier = []
    primary_jobs: list[dict[str, Any]] = []
    for threshold in UTILITY_THRESHOLDS:
        summary, jobs = analyze_threshold(
            direct_paths,
            args.strict_dir,
            threshold,
            include_jobs=threshold == PRIMARY_UTILITY_THRESHOLD,
        )
        frontier.append(summary)
        if threshold == PRIMARY_UTILITY_THRESHOLD:
            primary_jobs = jobs
    primary = next(
        row for row in frontier if row["utility_threshold"] == PRIMARY_UTILITY_THRESHOLD
    )
    passed = bool(
        primary["direct_contract_violations"] > 0
        and primary["mosaic_contract_violations"] == 0
        and primary["direct_deployments"]
        == primary["mosaic_deployments"] + primary["mosaic_abstentions"]
    )
    payload = {
        "name": "MOSAIC real-table-anchored admitted-shift stress v1",
        "status": "complete" if passed else "failed",
        "pass": passed,
        "analysis_status": (
            "Post-review deterministic stress specified after a development pilot; "
            "not preregistered and not an untouched diagnostic experiment."
        ),
        "claim_boundary": (
            "Each constructed law is an exact member of a learned MOSAIC bridge "
            "class anchored to stored real token tables. The analysis demonstrates "
            "existence of harmful within-class drift; it does not estimate how often "
            "such drift occurs in deployment."
        ),
        "privacy_advantage_threshold": PRIVACY_THRESHOLD,
        "primary_utility_threshold": PRIMARY_UTILITY_THRESHOLD,
        "specification": str(args.spec.relative_to(REPOSITORY)),
        "specification_sha256": sha256(args.spec),
        "direct_receipt_count": len(direct_paths),
        "strict_receipt_count": len(list(args.strict_dir.glob("*.json"))),
        "primary": primary,
        "frontier": frontier,
        "primary_jobs": primary_jobs,
    }
    atomic_json_dump(payload, args.output)
    print(json.dumps({"output": str(args.output), "pass": passed, "primary": primary}, indent=2))


if __name__ == "__main__":
    main()
