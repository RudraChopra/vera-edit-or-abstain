#!/usr/bin/env python3
"""Independent replay of the transform-exact MOSAIC confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
from scipy.stats import beta

from mosaic_exact import exact_external_attacker_risk, exact_external_utility_risk
from mosaic_invariant import (
    adaptive_pre_release_attacker_certificate,
    pre_release_utility_certificate,
)
from mosaic_transform_exact import (
    transform_exact_attacker_confidence_bound,
    transform_exact_utility_confidence_bound,
)


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
DEFAULT_PREREG = ROOT / "prereg_mosaic_transform_exact_v1.json"
DEFAULT_SIDECAR = ROOT / "prereg_mosaic_transform_exact_v1.sha256"
DEFAULT_REPORT = REPOSITORY / "research/artifacts/mosaic_transform_exact_confirmation_v1.json"
DEFAULT_OUTPUT = REPOSITORY / "research/artifacts/mosaic_transform_exact_confirmation_audit_v1.json"
TOLERANCE = 2e-7


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


def close(first: float, second: float) -> bool:
    return abs(float(first) - float(second)) <= TOLERANCE


def cp_interval(successes: int, trials: int) -> tuple[float, float]:
    lower = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(
        beta.ppf(0.975, successes + 1, trials - successes)
    )
    return lower, upper


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
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite audit output")
    prereg = load_json(args.prereg)
    expected_prereg_hash = args.sidecar.read_text(encoding="utf-8").strip().split()[0]
    if sha256(args.prereg) != expected_prereg_hash:
        raise RuntimeError("preregistration sidecar mismatch")
    report = load_json(args.report)
    if report["preregistration_sha256"] != expected_prereg_hash:
        raise RuntimeError("report is not linked to the locked preregistration")
    for relative, expected_hash in prereg["code_sha256"].items():
        if sha256(REPOSITORY / relative) != expected_hash:
            raise RuntimeError(f"post-run code hash mismatch: {relative}")

    population = np.asarray(prereg["population"]["laws"], dtype=np.float64)
    transforms = tuple(
        np.asarray(transform, dtype=np.float64)
        for transform in prereg["population"]["common_transform_extremes"]
    )
    libraries = tuple(transforms for _ in range(population.shape[0]))
    scenario_by_name = {str(item["name"]): item for item in prereg["scenarios"]}
    methods = tuple(str(method) for method in prereg["methods"])
    replicates = int(prereg["replicates_per_cell"])
    mismatches: list[str] = []
    indexed: dict[tuple[str, int, int, str], dict[str, Any]] = {}

    for row_index, row in enumerate(report["replicate_results"]):
        scenario_name = str(row["scenario"])
        n = int(row["sample_size_per_stratum"])
        seed = int(row["seed"])
        method = str(row["method"])
        key = (scenario_name, n, seed, method)
        if key in indexed:
            mismatches.append(f"duplicate row {key}")
            continue
        indexed[key] = row
        scenario = scenario_by_name[scenario_name]
        eta = float(scenario["contamination"])
        privacy_threshold = float(scenario["privacy_threshold"])
        utility_threshold = float(scenario["utility_threshold"])
        empirical = np.asarray(row["empirical_table"], dtype=np.float64)
        radii = np.asarray(row["l1_radii"], dtype=np.float64)
        channel = np.asarray(row["release_channel"], dtype=np.float64)
        decoder = tuple(int(value) for value in row["decoder"])
        if empirical.shape != population.shape or radii.shape != population.shape[:2]:
            mismatches.append(f"shape mismatch row {row_index}")
            continue
        expected_channel_shape = (
            population.shape[2],
            int(prereg["population"]["released_token_count"]),
        )
        if channel.shape != expected_channel_shape:
            mismatches.append(f"channel shape mismatch row {row_index}")
            continue
        if np.any(channel < -1e-10) or not np.allclose(channel.sum(axis=1), 1.0, atol=1e-9):
            mismatches.append(f"invalid channel row {row_index}")
            continue

        event = bool(np.all(np.abs(empirical - population).sum(axis=2) <= radii + 1e-12))
        if event != bool(row["confidence_event"]):
            mismatches.append(f"confidence event row {row_index}")

        privacy_values = []
        utility_values = []
        for label in range(population.shape[0]):
            if method == "transform_exact":
                privacy_certificate = transform_exact_attacker_confidence_bound(
                    empirical[label],
                    channel,
                    l1_radii=radii[label],
                    common_fine_token_channels=libraries[label],
                    contamination=eta,
                )
            elif method == "capacity_transfer":
                privacy_certificate = adaptive_pre_release_attacker_certificate(
                    empirical[label],
                    channel,
                    l1_radii=radii[label],
                    common_fine_token_channels=libraries[label],
                    contamination=eta,
                )
            else:
                mismatches.append(f"unknown method row {row_index}")
                continue
            privacy_values.append(float(privacy_certificate.normalized_advantage))
            for source in range(population.shape[1]):
                if method == "transform_exact":
                    utility_certificate = transform_exact_utility_confidence_bound(
                        empirical[label, source],
                        channel,
                        decoder,
                        true_label=label,
                        l1_radius=float(radii[label, source]),
                        common_fine_token_channels=libraries[label],
                        contamination=eta,
                    )
                else:
                    utility_certificate = pre_release_utility_certificate(
                        empirical[label, source],
                        channel,
                        decoder,
                        true_label=label,
                        l1_radius=float(radii[label, source]),
                        common_fine_token_channels=libraries[label],
                        contamination=eta,
                    )
                utility_values.append(float(utility_certificate.error_probability))
        if len(privacy_values) != population.shape[0]:
            continue
        if any(
            not close(observed, expected)
            for observed, expected in zip(row["certified_privacy_advantages"], privacy_values)
        ):
            mismatches.append(f"privacy certificate row {row_index}")
        certified_error = max(utility_values)
        if not close(row["certified_worst_conditional_error"], certified_error):
            mismatches.append(f"utility certificate row {row_index}")
        privacy_pass = all(value <= privacy_threshold + 1e-10 for value in privacy_values)
        deployed = bool(privacy_pass and certified_error <= utility_threshold + 1e-10)
        if deployed != bool(row["deployed"]):
            mismatches.append(f"decision row {row_index}")

        external_privacy = [
            exact_external_attacker_risk(
                population[label],
                channel,
                libraries[label],
                contamination=eta,
            ).normalized_advantage
            for label in range(population.shape[0])
        ]
        external_utility = max(
            exact_external_utility_risk(
                population[label, source],
                channel,
                decoder,
                true_label=label,
                common_fine_token_channels=libraries[label],
                contamination=eta,
            ).error_probability
            for label in range(population.shape[0])
            for source in range(population.shape[1])
        )
        safe = all(value <= privacy_threshold + 1e-9 for value in external_privacy) and (
            external_utility <= utility_threshold + 1e-9
        )
        false_acceptance = bool(deployed and not safe)
        checks = (
            (row["exact_worst_privacy_advantage"], max(external_privacy), "external privacy"),
            (row["exact_worst_conditional_error"], external_utility, "external utility"),
        )
        for observed, expected, label in checks:
            if not close(observed, expected):
                mismatches.append(f"{label} row {row_index}")
        if safe != bool(row["exact_safe"]):
            mismatches.append(f"safe label row {row_index}")
        if false_acceptance != bool(row["false_acceptance"]):
            mismatches.append(f"false acceptance row {row_index}")
        if bool(false_acceptance and event) != bool(row["failure_on_confidence_event"]):
            mismatches.append(f"event failure row {row_index}")
        if float(row["solver_gap"]) > 1e-10 or float(row["max_constraint_violation"]) > TOLERANCE:
            mismatches.append(f"solver certificate row {row_index}")

    expected_keys = set()
    for scenario_index, scenario in enumerate(prereg["scenarios"]):
        for n_value in scenario["sample_sizes_per_stratum"]:
            n = int(n_value)
            for replicate in range(replicates):
                seed = (
                    int(prereg["seed_base"])
                    + scenario_index * 10_000_000
                    + n * 10_000
                    + replicate
                )
                for method in methods:
                    expected_keys.add((str(scenario["name"]), n, seed, method))
    if set(indexed) != expected_keys:
        mismatches.append("registered row grid mismatch")

    paired_dominance = True
    for scenario_name, n, seed, method in expected_keys:
        if method != "transform_exact" or (scenario_name, n, seed, method) not in indexed:
            continue
        exact = indexed[(scenario_name, n, seed, "transform_exact")]
        transfer = indexed[(scenario_name, n, seed, "capacity_transfer")]
        if float(exact["certified_worst_conditional_error"]) > (
            float(transfer["certified_worst_conditional_error"]) + TOLERANCE
        ):
            paired_dominance = False
            mismatches.append(f"paired dominance {(scenario_name, n, seed)}")

    aggregates: dict[tuple[str, int, str], dict[str, int]] = defaultdict(
        lambda: {
            "replicates": 0,
            "deployments": 0,
            "false_acceptances": 0,
            "safe_deployments": 0,
            "confidence_event_count": 0,
            "failures_on_confidence_event": 0,
        }
    )
    for row in indexed.values():
        key = (str(row["scenario"]), int(row["sample_size_per_stratum"]), str(row["method"]))
        counts = aggregates[key]
        counts["replicates"] += 1
        counts["deployments"] += int(bool(row["deployed"]))
        counts["false_acceptances"] += int(bool(row["false_acceptance"]))
        counts["safe_deployments"] += int(bool(row["deployed"]) and bool(row["exact_safe"]))
        counts["confidence_event_count"] += int(bool(row["confidence_event"]))
        counts["failures_on_confidence_event"] += int(bool(row["failure_on_confidence_event"]))
    reported_cells = {
        (str(cell["scenario"]), int(cell["sample_size_per_stratum"]), str(cell["method"])): cell
        for cell in report["cells"]
    }
    for key, counts in aggregates.items():
        cell = reported_cells.get(key)
        if cell is None:
            mismatches.append(f"missing aggregate {key}")
            continue
        trials = counts["replicates"]
        for field, value in counts.items():
            if int(cell[field]) != value:
                mismatches.append(f"aggregate {field} {key}")
        for field, prefix in (
            ("deployments", "deployment"),
            ("false_acceptances", "false_acceptance"),
            ("safe_deployments", "safe_deployment"),
        ):
            rate = counts[field] / trials
            lower, upper = cp_interval(counts[field], trials)
            interval_matches = (
                close(cell[f"{prefix}_rate"], rate)
                and close(cell[f"{prefix}_cp95_lower"], lower)
                and close(cell[f"{prefix}_cp95_upper"], upper)
            )
            if not interval_matches:
                mismatches.append(f"aggregate interval {prefix} {key}")

    coverage = all(
        counts["false_acceptances"] / counts["replicates"] <= float(prereg["delta"]) + 1e-12
        and counts["failures_on_confidence_event"] == 0
        for counts in aggregates.values()
    )
    def safe_rate(n: int, method: str) -> float:
        counts = aggregates[("retention_and_exactness_value", n, method)]
        return counts["safe_deployments"] / counts["replicates"]

    n125_exact = safe_rate(125, "transform_exact")
    n125_transfer = safe_rate(125, "capacity_transfer")
    n250_exact = safe_rate(250, "transform_exact")
    n250_transfer = safe_rate(250, "capacity_transfer")
    gate125 = prereg["pass_conditions"]["retention_n125"]
    gate250 = prereg["pass_conditions"]["retention_n250"]
    retention125 = n125_exact >= float(
        gate125["minimum_transform_exact_safe_deployment_rate"]
    ) and n125_exact - n125_transfer >= float(
        gate125["minimum_safe_deployment_margin_over_capacity_transfer"]
    )
    retention250 = n250_exact >= float(
        gate250["minimum_transform_exact_safe_deployment_rate"]
    ) and n250_exact - n250_transfer >= float(
        gate250["minimum_safe_deployment_margin_over_capacity_transfer"]
    )
    passed = not mismatches and coverage and paired_dominance and retention125 and retention250
    output = {
        "name": str(prereg["audit_name"]),
        "pass": passed,
        "report_sha256": sha256(args.report),
        "preregistration_sha256": expected_prereg_hash,
        "audited_rows": len(indexed),
        "audited_privacy_certificates": len(indexed) * population.shape[0],
        "audited_utility_certificates": len(indexed) * population.shape[0] * population.shape[1],
        "audited_external_risks": len(indexed) * 2,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:100],
        "coverage": coverage,
        "pointwise_dominance": paired_dominance,
        "retention_n125": retention125,
        "retention_n250": retention250,
    }
    atomic_json(output, args.output)
    print(json.dumps(output, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
