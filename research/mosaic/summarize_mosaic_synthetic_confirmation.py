#!/usr/bin/env python3
"""Create paired, claim-scoped summaries from audited MOSAIC confirmation data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable

import numpy as np
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
DEFAULT_PREREG = ROOT / "prereg_mosaic_synthetic_v1.json"
DEFAULT_REPORT = (
    REPOSITORY / "research" / "artifacts" / "mosaic_synthetic_confirmation_v1.json"
)
DEFAULT_AUDIT = (
    REPOSITORY
    / "research"
    / "artifacts"
    / "mosaic_synthetic_confirmation_audit_v1.json"
)
DEFAULT_ALIGNMENT = (
    REPOSITORY
    / "research"
    / "artifacts"
    / "mosaic_synthetic_theory_alignment_audit_v1.json"
)
DEFAULT_OUTPUT = (
    REPOSITORY
    / "research"
    / "artifacts"
    / "mosaic_synthetic_claim_summary_v1.json"
)
BOOTSTRAP_REPETITIONS = 20_000
BOOTSTRAP_SEED = 20_260_718


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


def atomic_json_dump(payload: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=output.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(output)


def paired_binary_contrast(
    index: dict[tuple[str, int, int, str], dict[str, Any]],
    *,
    scenario: str,
    n: int,
    method_a: str,
    method_b: str,
    outcome: Callable[[dict[str, Any]], bool],
    outcome_name: str,
) -> dict[str, object]:
    seeds_a = {
        seed
        for current_scenario, current_n, seed, method in index
        if current_scenario == scenario and current_n == n and method == method_a
    }
    seeds_b = {
        seed
        for current_scenario, current_n, seed, method in index
        if current_scenario == scenario and current_n == n and method == method_b
    }
    if seeds_a != seeds_b or not seeds_a:
        raise AssertionError("paired contrast has missing or unmatched seeds")
    seeds = sorted(seeds_a)
    values_a = np.asarray(
        [outcome(index[(scenario, n, seed, method_a)]) for seed in seeds],
        dtype=np.float64,
    )
    values_b = np.asarray(
        [outcome(index[(scenario, n, seed, method_b)]) for seed in seeds],
        dtype=np.float64,
    )
    differences = values_a - values_b
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_indices = rng.integers(
        0, len(seeds), size=(BOOTSTRAP_REPETITIONS, len(seeds))
    )
    bootstrap_means = differences[bootstrap_indices].mean(axis=1)
    a_only = int(np.sum((values_a == 1.0) & (values_b == 0.0)))
    b_only = int(np.sum((values_a == 0.0) & (values_b == 1.0)))
    discordant = a_only + b_only
    p_value = 1.0 if discordant == 0 else float(
        binomtest(a_only, discordant, p=0.5, alternative="two-sided").pvalue
    )
    return {
        "scenario": scenario,
        "sample_size_per_stratum": n,
        "outcome": outcome_name,
        "method_a": method_a,
        "method_b": method_b,
        "replicates": len(seeds),
        "method_a_rate": float(values_a.mean()),
        "method_b_rate": float(values_b.mean()),
        "paired_rate_difference_a_minus_b": float(differences.mean()),
        "paired_bootstrap_95_lower": float(np.quantile(bootstrap_means, 0.025)),
        "paired_bootstrap_95_upper": float(np.quantile(bootstrap_means, 0.975)),
        "a_only_discordant": a_only,
        "b_only_discordant": b_only,
        "exact_mcnemar_two_sided_p": p_value,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prereg = load_json(args.prereg)
    report = load_json(args.report)
    audit = load_json(args.audit)
    alignment = load_json(args.alignment)
    prereg_hash = sha256(args.prereg)
    report_hash = sha256(args.report)
    if report["preregistration_sha256"] != prereg_hash:
        raise AssertionError("report is not tied to this preregistration")
    if audit["report_sha256"] != report_hash or audit["pass"] is not True:
        raise AssertionError("report lacks a passing independent replay")
    if alignment["confirmation_sha256"] != report_hash or alignment["pass"] is not True:
        raise AssertionError("report lacks a passing theory-alignment audit")
    if report["pass_conditions"]["all_pass"] is not True:
        raise AssertionError("locked confirmation gate did not pass")

    index = {
        (
            str(row["scenario"]),
            int(row["sample_size_per_stratum"]),
            int(row["seed"]),
            str(row["method"]),
        ): row
        for row in report["replicate_results"]
    }
    if len(index) != len(report["replicate_results"]):
        raise AssertionError("confirmation contains duplicate method receipts")

    killer = prereg["pass_conditions"]["killer_contrast"]
    retention = prereg["pass_conditions"]["retention"]
    contrasts = [
        paired_binary_contrast(
            index,
            scenario=str(killer["scenario"]),
            n=int(killer["sample_size_per_stratum"]),
            method_a="plugin_continuum",
            method_b="mosaic",
            outcome=lambda row: bool(row["false_acceptance"]),
            outcome_name="false_acceptance",
        )
    ]
    for comparator in retention["comparators"]:
        contrasts.append(
            paired_binary_contrast(
                index,
                scenario=str(retention["scenario"]),
                n=int(retention["sample_size_per_stratum"]),
                method_a="mosaic",
                method_b=str(comparator),
                outcome=lambda row: bool(row["deployed"] and row["exact_safe"]),
                outcome_name="safe_deployment",
            )
        )

    payload: dict[str, object] = {
        "name": "MOSAIC audited paired claim summary v1",
        "status": "derived_only_from_passing_locked_receipts",
        "preregistration_sha256": prereg_hash,
        "confirmation_sha256": report_hash,
        "confirmation_audit_sha256": sha256(args.audit),
        "theory_alignment_audit_sha256": sha256(args.alignment),
        "contrasts": contrasts,
        "scope": (
            "The four paired contrasts were specified in code before the locked "
            "confirmation was read. Bootstrap intervals are descriptive; exact "
            "McNemar p-values use only paired discordances."
        ),
    }
    atomic_json_dump(payload, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
