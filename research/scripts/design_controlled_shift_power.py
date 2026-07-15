"""Prospective seed-count and endpoint power analysis from design-only seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta, binomtest, multinomial


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESIGN = ROOT / "artifacts" / "vera_controlled_shift_design.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "vera_controlled_shift_power.json"
DATASETS = ("Bios", "CivilComments-WILDS", "GaitPDB", "Waterbirds")


def cp_upper(successes: int, n: int, alpha: float) -> float:
    if successes == n:
        return 1.0
    return float(beta.ppf(1.0 - alpha, successes + 1, n - successes))


def exact_sign_power(n: int, probabilities: tuple[float, float, float]) -> float:
    power = 0.0
    for favorable in range(n + 1):
        for adverse in range(n - favorable + 1):
            ties = n - favorable - adverse
            if favorable <= adverse or favorable + adverse == 0:
                continue
            p_value = binomtest(
                favorable,
                favorable + adverse,
                p=0.5,
                alternative="two-sided",
            ).pvalue
            if p_value < 0.05:
                power += float(
                    multinomial.pmf(
                        [favorable, adverse, ties],
                        n=n,
                        p=probabilities,
                    )
                )
    return power


def locked_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in report["rows"]
        if row["requested_gamma"] == 1.1
        and row["total_budget"] == 4000
        and row["allocation"] == "targeted_floor_0.15"
    ]


def cluster_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (int(row["seed"]), str(row["dataset"]), str(row["rule"])): row
        for row in rows
    }
    seeds = sorted({int(row["seed"]) for row in rows})
    output = []
    for seed in seeds:
        point_count = sum(
            bool(by_key[(seed, dataset, "validation_point_selection")]["violation"])
            for dataset in DATASETS
        )
        vector_count = sum(
            bool(by_key[(seed, dataset, "vera_vector_envelope")]["violation"])
            for dataset in DATASETS
        )
        output.append({
            "seed": seed,
            "point_violation_count": point_count,
            "vector_violation_count": vector_count,
            "favors_vector": point_count > vector_count,
            "favors_point": vector_count > point_count,
            "tie": point_count == vector_count,
        })
    return output


def retention_intervals(
    rows: list[dict[str, Any]], *, bootstrap_replicates: int
) -> dict[str, Any]:
    by_key = {
        (int(row["seed"]), str(row["dataset"]), str(row["rule"])): row
        for row in rows
    }
    seeds = sorted({int(row["seed"]) for row in rows})

    def statistic(sample: np.ndarray | list[int]) -> tuple[float, float, float]:
        opportunities = vector = common = 0
        for seed in sample:
            for dataset in DATASETS:
                opportunities += bool(
                    by_key[(int(seed), dataset, "external_oracle")]["deployed"]
                )
                vector += bool(
                    by_key[(int(seed), dataset, "vera_vector_envelope")]["safe"]
                )
                common += bool(
                    by_key[(int(seed), dataset, "vera_common_radius")]["safe"]
                )
        vector_retention = vector / opportunities
        common_retention = common / opportunities
        ratio = float("inf") if common == 0 and vector > 0 else vector / max(common, 1)
        return vector_retention, common_retention, ratio

    rng = np.random.default_rng(2_027_071_504)
    bootstrap = np.asarray([
        statistic(rng.choice(seeds, size=len(seeds), replace=True))
        for _ in range(bootstrap_replicates)
    ])
    point = statistic(seeds)
    return {
        "bootstrap_unit": "seed cluster across all four supported datasets",
        "bootstrap_replicates": bootstrap_replicates,
        "vector_safe_retention": point[0],
        "vector_safe_retention_ci95": [
            float(np.quantile(bootstrap[:, 0], 0.025)),
            float(np.quantile(bootstrap[:, 0], 0.975)),
        ],
        "common_safe_retention": point[1],
        "common_safe_retention_ci95": [
            float(np.quantile(bootstrap[:, 1], 0.025)),
            float(np.quantile(bootstrap[:, 1], 0.975)),
        ],
        "vector_to_common_ratio": point[2],
        "vector_to_common_ratio_ci95": [
            float(np.quantile(bootstrap[:, 2], 0.025)),
            float(np.quantile(bootstrap[:, 2], 0.975)),
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--planned-seeds", type=int, default=64)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = json.loads(args.design.read_text(encoding="utf-8"))
    rows = locked_rows(report)
    clusters = cluster_records(rows)
    favorable = sum(record["favors_vector"] for record in clusters)
    adverse = sum(record["favors_point"] for record in clusters)
    ties = sum(record["tie"] for record in clusters)
    n_design = len(clusters)
    plugin = (favorable / n_design, adverse / n_design, ties / n_design)
    jeffreys = (
        (favorable + 0.5) / (n_design + 1.5),
        (adverse + 0.5) / (n_design + 1.5),
        (ties + 0.5) / (n_design + 1.5),
    )
    minimum = 1
    while cp_upper(0, minimum, 0.05) >= 0.05:
        minimum += 1
    planned = int(args.planned_seeds)
    if planned < minimum or planned % len(DATASETS) != 0:
        raise ValueError(
            "planned seeds must meet the zero-event safety bound and balance datasets"
        )
    payload = {
        "name": "VERA controlled-shift prospective power analysis",
        "analysis_tier": "exploratory design only; no fresh seeds read",
        "design_seeds": sorted(record["seed"] for record in clusters),
        "fresh_seed_block": list(range(45, 45 + planned)),
        "planned_seed_count": planned,
        "fresh_block_disjoint_from_design": not bool(
            set(range(45, 45 + planned))
            & {record["seed"] for record in clusters}
        ),
        "safety_endpoint": {
            "unit": "one prespecified rotating dataset decision per seed",
            "rotation": list(DATASETS),
            "alpha": 0.05,
            "target_upper_bound": 0.05,
            "minimum_zero_event_seed_count": minimum,
            "planned_zero_event_cp95_upper": cp_upper(0, planned, 0.05),
            "reason_for_64": (
                "The minimum is 59; 64 gives 16 sentinel decisions per dataset "
                "and a safety margin above the exact minimum."
            ),
        },
        "paired_primary_endpoint": {
            "unit": "seed cluster",
            "design_favorable_clusters": favorable,
            "design_adverse_clusters": adverse,
            "design_ties": ties,
            "test": "two-sided exact sign test on non-tied seed-level violation counts",
            "plugin_category_probabilities": plugin,
            "jeffreys_smoothed_category_probabilities": jeffreys,
            "plugin_power_at_planned_n": exact_sign_power(planned, plugin),
            "jeffreys_smoothed_power_at_planned_n": exact_sign_power(
                planned, jeffreys
            ),
        },
        "usefulness_endpoint": retention_intervals(
            rows, bootstrap_replicates=args.bootstrap_replicates
        ),
        "locked_design_candidate": {
            "requested_gamma": 1.1,
            "total_contract_observation_budget": 4000,
            "allocation": "targeted_floor_0.15",
            "candidate_count": 12,
            "attacker_count": 4,
        },
        "success_wording": {
            "paired_reduction": "positive effect and exact sign-test p < 0.05",
            "safety": "one-sided 95% upper bound at or below 0.05",
            "usefulness": "safe-retention point estimate at least 0.20",
            "vector_advantage": "vector/common retention ratio at least 2.0",
        },
        "claim_boundary": (
            "The power calculation chooses sample size and endpoints only. "
            "Seeds 45 onward must remain unread until protocol and hash commits are pushed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "planned_seed_count": planned,
        "minimum_zero_event_seed_count": minimum,
        "planned_zero_event_cp95_upper": payload["safety_endpoint"]["planned_zero_event_cp95_upper"],
        "jeffreys_smoothed_power": payload["paired_primary_endpoint"]["jeffreys_smoothed_power_at_planned_n"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
