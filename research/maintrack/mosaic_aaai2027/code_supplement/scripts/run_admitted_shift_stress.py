#!/usr/bin/env python3
"""Replay the admitted-shift stress experiment and threshold frontier."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replay_common import finish, load_json, parser, require_equal


def main() -> None:
    args = parser(__doc__ or "", default_seed=1200, allowed_seeds=range(1200, 1220)).parse_args()
    report = load_json("admitted_shift_stress.json")
    primary = report["primary"]
    primary_observed = {
        "direct_deployments": primary["direct_deployments"],
        "direct_contract_violations": primary["direct_contract_violations"],
        "mosaic_deployments": primary["mosaic_deployments"],
        "mosaic_abstentions": primary["mosaic_abstentions"],
        "mosaic_contract_violations": primary["mosaic_contract_violations"],
    }
    require_equal(
        "primary admitted-shift cell",
        primary_observed,
        {
            "direct_deployments": 36,
            "direct_contract_violations": 16,
            "mosaic_deployments": 20,
            "mosaic_abstentions": 16,
            "mosaic_contract_violations": 0,
        },
    )
    frontier = [
        (float(row["utility_threshold"]), row["direct_contract_violations"], row["direct_deployments"], row["mosaic_contract_violations"])
        for row in report["frontier"]
    ]
    expected_frontier = [
        (0.30, 18, 20, 0),
        (0.35, 4, 20, 0),
        (0.40, 16, 36, 0),
        (0.45, 20, 40, 0),
        (0.49, 23, 43, 0),
    ]
    require_equal("threshold frontier", frontier, expected_frontier)
    claims = {
        "expected": "direct violates 16/36; MOSAIC releases 20 safe and abstains on 16",
        "primary": primary_observed,
        "frontier": [
            {"threshold": threshold, "direct_violations": violations, "direct_deployments": deployments, "mosaic_violations": mosaic_violations}
            for threshold, violations, deployments, mosaic_violations in frontier
        ],
    }
    finish("admitted_shift_stress", args.seed, claims, args.output)


if __name__ == "__main__":
    main()
