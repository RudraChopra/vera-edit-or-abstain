#!/usr/bin/env python3
"""Replay the five-rule 100-job real-feature comparison."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replay_common import finish, load_json, parser, require_equal


EXPECTED = {
    "strict_mosaic": (20, 0),
    "direct_target": (36, 0),
    "bridge_plugin": (47, 7),
    "validation_plugin": (80, 18),
    "unconditional": (100, 38),
}


def main() -> None:
    args = parser(__doc__ or "", default_seed=1200, allowed_seeds=range(1200, 1220)).parse_args()
    report = load_json("real_100jobs.json")
    observed = {}
    for rule, rows in report["jobs_by_rule"].items():
        deployments = sum(row["decision"] == "deploy" for row in rows)
        violations = sum(bool(row["false_acceptance"]) for row in rows)
        observed[rule] = (deployments, violations)
    require_equal("100-job rule totals", observed, EXPECTED)
    claims = {
        "expected": "strict 20/100 (0); direct 36/100 (0); bridge plug-in 47 (7); validation 80 (18); unconditional 100 (38)",
        "rules": {
            rule: {"deployments": value[0], "contract_violations": value[1]}
            for rule, value in observed.items()
        },
    }
    finish("real_100jobs", args.seed, claims, args.output)


if __name__ == "__main__":
    main()
