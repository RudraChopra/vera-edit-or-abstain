#!/usr/bin/env python3
"""Replay bridge power and misspecification rejection rates."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replay_common import finish, load_json, parser, require_equal


def main() -> None:
    args = parser(
        __doc__ or "",
        default_seed=930000000,
        allowed_seeds=range(930000000, 930001000),
    ).parse_args()
    report = load_json("bridge_power.json")
    valid = sorted(
        (row["sample_size_per_stratum"], row["acceptances"])
        for row in report["cells"]
        if row["scenario"] == "compatible_common_transform"
    )
    require_equal("valid bridge power", valid, [(500, 0), (1000, 310), (2000, 1000), (5000, 1000)])
    invalid = [row for row in report["cells"] if not row["model_valid"]]
    require_equal("invalid mechanism acceptances", sum(row["acceptances"] for row in invalid), 0)
    require_equal("invalid cells", len(invalid), 8)
    claims = {
        "expected": "valid acceptance 0%,31%,100%,100%; both invalid mechanisms rejected",
        "valid_acceptance": [
            {"sample_size_per_stratum": n, "acceptances": accepted, "trials": 1000}
            for n, accepted in valid
        ],
        "invalid_cells": len(invalid),
        "invalid_acceptances": 0,
    }
    finish("bridge_power", args.seed, claims, args.output)


if __name__ == "__main__":
    main()
