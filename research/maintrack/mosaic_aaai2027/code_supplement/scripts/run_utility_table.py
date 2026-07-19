#!/usr/bin/env python3
"""Replay the released-interface utility table."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replay_common import finish, load_json, parser, require_close, require_equal


def metric(report: dict, dataset: str, name: str) -> dict:
    return report["datasets"][dataset]["metrics"][name]


def main() -> None:
    args = parser(__doc__ or "", default_seed=1200, allowed_seeds=range(1200, 1220)).parse_args()
    report = load_json("release_utility_table.json")
    bios = report["datasets"]["BiasBios-Clinical"]
    require_equal("BiasBios releases", bios["releases"], 20)
    checks = {
        "released": (metric(report, "BiasBios-Clinical", "Released interface")["mean"], 0.8627794090552754),
        "four_bin": (metric(report, "BiasBios-Clinical", "4-bin tokenizer")["mean"], 0.8726750000000001),
        "full_edited": (metric(report, "BiasBios-Clinical", "Full edited features")["mean"], 0.9013375),
        "unedited": (metric(report, "BiasBios-Clinical", "Unedited features")["mean"], 0.9032625),
    }
    for label, (observed, expected) in checks.items():
        require_close(label, observed, expected)
    released_ci = metric(report, "BiasBios-Clinical", "Released interface")["mean_t95_interval"]
    require_close("released CI lower", released_ci[0], 0.8571932801602343)
    require_close("released CI upper", released_ci[1], 0.8683655379503165)
    claims = {
        "expected": "BiasBios released .863 [.857,.868], four-bin .873, full-edited .901, unedited .903; Waterbirds included",
        "datasets": report["datasets"],
    }
    finish("utility_table", args.seed, claims, args.output)


if __name__ == "__main__":
    main()
