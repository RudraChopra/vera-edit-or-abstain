#!/usr/bin/env python3
"""Replay the locked 1,000-trial synthetic safety cell."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replay_common import clopper_pearson, finish, load_json, parser, require_close, require_equal


def main() -> None:
    args = parser(
        __doc__ or "",
        default_seed=1260000,
        allowed_seeds=range(1260000, 1261000),
    ).parse_args()
    report = load_json("synthetic_safety_trials.json")
    rows = report["trials"]
    trials = len(rows)
    naive = sum(bool(row["plugin_continuum_false_acceptance"]) for row in rows)
    mosaic = sum(bool(row["mosaic_false_acceptance"]) for row in rows)
    require_equal("trials", trials, 1000)
    require_equal("naive violations", naive, 421)
    require_equal("MOSAIC violations", mosaic, 0)
    naive_ci = clopper_pearson(naive, trials)
    mosaic_ci = clopper_pearson(mosaic, trials)
    require_close("naive rate", naive / trials, 0.421)
    require_close("MOSAIC rate", mosaic / trials, 0.0)
    claims = {
        "expected": "naive continuum 421/1000 (42.1%); MOSAIC 0/1000",
        "trials": trials,
        "naive_continuum": {"violations": naive, "rate": naive / trials, "cp95": naive_ci},
        "mosaic": {"violations": mosaic, "rate": mosaic / trials, "cp95": mosaic_ci},
    }
    finish("synthetic_safety", args.seed, claims, args.output)


if __name__ == "__main__":
    main()
