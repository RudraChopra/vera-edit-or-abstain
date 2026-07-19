#!/usr/bin/env python3
"""Replay the matched MOSAIC versus Holm-LTT retention comparison."""

from __future__ import annotations

import sys
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replay_common import finish, load_json, parser, require_close, require_equal


def main() -> None:
    args = parser(
        __doc__ or "",
        default_seed=910000000,
        allowed_seeds=range(910000000, 910001000),
    ).parse_args()
    report = load_json("matched_baseline_trials.json")
    rows = report["trials"]
    mosaic = sum(bool(row["mosaic_deployed"]) for row in rows)
    ltt = sum(bool(row["holm_ltt_deployed"]) for row in rows)
    mosaic_only = sum(row["mosaic_deployed"] and not row["holm_ltt_deployed"] for row in rows)
    ltt_only = sum(row["holm_ltt_deployed"] and not row["mosaic_deployed"] for row in rows)
    p_value = float(binomtest(min(mosaic_only, ltt_only), mosaic_only + ltt_only, 0.5).pvalue)
    require_equal("pairs", len(rows), 1000)
    require_equal("MOSAIC releases", mosaic, 579)
    require_equal("Holm-LTT releases", ltt, 283)
    require_equal("MOSAIC-only pairs", mosaic_only, 312)
    require_equal("Holm-LTT-only pairs", ltt_only, 16)
    require_close("McNemar p-value", p_value, 2.2793313996457506e-72, 1e-84)
    claims = {
        "expected": "MOSAIC 579/1000; Holm-LTT 283/1000; exact McNemar p=2.28e-72",
        "pairs": len(rows),
        "mosaic_releases": mosaic,
        "holm_ltt_releases": ltt,
        "discordant": {"mosaic_only": mosaic_only, "holm_ltt_only": ltt_only},
        "exact_two_sided_mcnemar_p": p_value,
    }
    finish("matched_baselines", args.seed, claims, args.output)


if __name__ == "__main__":
    main()
