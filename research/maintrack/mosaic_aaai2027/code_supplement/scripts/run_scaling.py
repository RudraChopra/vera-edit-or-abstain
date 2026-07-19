#!/usr/bin/env python3
"""Replay the token/source alphabet scaling study."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replay_common import finish, load_json, parser, require_close, require_equal


def main() -> None:
    args = parser(__doc__ or "", default_seed=4100, allowed_seeds=range(4100, 4105)).parse_args()
    report = load_json("scaling_study.json")
    rows = report["rows"]
    require_equal("token counts", sorted({row["token_count"] for row in rows}), [4, 8, 16, 32, 64])
    require_equal("source counts", sorted({row["source_count"] for row in rows}), [2, 3, 4])
    require_equal("optimizations", len(rows), 75)
    require_equal(".40 certifications", sum(bool(row["deploy"]) for row in rows), 75)
    summary = report["summary"]
    k4g3 = next(row for row in summary if row["token_count"] == 4 and row["source_count"] == 3)
    k64g4 = next(row for row in summary if row["token_count"] == 64 and row["source_count"] == 4)
    require_equal("K=4 n-for-radius", k4g3["required_n_for_k4_radius"], 2000)
    require_equal("K=64 n-for-radius", k64g4["required_n_for_k4_radius"], 12817)
    require_close("K=64,G=4 median solve", k64g4["wall_clock_seconds_median"], 0.3588683750003838)
    require_equal("active assignments", int(k64g4["active_attacker_assignments_median"]), 4)
    claims = {
        "expected": "75/75 certify at .40; K=64,G=4 median 0.359s; n-for-radius 2000 to 12817",
        "optimizations": len(rows),
        "certifications_at_040": sum(bool(row["deploy"]) for row in rows),
        "summary": summary,
    }
    finish("scaling", args.seed, claims, args.output)


if __name__ == "__main__":
    main()
