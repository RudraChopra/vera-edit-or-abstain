#!/usr/bin/env python3
"""Replay the 60-job natural multistate ACS confirmation."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replay_common import finish, parser, require_equal


EXPECTED_FRONTIER = {
    "4": {
        "mosaic": {"0.30": (0, 0), "0.35": (12, 0), "0.40": (15, 0), "0.45": (46, 0), "0.49": (57, 0)},
        "direct": {"0.30": (20, 0), "0.35": (40, 1), "0.40": (47, 0), "0.45": (60, 0), "0.49": (60, 0)},
    },
    "8": {
        "mosaic": {"0.30": (0, 0), "0.35": (0, 0), "0.40": (0, 0), "0.45": (0, 0), "0.49": (1, 0)},
        "direct": {"0.30": (22, 0), "0.35": (40, 0), "0.40": (47, 0), "0.45": (60, 0), "0.49": (60, 0)},
    },
}


def main() -> None:
    args = parser(__doc__ or "", default_seed=1400, allowed_seeds=range(1400, 1405)).parse_args()
    summary_path = ROOT / "artifacts/natural_shift/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require_equal("receipt count", summary["receipt_count"], 60)
    require_equal("optimization failures", summary["optimization_failures"], {})
    for alphabet, rules in EXPECTED_FRONTIER.items():
        for rule, thresholds in rules.items():
            for threshold, (releases, violations) in thresholds.items():
                cell = summary["cells"][alphabet][rule][threshold]
                require_equal(f"K={alphabet} {rule} {threshold} releases", cell["deployments"], releases)
                require_equal(f"K={alphabet} {rule} {threshold} violations", cell["false_acceptances"], violations)
    primary = summary["cells"]["4"]["mosaic"]["0.40"]
    require_equal("primary operational batches", primary["operational_draws"], 1500)
    require_equal("primary operational violations", primary["operational_contract_violations"], 0)
    task_releases = {
        task: cell["mosaic"]["deployments"]
        for task, cell in summary["primary_breakdowns"]["4"]["task"].items()
    }
    state_releases = {
        state: cell["mosaic"]["deployments"]
        for state, cell in summary["primary_breakdowns"]["4"]["state"].items()
    }
    require_equal("primary task releases", task_releases, {"employment": 13, "income": 2, "public_coverage": 0})
    require_equal("primary state releases", state_releases, {"FL": 4, "IL": 2, "NY": 5, "WA": 4})
    claims = {
        "expected": "K=4 MOSAIC 15/60 releases, 0/15 held-out violations, 0/1,500 operational violations",
        "frontier": EXPECTED_FRONTIER,
        "primary_task_releases": task_releases,
        "primary_state_releases": state_releases,
    }
    finish("acs_multistate", args.seed, claims, args.output)


if __name__ == "__main__":
    main()
