#!/usr/bin/env python3
"""Replay the ACSIncome California-to-Texas bridge result."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replay_common import finish, load_json, parser, require_equal


def main() -> None:
    args = parser(__doc__ or "", default_seed=1305, allowed_seeds=range(1305, 1310)).parse_args()
    report = load_json("acs_ca_tx.json")
    diagnosis = load_json("acs_primary_infeasibility.json")
    selections = report["selection_by_utility_threshold"]
    require_equal(".40 releases", selections["0.40"]["deployments"], 0)
    require_equal(".40 abstentions", selections["0.40"]["abstentions"], 5)
    for threshold in ("0.45", "0.49"):
        require_equal(f"{threshold} releases", selections[threshold]["deployments"], 5)
        require_equal(f"{threshold} violations", selections[threshold]["diagnostic_contract_violations"], 0)
        require_equal(f"{threshold} identity no-ops", selections[threshold]["selected_method_counts"].get("Identity"), 2)
    errors = sorted(float(row["certified_error"]) for row in diagnosis["rows"])
    require_equal("seed rows", len(errors), 5)
    if errors[0] < 0.4023 - 5e-5 or errors[-1] > 0.4145 + 5e-5:
        raise AssertionError(f"best .40 error range outside expected [.4023,.4145]: {errors}")
    claims = {
        "expected": "abstain 5/5 at .40; release 5/5 at .45 and .49; two identity no-ops",
        "selection_by_utility_threshold": selections,
        "best_primary_error_range": [errors[0], errors[-1]],
    }
    finish("acs_ca_tx", args.seed, claims, args.output)


if __name__ == "__main__":
    main()
