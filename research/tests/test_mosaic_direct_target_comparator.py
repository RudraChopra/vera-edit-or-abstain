from __future__ import annotations

import numpy as np

from run_mosaic_direct_target_comparator import RULE, solve_candidate


def test_direct_target_abstains_for_missing_target_support() -> None:
    counts = np.asarray(
        [
            [[3, 1, 0, 0], [0, 0, 0, 0]],
            [[2, 1, 1, 0], [1, 2, 0, 1]],
        ]
    )
    row = solve_candidate(
        {
            "candidate": "candidate",
            "method": "Identity",
            "strength": "none",
            "provenance": {},
            "bridge_token_counts": counts.tolist(),
            "diagnostic_token_counts": counts.tolist(),
        },
        {
            "fine_token_count": 4,
            "per_candidate_table_delta": 0.01,
            "primary_released_token_count": 2,
            "privacy_advantage_threshold": 0.35,
            "utility_thresholds": [0.40],
        },
    )
    assert row["direct_target_error"] == "missing target source-label stratum"
    assert RULE not in row
