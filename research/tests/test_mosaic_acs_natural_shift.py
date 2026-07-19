from __future__ import annotations

import numpy as np

from run_mosaic_acs_natural_shift import (
    expected_protocol,
    operational_replay,
    puma_bridge_diagnostic_split,
)


def test_puma_split_is_disjoint_deterministic_and_environment_level() -> None:
    environments = np.repeat(np.arange(12), 5)
    indices = np.arange(len(environments))
    first = puma_bridge_diagnostic_split(indices, environments, seed=11)
    second = puma_bridge_diagnostic_split(indices, environments, seed=11)
    assert all(np.array_equal(a, b) for a, b in zip(first[:2], second[:2], strict=True))
    assert set(first[2]).isdisjoint(first[3])
    assert set(environments[first[0]]) == set(first[2])
    assert set(environments[first[1]]) == set(first[3])


def test_operational_replay_constant_private_channel() -> None:
    counts = np.asarray(
        [
            [[50, 50], [50, 50]],
            [[50, 50], [50, 50]],
        ]
    )
    result = operational_replay(
        counts,
        [[0.5, 0.5], [0.5, 0.5]],
        [0, 1],
        seed=3,
        draws=20,
    )
    assert result["draws"] == 20
    assert result["source_advantage"]["maximum"] < 0.35
    assert 0.4 < result["worst_conditional_error"]["median"] < 0.65


def test_protocol_allocates_familywise_error_across_both_alphabets() -> None:
    protocol = expected_protocol()
    assert protocol["fine_token_counts"] == [4, 8]
    assert protocol["per_candidate_table_delta"] == 0.05 / (2 * 2 * 13)
