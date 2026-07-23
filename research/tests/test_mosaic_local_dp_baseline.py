from __future__ import annotations

import itertools
import math

import numpy as np

from run_mosaic_local_dp_baseline import (
    KEEP_PROBABILITY,
    LOCAL_DP_EPSILON,
    SOURCE_THRESHOLD,
    expected_protocol,
    randomized_response_channel,
)


def test_randomized_response_matches_registered_advantage():
    assert math.isclose(2.0 * KEEP_PROBABILITY - 1.0, SOURCE_THRESHOLD)
    assert math.isclose(
        math.exp(LOCAL_DP_EPSILON),
        KEEP_PROBABILITY / (1.0 - KEEP_PROBABILITY),
    )


def test_every_task_map_channel_is_epsilon_local_dp():
    for task_map in itertools.product((0, 1), repeat=4):
        channel = randomized_response_channel(task_map)
        assert np.allclose(channel.sum(axis=1), 1.0)
        for output in range(2):
            ratios = (
                channel[:, output, None]
                / channel[None, :, output]
            )
            assert float(np.max(ratios)) <= math.exp(LOCAL_DP_EPSILON) + 1e-12


def test_protocol_enumerates_all_four_token_maps():
    protocol = expected_protocol()
    assert protocol["fine_token_count"] == 4
    assert protocol["released_token_count"] == 2
    assert "16 deterministic" in protocol["decoder_search"]
