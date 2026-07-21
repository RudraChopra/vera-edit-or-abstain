from __future__ import annotations

import numpy as np

from run_mosaic_qwen_temporal_confirmation import (
    sampled_release_tokens,
    training_decoder,
)


def test_training_decoder_uses_construction_majority_only() -> None:
    tokens = np.asarray([0, 0, 0, 1, 1, 2], dtype=np.int16)
    target = np.asarray([0, 0, 1, 1, 1, 0], dtype=np.int16)

    assert training_decoder(tokens, target, token_count=4) == (0, 1, 0, 0)


def test_operational_sampler_respects_channel_and_seed() -> None:
    fine = np.asarray([0, 1, 0, 1], dtype=np.int16)
    identity = np.eye(2)
    assert np.array_equal(
        sampled_release_tokens(fine, identity, np.random.default_rng(7)),
        fine,
    )

    channel = np.asarray([[0.25, 0.75], [0.80, 0.20]])
    first = sampled_release_tokens(fine, channel, np.random.default_rng(8))
    second = sampled_release_tokens(fine, channel, np.random.default_rng(8))
    assert np.array_equal(first, second)
