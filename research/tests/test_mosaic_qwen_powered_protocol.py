from __future__ import annotations

import numpy as np

from lock_mosaic_qwen_powered_confirmation import protocol
from prepare_civilcomments_qwen_powered_store import sha256_array
from run_mosaic_qwen_powered_confirmation import (
    diagnostic_counts,
    expected_balanced_accuracy,
)


def test_qwen_powered_protocol_fixes_binary_temporal_study() -> None:
    value = protocol()
    assert value["fine_token_count"] == 2
    assert value["primary_utility_threshold"] == 0.40
    assert value["seeds"] == [4201, 4202, 4203, 4204, 4205]
    assert value["familywise_delta"] == 0.05


def test_qwen_expected_accuracy_and_counts() -> None:
    tokens = np.asarray([0, 0, 1, 1], dtype=np.int16)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int16)
    sources = np.asarray([0, 1, 0, 1], dtype=np.int16)
    channel = np.eye(2)
    assert expected_balanced_accuracy(
        tokens, labels, channel, (0, 1)
    ) == 1.0
    assert diagnostic_counts(labels, sources) == [[1, 1], [1, 1]]
    assert sha256_array(np.asarray([1, 2, 3], dtype=np.int64)) == (
        "e2e2033ae7e19d680599d4eb0a1359a2"
        "b48ec5baac75066c317fbf85159c54ef"
    )
