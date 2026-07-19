from __future__ import annotations

import numpy as np

from prepare_acs_income_store import SPLIT_REFERENCE, SPLIT_TRAIN, make_california_splits


def test_california_split_is_deterministic_and_nonempty() -> None:
    first = make_california_splits(25, seed=7)
    second = make_california_splits(25, seed=7)
    assert np.array_equal(first, second)
    assert int((first == SPLIT_REFERENCE).sum()) == 5
    assert int((first == SPLIT_TRAIN).sum()) == 20
