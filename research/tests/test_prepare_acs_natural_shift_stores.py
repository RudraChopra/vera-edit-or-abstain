from __future__ import annotations

import numpy as np
import pandas as pd

from prepare_acs_natural_shift_stores import extract_task, reference_split


class _Problem:
    features = ["AGEP", "SEX", "SCHL"]
    target = "TARGET"
    target_transform = staticmethod(lambda values: values > 0)
    _preprocess = staticmethod(lambda frame: frame.loc[frame["KEEP"] == 1])
    _postprocess = staticmethod(lambda values: np.nan_to_num(values, nan=-1))


def test_extract_task_excludes_source_and_preserves_puma() -> None:
    frame = pd.DataFrame(
        {
            "AGEP": [20, 30, 40],
            "SEX": [1, 2, 1],
            "SCHL": [10.0, np.nan, 14.0],
            "TARGET": [1, 0, 1],
            "PUMA": [100, 200, 300],
            "KEEP": [1, 1, 0],
        }
    )
    features, labels, sources, environments, columns = extract_task(frame, _Problem())
    assert columns == ("AGEP", "SCHL")
    assert features.tolist() == [[20.0, 10.0], [30.0, -1.0]]
    assert labels.tolist() == [1, 0]
    assert sources.tolist() == [0, 1]
    assert environments.tolist() == [100, 200]


def test_reference_split_is_deterministic_and_nonempty() -> None:
    first = reference_split(100, seed=7)
    second = reference_split(100, seed=7)
    assert np.array_equal(first, second)
    assert np.count_nonzero(first == 1) == 20
    assert np.count_nonzero(first == 0) == 80
