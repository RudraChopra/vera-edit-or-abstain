from __future__ import annotations

import pytest

from summarize_mosaic_release_utility_table import summarize


def test_utility_summary_reports_mean_and_interval() -> None:
    row = summarize([0.7, 0.8, 0.9])
    assert row["n"] == 3
    assert row["mean"] == pytest.approx(0.8)
    assert row["mean_t95_interval"][0] < 0.8 < row["mean_t95_interval"][1]
