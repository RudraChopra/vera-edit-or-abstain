from __future__ import annotations

import math

from run_mosaic_real_proxy_powered import (
    CALIBRATION_CURVE_FRACTIONS,
    FAMILY_FAILURE,
    PARTITION_FRACTIONS,
    TOKEN_COUNT,
    expected_protocol,
)


def test_powered_proxy_protocol_is_fixed_and_exhaustive() -> None:
    assert TOKEN_COUNT == 2
    assert FAMILY_FAILURE == 0.05
    assert CALIBRATION_CURVE_FRACTIONS == (0.25, 0.50, 0.75, 1.00)
    assert math.isclose(sum(PARTITION_FRACTIONS.values()), 1.0)
    assert PARTITION_FRACTIONS["calibration"] == 0.40
    assert expected_protocol()["confidence_region"] == (
        "coordinate_clopper_pearson"
    )


def test_calibration_curve_cannot_select_the_primary_design() -> None:
    protocol = expected_protocol()
    assert "only the fixed full-calibration point" in protocol["curve_role"]
    assert "no diagnostic outcome" in protocol["design"]
