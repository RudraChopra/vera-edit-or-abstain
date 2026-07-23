from __future__ import annotations

from audit_mosaic_qwen_powered_confirmation import normalize


def test_normalize_rounds_floats_and_orders_nested_records() -> None:
    value = {
        "z": [0.12345678901234],
        "a": {"b": 2, "a": 1},
    }
    assert normalize(value) == {
        "a": {"a": 1, "b": 2},
        "z": [0.123456789012],
    }
