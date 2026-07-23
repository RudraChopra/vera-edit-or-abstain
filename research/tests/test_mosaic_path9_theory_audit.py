from __future__ import annotations

from audit_mosaic_path9_theory import (
    binary_lower_bound,
    weissman_upper_bound,
)


def test_theory_audit_reconstructs_locked_examples() -> None:
    assert binary_lower_bound(0.1, 0.025, 0.025, 0.025) == 439
    assert weissman_upper_bound(2, 0.025, 0.025) == 14023
