from __future__ import annotations

from pathlib import Path

from analyze_mosaic_acs_infeasibility import analyze_receipt


ROOT = Path(__file__).resolve().parents[1]


def test_acs_error_decomposition_closes() -> None:
    row = analyze_receipt(
        ROOT
        / "artifacts/mosaic_acs_bridge_strict_v3_receipts"
        / "ACSIncome-CA-TX__seed1305.json"
    )
    decomposition = row["worst_stratum_decomposition"]
    recomposed = (
        decomposition["center_error"]
        + decomposition["sampling_charge"]
        + decomposition["bridge_residual_charge"]
        + row["strict_outward_guard_gap"]
    )
    assert abs(recomposed - row["certified_error"]) <= 1e-10
    assert row["margin_to_primary_contract"] > 0.0
