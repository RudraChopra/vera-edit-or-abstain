from __future__ import annotations

from pathlib import Path

from audit_mosaic_real_proxy_mass_confirmation import audit


ROOT = Path(__file__).resolve().parents[2]


def test_proxy_mass_confirmation_independent_audit_passes() -> None:
    payload = audit(
        ROOT
        / "research/artifacts/mosaic_real_proxy_mass_confirmation_v1.json"
    )
    assert payload["pass"] is True
    assert payload["headline"]["certified_worst_conditional_error_upper"] < 0.40
    assert payload["calibration_curve"][1]["decision"] == "abstain"
    assert payload["calibration_curve"][2]["decision"] == "deploy"
