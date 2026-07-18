"""Regression checks for the current AAAI source-readiness semantics."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_aaai2027_source_readiness import collect_checks


def test_current_source_checks_the_p0_disclosure_and_current_matrix() -> None:
    checks, _ = collect_checks()
    by_key = {check.key: check for check in checks}

    assert by_key["official_baseline_matrix_present"].status == "pass"
    assert by_key["final_p0_negative_result_disclosed"].status == "pass"
    assert by_key["anonymous_source_identity_free"].status == "pass"
    assert by_key["clinical_boundary_present"].status == "pass"
