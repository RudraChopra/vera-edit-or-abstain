from __future__ import annotations

import json
from pathlib import Path

from audit_mosaic_path9_goal_completion import collect_checks


def test_goal_audit_checks_every_requested_objective(tmp_path: Path) -> None:
    cam_summary = tmp_path / "cam-summary.json"
    cam_audit = tmp_path / "cam-audit.json"
    cam_summary.write_text(
        json.dumps(
            {
                "main_paper_inclusion_gate_passed": True,
                "primary_release_count": 5,
                "primary_heldout_violation_count": 0,
                "operational_violation_count": 0,
            }
        ),
        encoding="utf-8",
    )
    cam_audit.write_text(json.dumps({"pass": True}), encoding="utf-8")
    checks = collect_checks(
        cam_summary_path=cam_summary,
        cam_audit_path=cam_audit,
    )
    assert len(checks) >= 16
    assert all(checks.values())
