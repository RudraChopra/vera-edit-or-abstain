from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "research" / "mosaic" / "summarize_mosaic_bridge_grouped.py"
SUMMARY = ROOT / "research" / "artifacts" / "mosaic_bridge_evidence_summary_v2.json"
STRICT = ROOT / "research" / "artifacts" / "mosaic_bridge_strict_v2_receipts_v1"


def test_grouped_summary_exposes_dataset_concentration(tmp_path: Path) -> None:
    output = tmp_path / "grouped.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--evidence-summary",
            str(SUMMARY),
            "--strict-receipts",
            str(STRICT),
            "--output",
            str(output),
            "--bootstrap-repetitions",
            "100",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["per_dataset"]["BiasBios-Clinical"]["rules"]["strict_mosaic"]["deployments"] == 20
    assert (
        report["per_dataset"]["Camelyon17-WILDS"]["strict_details"]["limiting_contract"]
        == "zero retained mass (audited missing support)"
    )
    assert report["leave_one_dataset_out"]["BiasBios-Clinical"]["strict_deployments"] == 0
    assert "not be read as an interval over 100 independent real domains" in report["claim_boundary"]
