from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_locked_path9_theory_artifact_passes_every_gate() -> None:
    report = json.loads(
        (ROOT / "research/artifacts/mosaic_path9_theory_v1.json").read_text()
    )
    audit = json.loads(
        (
            ROOT / "research/artifacts/mosaic_path9_theory_v1_audit.json"
        ).read_text()
    )
    assert report["pass"] is True
    assert audit["pass"] is True
    assert all(audit["checks"].values())
