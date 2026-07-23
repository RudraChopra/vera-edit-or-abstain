#!/usr/bin/env python3
"""Audit the locked official FARE comparison under MOSAIC's proxy contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "research/artifacts/mosaic_fare_proxy_comparison_v1.json"
OUTPUT = ROOT / "research/artifacts/mosaic_fare_proxy_comparison_v1_audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    candidates = report["candidates"]
    expected = {
        f"FARE::leaves={leaves},alpha={alpha:.2f}"
        for leaves in (2, 4)
        for alpha in (0.80, 0.90, 0.95, 0.99)
    }
    certified = [
        candidate
        for candidate in candidates
        if candidate["decision"] == "deploy"
    ]
    checks = {
        "complete_registered_grid": {
            candidate["candidate"] for candidate in candidates
        }
        == expected,
        "all_calibration_cells_observed": all(
            candidate["calibration_cells_observed"]
            for candidate in candidates
        ),
        "summary_matches": (
            report["summary"]["candidate_count"] == len(candidates)
            and report["summary"]["certified_candidates"] == len(certified)
            and report["summary"]["false_acceptances"]
            == sum(candidate["false_acceptance"] for candidate in candidates)
        ),
        "selected_matches": (
            (report["selected"] is None and not certified)
            or (
                report["selected"] is not None
                and report["selected"]["decision"] == "deploy"
            )
        ),
        "no_false_acceptances": not any(
            candidate["false_acceptance"] for candidate in candidates
        ),
    }
    payload = {
        "name": "Official FARE commensurate proxy comparison audit",
        "report_sha256": sha256(REPORT),
        "checks": checks,
        "pass": all(checks.values()),
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    if not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
