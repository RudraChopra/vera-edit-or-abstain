#!/usr/bin/env python3
"""Audit the locked real-attribute-imputer boundary study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "research/artifacts/mosaic_real_proxy_v1.json"
OUTPUT = ROOT / "research/artifacts/mosaic_real_proxy_v1_audit.json"


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    radii = [
        value
        for row in report["proxy_certificate"]["conditional_l1_radii"]
        for value in row
    ]
    checks = {
        "real_proxy_is_nontrivial": report["proxy_balanced_accuracy"] >= 0.60,
        "all_calibration_cells_observed": report["gates"][
            "all_calibration_cells_observed"
        ],
        "decision_is_fail_closed": (
            report["release"]["decision"] == "abstain"
            and report["release"]["release_channel"] is None
        ),
        "known_abstention_reason": report["release"]["reason"]
        == "ABSTAIN_EXACT_UTILITY_CONTRACT",
        "finite_reported_radii": all(
            value >= 0 and value <= 2 for value in radii
        ),
        "no_false_acceptance": report["gates"]["no_false_acceptance"],
    }
    payload = {
        "name": "MOSAIC real proxy study audit",
        "report_sha256": hashlib.sha256(REPORT.read_bytes()).hexdigest(),
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
