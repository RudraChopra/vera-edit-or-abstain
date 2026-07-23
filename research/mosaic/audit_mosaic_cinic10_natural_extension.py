#!/usr/bin/env python3
"""Audit the locked 40-seed CINIC-10 natural-origin extension."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "research/artifacts/mosaic_cinic10_natural_v2.json"
OUTPUT = ROOT / "research/artifacts/mosaic_cinic10_natural_v2_audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    rows = report["rows"]
    selected = [
        candidate
        for row in rows
        for candidate in row["candidates"]
        if candidate["candidate"] == row["selected_primary_candidate"]
    ]
    false_acceptances = sum(
        value["threshold_decisions"]["0.40"]["false_acceptance"]
        for value in selected
    )
    checks = {
        "forty_registered_seeds": len(rows) == 40,
        "unique_registered_seeds": len({row["seed"] for row in rows}) == 40,
        "two_candidates_per_seed": all(
            len(row["candidates"]) == 2 for row in rows
        ),
        "all_strata_supported": all(
            min(
                value
                for table in (
                    candidate["reference_stratum_counts"],
                    candidate["bridge_stratum_counts"],
                )
                for label in table
                for value in label
            )
            > 0
            for row in rows
            for candidate in row["candidates"]
        ),
        "summary_matches": (
            report["summary"]["primary_releases"] == len(selected)
            and report["summary"]["primary_false_acceptances"]
            == false_acceptances
        ),
        "registered_gate_matches": report["pass"] == (
            len(selected) >= 30 and false_acceptances == 0
        ),
    }
    payload = {
        "name": "MOSAIC CINIC-10 natural-origin extension audit v2",
        "report_sha256": sha256(REPORT),
        "checks": checks,
        "pass": all(checks.values()),
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
