#!/usr/bin/env python3
"""Independently audit the ACS proxy source-mass confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from mosaic_transform_exact_optimizer import optimize_transform_exact_channel


ROOT = Path(__file__).resolve().parents[2]
REPORT = (
    ROOT / "research/artifacts/mosaic_real_proxy_mass_confirmation_v1.json"
)
LOCK = (
    ROOT / "research/mosaic/prereg_mosaic_real_proxy_mass_confirmation_v1.json"
)
OUTPUT = (
    ROOT
    / "research/artifacts/"
    "mosaic_real_proxy_mass_confirmation_audit_v1.json"
)
PRIVACY_THRESHOLD = 0.35
UTILITY_THRESHOLD = 0.40


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def close(left: float, right: float, tolerance: float = 2e-8) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def audit(report_path: Path) -> dict[str, Any]:
    report = load(report_path)
    lock = load(LOCK)
    failures: list[str] = []
    checks = 0

    def require(condition: bool, message: str) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            failures.append(message)

    require(report["lock_sha256"] == sha256(LOCK), "lock hash differs")
    require(report["protocol"] == lock["protocol"], "protocol differs")
    require(report["claim_boundary"] == lock["claim_boundary"], "claim differs")
    require(report["passed"] is True, "primary gate did not pass")
    require(report["proxy_balanced_accuracy"] >= 0.60, "proxy is trivial")
    curve = report["calibration_curve"]
    require(
        [row["fraction"] for row in curve] == [0.25, 0.5, 0.75, 1.0],
        "calibration fractions differ",
    )
    rows = [int(row["calibration_rows"]) for row in curve]
    require(rows == sorted(rows), "calibration rows are not monotone")
    radii = [float(row["maximum_conditional_l1_radius"]) for row in curve]
    require(
        all(left > right for left, right in zip(radii, radii[1:])),
        "calibrated radii do not shrink monotonically",
    )
    require(
        [row["decision"] for row in curve]
        == ["abstain", "abstain", "deploy", "deploy"],
        "locked calibration decision curve differs",
    )

    certificate = report["primary_proxy_certificate"]
    base_per_event = float(certificate["per_event_failure_probability"])
    mass_per_event = float(
        certificate["source_mass_per_event_failure_probability"]
    )
    family_spend = 10 * base_per_event + 4 * mass_per_event
    require(family_spend <= 0.05 + 1e-12, "confidence family overspends")
    require(
        certificate["source_mass_calibration_sample_size"]
        == rows[-1],
        "source-mass calibration sample differs",
    )
    for label in certificate["labels"]:
        lowers = np.asarray(label["source_mass_interval_lowers"])
        uppers = np.asarray(label["source_mass_interval_uppers"])
        require(np.all(lowers >= 0.0), "negative source-mass lower")
        require(np.all(uppers <= 1.0), "source-mass upper exceeds one")
        require(np.all(lowers <= uppers), "source-mass interval is empty")

    centers = np.asarray(certificate["conditional_centers"], dtype=np.float64)
    conditional_radii = np.asarray(
        certificate["conditional_l1_radii"], dtype=np.float64
    )
    identity = ((np.eye(2),), (np.eye(2),))
    solution = optimize_transform_exact_channel(
        centers,
        l1_radii=conditional_radii,
        common_channels_by_label=identity,
        contaminations=(0.0, 0.0),
        privacy_advantage_thresholds=(
            PRIVACY_THRESHOLD,
            PRIVACY_THRESHOLD,
        ),
        released_token_count=2,
        maximum_worst_conditional_error=UTILITY_THRESHOLD,
        solver_time_limit_seconds=300.0,
    )
    release = report["release"]
    require(release["decision"] == "deploy", "primary decision is not deploy")
    require(
        np.allclose(
            np.asarray(release["release_channel"]),
            solution.release_channel,
            atol=2e-8,
            rtol=0.0,
        ),
        "replayed release channel differs",
    )
    require(
        tuple(release["decoder"]) == tuple(solution.decoder),
        "replayed decoder differs",
    )
    replay_privacy = [
        float(value.normalized_advantage)
        for value in solution.privacy_certificates
    ]
    require(
        np.allclose(
            release["certified_source_advantage_upper"],
            replay_privacy,
            atol=2e-8,
            rtol=0.0,
        ),
        "replayed source bounds differ",
    )
    require(
        close(
            release["certified_worst_conditional_error_upper"],
            solution.certified_worst_conditional_error,
        ),
        "replayed utility bound differs",
    )
    require(max(replay_privacy) <= PRIVACY_THRESHOLD, "privacy contract fails")
    require(
        solution.certified_worst_conditional_error <= UTILITY_THRESHOLD,
        "utility contract fails",
    )
    require(release["diagnostic_safe"] is True, "diagnostic is unsafe")
    require(
        release["diagnostic_source_advantage"] <= PRIVACY_THRESHOLD,
        "diagnostic privacy contract fails",
    )
    require(
        release["diagnostic_worst_conditional_error"] <= UTILITY_THRESHOLD,
        "diagnostic utility contract fails",
    )
    require(all(report["gates"].values()), "one or more primary gates fail")

    return {
        "name": "MOSAIC ACS proxy source-mass independent audit v1",
        "pass": not failures,
        "checks": checks,
        "failures": failures,
        "report_sha256": sha256(report_path),
        "lock_sha256": sha256(LOCK),
        "family_failure_spend": family_spend,
        "calibration_curve": [
            {
                "rows": row["calibration_rows"],
                "radius": row["maximum_conditional_l1_radius"],
                "decision": row["decision"],
            }
            for row in curve
        ],
        "headline": {
            "proxy_balanced_accuracy": report["proxy_balanced_accuracy"],
            "certified_source_advantage_upper": replay_privacy,
            "certified_worst_conditional_error_upper": (
                solution.certified_worst_conditional_error
            ),
            "diagnostic_source_advantage": (
                release["diagnostic_source_advantage"]
            ),
            "diagnostic_worst_conditional_error": (
                release["diagnostic_worst_conditional_error"]
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    payload = audit(args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
