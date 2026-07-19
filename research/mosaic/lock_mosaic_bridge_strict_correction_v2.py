#!/usr/bin/env python3
"""Lock the disclosed v2 bridge correction before its complete rerun."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from mosaic_real import sha256


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
DEFAULT_STRICT_OUTPUT = ROOT / "prereg_mosaic_bridge_strict_amendment_v2.json"
DEFAULT_RATIONAL_OUTPUT = ROOT / "prereg_mosaic_bridge_rational_audit_v2.json"
CORRECTION_FILES = (
    "research/mosaic/mosaic_strict_certification_v2.py",
    "research/mosaic/replay_mosaic_bridge_strict_v2.py",
    "research/mosaic/audit_mosaic_bridge_strict_v2.py",
    "research/mosaic/lock_mosaic_bridge_strict_correction_v2.py",
    "research/tests/test_mosaic_strict_certification_v2.py",
    "research/mosaic/mosaic_strict_certification.py",
)
RATIONAL_FILES = (
    "research/mosaic/audit_mosaic_bridge_rational.py",
    "research/mosaic/mosaic_rational_certificate.py",
    "research/tests/test_mosaic_rational_certificate.py",
)
V1_EVIDENCE = (
    "research/artifacts/mosaic_bridge_strict_manifest_v1.json",
    "research/artifacts/mosaic_bridge_strict_audit_v1.json",
    "research/artifacts/mosaic_bridge_rational_audit_v1.json",
    "research/artifacts/mosaic_bridge_evidence_summary_v1.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prereg",
        type=Path,
        default=ROOT / "prereg_mosaic_bridge_v1.json",
    )
    parser.add_argument(
        "--v1-amendment",
        type=Path,
        default=ROOT / "prereg_mosaic_bridge_strict_amendment_v1.json",
    )
    parser.add_argument("--strict-output", type=Path, default=DEFAULT_STRICT_OUTPUT)
    parser.add_argument(
        "--rational-output", type=Path, default=DEFAULT_RATIONAL_OUTPUT
    )
    return parser.parse_args()


def write_lock(path: Path, payload: dict[str, object]) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if path.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite {path} or its sidecar")
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(data, encoding="utf-8")
    digest = hashlib.sha256(data.encode("utf-8")).hexdigest()
    sidecar.write_text(digest + "\n", encoding="utf-8")
    return digest


def checked_hash(path: Path) -> str:
    digest = sha256(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.read_text(encoding="utf-8").strip() != digest:
        raise ValueError(f"sidecar mismatch for {path}")
    return digest


def main() -> None:
    args = parse_args()
    prereg_hash = checked_hash(args.prereg)
    v1_amendment_hash = checked_hash(args.v1_amendment)
    evidence_hashes = {
        relative: sha256(REPOSITORY / relative) for relative in V1_EVIDENCE
    }
    code_hashes = {
        relative: sha256(REPOSITORY / relative) for relative in CORRECTION_FILES
    }
    now = datetime.now(timezone.utc).isoformat()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    strict_payload: dict[str, object] = {
        "project": "MOSAIC disclosed strict numerical correction v2",
        "status": "locked_after_v1_outcome_inspection_before_any_v2_replay",
        "locked_at": now,
        "repository_head_before_v2_replay": head,
        "original_preregistration_sha256": prereg_hash,
        "v1_strict_amendment_sha256": v1_amendment_hash,
        "v1_evidence_sha256": evidence_hashes,
        "timing_disclosure": {
            "v1_matrix_complete": True,
            "v1_outcomes_inspected": True,
            "v1_primary_deployments_seen": 0,
            "specific_failure_inspected": (
                "A BiasBios INLP rank-4 receipt retained about 0.80 before the "
                "v1 repair and zero afterward because an exactly zero transform "
                "output column received an artificial positive denominator."
            ),
        },
        "trigger": (
            "The completed v1 audit exposed a repair-layer implementation error: "
            "adding the feasibility guard to an identically zero transform column "
            "changed the vacuous inequality lower >= retained * 0 into a false "
            "retained-mass restriction."
        ),
        "correction": {
            "scope": (
                "Skip guard-ratio contraction only for output columns whose full "
                "serialized transform column is exactly zero."
            ),
            "unchanged_checks": (
                "Recompute every original bridge slack, including structural-zero "
                "columns, and require every slack to be nonnegative."
            ),
            "unchanged_components": [
                "datasets and stored token tables",
                "all 100 dataset-seed receipts",
                "13 candidates per receipt and their ordering",
                "source and utility thresholds",
                "confidence radii and multiplicity allocation",
                "release optimization and outward risk guards",
                "diagnostic splits and comparator rules",
            ],
            "expected_direction_only": (
                "The correction may restore positive retained mass when exact-zero "
                "outputs occur; no deployment count or favorable outcome is locked."
            ),
        },
        "required_outputs": {
            "strict_receipts": 100,
            "candidate_rows": 1300,
            "global_optimizations": 1400,
            "minimum_membership_slack": 0.0,
            "maximum_decision_tolerance": 0.0,
            "deterministic_replay_of_every_receipt": True,
            "exact_rational_audit_of_every_bridge_and_release": True,
        },
        "stopping_rule": (
            "Rerun and audit all 100 receipts regardless of interim outcomes; "
            "preserve and report v1 as the superseded failed repair."
        ),
        "claim_rule": (
            "Version 2 is disclosed post-audit corrective evidence, not an "
            "original preregistered outcome. Any paper claim must identify that "
            "status and retain the v1 record."
        ),
        "code_sha256": code_hashes,
    }
    strict_hash = write_lock(args.strict_output, strict_payload)
    rational_payload: dict[str, object] = {
        "project": "MOSAIC v2 exact-rational serialized-certificate audit",
        "status": "locked_after_v1_outcome_inspection_before_any_v2_replay",
        "locked_at": now,
        "strict_amendment_sha256": strict_hash,
        "arithmetic_contract": {
            "scope": "all 1,300 bridges and all 1,400 release certificates",
            "serialized_numbers": "interpreted as exact decimal rationals",
            "stochastic_rows": "renormalized exactly over the rationals",
            "multinomial_radii": "inflated outward by exactly 1e-12",
            "bridge_gate": "every exact rational membership slack is nonnegative",
            "risk_gate": (
                "every stored outward source and utility bound is at least its "
                "exact rational recomputation"
            ),
        },
        "timing_disclosure": strict_payload["timing_disclosure"],
        "code_sha256": {
            relative: sha256(REPOSITORY / relative)
            for relative in RATIONAL_FILES
        },
    }
    rational_hash = write_lock(args.rational_output, rational_payload)
    print(json.dumps({"strict": strict_hash, "rational": rational_hash}, indent=2))


if __name__ == "__main__":
    main()
