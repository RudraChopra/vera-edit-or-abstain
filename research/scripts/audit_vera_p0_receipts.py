"""Verify completeness and mechanical integrity of the locked VERA P0 receipts.

This gate deliberately checks identities, hashes, and array schemas only.  It
does not calculate any target, leakage, selection, or deployment outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREG = ROOT / "prereg_vera_p0_confirmation_v4.json"
DEFAULT_HASH = ROOT / "prereg_vera_p0_confirmation_v4.sha256"
DEFAULT_RECEIPTS = Path(
    "/Volumes/Backups/FARO/artifacts/vera_p0_confirmation_v4_receipts"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "vera_p0_confirmation_v4_receipt_audit.json"
CANDIDATE_DISPLAY_NAMES = {"rlace": "R-LACE"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def expected_receipts(prereg: dict[str, Any]) -> dict[str, tuple[str, int, str]]:
    study = prereg["real_study"]
    return {
        f"{dataset}__{method}__seed-{seed}.json": (dataset, int(seed), method)
        for dataset in study["datasets"]
        for method in study["methods"]
        for seed in prereg["data_policy"]["confirmatory_seeds"]
    }


def audit(
    prereg: dict[str, Any], prereg_hash: str, receipt_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    study = prereg["real_study"]
    expected = expected_receipts(prereg)
    required_arrays = set(study["construction_receipt_schema"]["required_arrays"])
    heldout = str(study["heldout_attacker"]["name"])
    expected_attackers = study["leakage_attackers"]
    method_counts = {
        str(method): int(config["candidate_count"])
        for method, config in study["methods"].items()
    }
    observed = {path.name for path in receipt_dir.glob("*.json")}
    findings: list[dict[str, Any]] = []
    split_identities: dict[tuple[str, int], dict[str, str]] = {}

    for name in sorted(set(expected).difference(observed)):
        findings.append({"kind": "missing_receipt", "receipt": name})
    for name in sorted(observed.difference(expected)):
        findings.append({"kind": "unexpected_receipt", "receipt": name})

    checked_audits = 0
    for name, (dataset, seed, method) in sorted(expected.items()):
        path = receipt_dir / name
        if not path.is_file():
            continue
        try:
            receipt = load_json(path)
        except Exception as exc:  # Report every malformed receipt in one pass.
            findings.append({"kind": "unreadable_receipt", "receipt": name, "detail": str(exc)})
            continue

        required = {
            "claim_grade": True,
            "smoke": False,
            "claim_configuration_verified": True,
            "prereg_sha256": prereg_hash,
            "dataset": dataset,
            "seed": seed,
        }
        for field, expected_value in required.items():
            actual = receipt.get(field)
            if actual != expected_value:
                findings.append(
                    {
                        "kind": "receipt_field_mismatch",
                        "receipt": name,
                        "field": field,
                        "expected": expected_value,
                        "observed": actual,
                    }
                )
        if receipt.get("method_family") != method:
            findings.append(
                {
                    "kind": "method_mismatch",
                    "receipt": name,
                    "expected": method,
                    "observed": receipt.get("method_family"),
                }
            )
        if receipt.get("registered_attackers") != expected_attackers:
            findings.append({"kind": "attacker_portfolio_mismatch", "receipt": name})
        heldout_config = receipt.get("heldout_attacker", {})
        if not isinstance(heldout_config, dict) or heldout_config.get("name") != heldout:
            findings.append({"kind": "heldout_attacker_mismatch", "receipt": name})

        candidates = receipt.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != method_counts[method]:
            findings.append(
                {
                    "kind": "candidate_count_mismatch",
                    "receipt": name,
                    "expected": method_counts[method],
                    "observed": None if not isinstance(candidates, list) else len(candidates),
                }
            )
            continue

        indices = receipt.get("indices")
        if not isinstance(indices, dict):
            findings.append({"kind": "missing_split_identities", "receipt": name})
        else:
            try:
                identity = {
                    split: str(indices[split]["sha256"])
                    for split in ("train", "construction", "certification", "external")
                }
            except (KeyError, TypeError):
                findings.append({"kind": "malformed_split_identities", "receipt": name})
            else:
                key = (dataset, seed)
                prior = split_identities.setdefault(key, identity)
                if prior != identity:
                    findings.append(
                        {"kind": "split_identity_mismatch", "receipt": name, "dataset": dataset, "seed": seed}
                    )

        seen_keys: set[str] = set()
        for candidate in candidates:
            if not isinstance(candidate, dict):
                findings.append({"kind": "malformed_candidate", "receipt": name})
                continue
            candidate_key = str(candidate.get("candidate_key", ""))
            audit_path = Path(str(candidate.get("audit_npz", "")))
            if not candidate_key or candidate_key in seen_keys:
                findings.append({"kind": "candidate_key_mismatch", "receipt": name})
                continue
            seen_keys.add(candidate_key)
            expected_display_name = CANDIDATE_DISPLAY_NAMES.get(
                method, study["methods"][method]["display_name"]
            )
            if candidate.get("method") != expected_display_name:
                findings.append(
                    {
                        "kind": "candidate_method_mismatch",
                        "receipt": name,
                        "candidate": candidate_key,
                        "expected": expected_display_name,
                        "observed": candidate.get("method"),
                    }
                )
            if not audit_path.is_file():
                findings.append({"kind": "missing_audit", "receipt": name, "candidate": candidate_key})
                continue
            if sha256(audit_path) != candidate.get("audit_npz_sha256"):
                findings.append({"kind": "audit_hash_mismatch", "receipt": name, "candidate": candidate_key})
                continue
            try:
                with np.load(audit_path, allow_pickle=False) as archive:
                    keys = set(archive.files)
            except Exception as exc:
                findings.append(
                    {"kind": "unreadable_audit", "receipt": name, "candidate": candidate_key, "detail": str(exc)}
                )
                continue
            checked_audits += 1
            missing = sorted(required_arrays.difference(keys))
            heldout_keys = {
                f"heldout_leakage_correct_{split}__{heldout}"
                for split in ("construction", "certification", "external")
            }
            if missing:
                findings.append(
                    {"kind": "missing_construction_arrays", "receipt": name, "candidate": candidate_key, "arrays": missing}
                )
            if heldout_keys.difference(keys):
                findings.append(
                    {"kind": "missing_heldout_arrays", "receipt": name, "candidate": candidate_key}
                )

    summary = {
        "expected_receipts": len(expected),
        "observed_receipts": len(observed),
        "checked_audits": checked_audits,
        "expected_candidates": len(prereg["data_policy"]["confirmatory_seeds"])
        * len(study["datasets"])
        * int(study["candidate_count_total"]),
    }
    return findings, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--hash-file", type=Path, default=DEFAULT_HASH)
    parser.add_argument("--receipt-dir", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite audit output: {args.output}")
    prereg = load_json(args.prereg)
    prereg_hash = args.hash_file.read_text(encoding="utf-8").strip().split()[0]
    if sha256(args.prereg) != prereg_hash:
        raise RuntimeError("preregistration sidecar hash mismatch")
    findings, summary = audit(prereg, prereg_hash, args.receipt_dir)
    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration": str(args.prereg),
        "preregistration_sha256": prereg_hash,
        "receipt_dir": str(args.receipt_dir),
        "passed": not findings,
        "summary": summary,
        "findings": findings,
        "scope": "mechanical receipt identity, hash, split, and schema checks only; no scientific outcomes are calculated",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if findings:
        raise SystemExit(f"receipt audit failed with {len(findings)} finding(s)")


if __name__ == "__main__":
    main()
