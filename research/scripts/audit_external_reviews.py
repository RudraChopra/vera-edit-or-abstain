"""Fail-closed audit for two genuine external ML-publisher reviews."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "private" / "external_review_registry.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "external_review_audit.json"


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def resolve_registry_path(value: Any) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else ROOT.parent / path


def review_valid(
    review: dict[str, Any], frozen_main_hash: str
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    review_id = str(review.get("review_id", "unknown"))
    required_text = (
        "reviewer_name",
        "affiliation",
        "publication_url",
        "requested_date",
        "received_date",
        "review_file",
        "main_pdf_sha256_reviewed",
    )
    for key in required_text:
        if not nonempty(review.get(key)):
            failures.append(f"{review_id}: missing {key}")
    publication_url = str(review.get("publication_url", ""))
    if publication_url and not publication_url.startswith(("https://", "http://")):
        failures.append(f"{review_id}: publication URL is not HTTP(S)")
    if review.get("publishes_in_machine_learning") is not True:
        failures.append(f"{review_id}: ML publication qualification not verified")
    if review.get("cold_review_attested") is not True:
        failures.append(f"{review_id}: cold-review attestation is absent")
    if review.get("conflict_disclosed") is not True:
        failures.append(f"{review_id}: conflict disclosure is absent")
    if review.get("human_authorship_attested") is not True:
        failures.append(f"{review_id}: human authorship is not attested")
    if review.get("ltt_overlap_explicitly_assessed") is not True:
        failures.append(f"{review_id}: LTT overlap was not explicitly assessed")
    if not isinstance(review.get("unaddressed_ltt_overlap"), bool):
        failures.append(f"{review_id}: LTT-overlap verdict is not explicit")
    if review.get("prompt_risk_control_overlap_explicitly_assessed") is not True:
        failures.append(
            f"{review_id}: Prompt Risk Control overlap was not explicitly assessed"
        )
    if not isinstance(review.get("unaddressed_prompt_risk_control_overlap"), bool):
        failures.append(
            f"{review_id}: Prompt Risk Control overlap verdict is not explicit"
        )
    if review.get("all_findings_transcribed") is not True:
        failures.append(f"{review_id}: complete finding transcription is not attested")
    if frozen_main_hash and review.get("main_pdf_sha256_reviewed") != frozen_main_hash:
        failures.append(f"{review_id}: reviewed main PDF hash is not frozen hash")
    review_file = Path(str(review.get("review_file", "")))
    if review_file and not review_file.is_absolute():
        review_file = ROOT.parent / review_file
    if not review_file.is_file() or review_file.stat().st_size < 200:
        failures.append(f"{review_id}: review file is absent or too short")
    return not failures, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = load_json(args.registry)
    frozen = registry.get("frozen_package", {})
    frozen = frozen if isinstance(frozen, dict) else {}
    frozen_main_hash = str(frozen.get("anonymous_main_pdf_sha256", ""))
    reviews = registry.get("reviews", [])
    reviews = reviews if isinstance(reviews, list) else []
    findings = registry.get("findings", [])
    findings = findings if isinstance(findings, list) else []

    failures: list[str] = []
    if not registry:
        failures.append("private external-review registry is absent")
    frozen_specs = (
        ("anonymous_main_pdf", "anonymous_main_pdf_sha256"),
        ("anonymous_supplement_pdf", "anonymous_supplement_pdf_sha256"),
        ("anonymous_code_archive", "anonymous_code_archive_sha256"),
    )
    frozen_files_verified = True
    for path_key, hash_key in frozen_specs:
        path = resolve_registry_path(frozen.get(path_key))
        expected = str(frozen.get(hash_key, ""))
        if not path.is_file() or len(expected) != 64 or sha256(path) != expected:
            frozen_files_verified = False
            failures.append(f"frozen package file/hash mismatch: {path_key}")
    valid_reviews: list[dict[str, Any]] = []
    for value in reviews:
        if not isinstance(value, dict):
            failures.append("review registry contains a non-object review")
            continue
        valid, review_failures = review_valid(value, frozen_main_hash)
        failures.extend(review_failures)
        if valid:
            valid_reviews.append(value)

    valid_ids = {str(review["review_id"]) for review in valid_reviews}
    valid_names = {
        str(review["reviewer_name"]).strip().casefold() for review in valid_reviews
    }
    valid_publications = {
        str(review["publication_url"]).strip() for review in valid_reviews
    }
    critical_major = [
        finding
        for finding in findings
        if isinstance(finding, dict)
        and str(finding.get("severity", "")).lower() in {"critical", "major"}
    ]
    unresolved = [
        finding
        for finding in critical_major
        if str(finding.get("status", "")).lower() not in {"fixed", "rebutted"}
        or not nonempty(finding.get("resolution"))
        or not nonempty(finding.get("paper_or_code_location"))
    ]
    unresolved_critical = sum(
        str(finding.get("severity", "")).lower() == "critical"
        for finding in unresolved
    )
    unresolved_major = sum(
        str(finding.get("severity", "")).lower() == "major"
        for finding in unresolved
    )
    ltt_unaddressed = sum(
        review.get("unaddressed_ltt_overlap") is True for review in valid_reviews
    )
    prompt_risk_unaddressed = sum(
        review.get("unaddressed_prompt_risk_control_overlap") is True
        for review in valid_reviews
    )
    orphaned_critical_major = [
        finding
        for finding in critical_major
        if str(finding.get("review_id")) not in valid_ids
    ]
    response_complete = (
        len(valid_reviews) >= 2
        and not unresolved
        and not orphaned_critical_major
    )
    identity_verified = (
        registry.get("reviewer_identity_evidence_human_verified") is True
    )
    passed = (
        len(valid_reviews) >= 2
        and len(valid_names) >= 2
        and len(valid_publications) >= 2
        and unresolved_critical == 0
        and unresolved_major == 0
        and ltt_unaddressed == 0
        and prompt_risk_unaddressed == 0
        and response_complete
        and identity_verified
        and frozen_files_verified
    )
    audit = {
        "name": "VERA external human-review audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "registry_present": bool(registry),
        "registry_sha256": sha256(args.registry) if args.registry.is_file() else None,
        "frozen_files_verified": frozen_files_verified,
        "completed_review_count": len(valid_reviews),
        "ml_publisher_reviewer_count": sum(
            review.get("publishes_in_machine_learning") is True
            for review in valid_reviews
        ),
        "distinct_valid_reviewer_count": len(valid_names),
        "unresolved_critical_count": unresolved_critical,
        "unresolved_major_count": unresolved_major,
        "reviewers_flagging_unaddressed_ltt_overlap": ltt_unaddressed,
        "reviewers_flagging_unaddressed_prompt_risk_control_overlap": (
            prompt_risk_unaddressed
        ),
        "response_ledger_complete": response_complete,
        "reviewer_identity_evidence_human_verified": identity_verified,
        "reviewer_ids": sorted(valid_ids),
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
