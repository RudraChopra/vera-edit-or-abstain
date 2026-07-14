"""Audit FARO claim ledger support and forbidden manuscript claims."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
LEDGER_PATH = ROOT / "configs" / "faro_claim_ledger.json"
CHECKLIST_PATH = ROOT / "maintrack" / "CLAIM_LEDGER.md"
DEFAULT_JSON = ARTIFACT_DIR / "claim_ledger_audit.json"
DEFAULT_MD = ARTIFACT_DIR / "claim_ledger_audit.md"
DEFAULT_CSV = ARTIFACT_DIR / "claim_ledger_audit.csv"


@dataclass(frozen=True)
class Check:
    key: str
    status: str
    evidence: str
    requirement: str
    next_step: str


def status(condition: bool) -> str:
    return "pass" if condition else "fail"


def bool_word(value: bool) -> str:
    return "yes" if value else "no"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def resolve(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT.parent / candidate


def text_for_paths(paths: list[str]) -> str:
    chunks: list[str] = []
    for raw in paths:
        path = resolve(raw)
        if path.exists():
            chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    ledger = load_json(LEDGER_PATH)
    allowed = ledger.get("allowed_claims", [])
    forbidden = ledger.get("forbidden_claims", [])
    audited_paths = ledger.get("audited_manuscript_paths", [])
    if not isinstance(allowed, list):
        allowed = []
    if not isinstance(forbidden, list):
        forbidden = []
    if not isinstance(audited_paths, list):
        audited_paths = []

    allowed_with_evidence = [
        claim
        for claim in allowed
        if isinstance(claim, dict)
        and claim.get("manuscript_status") == "allowed"
        and isinstance(claim.get("evidence"), list)
        and claim["evidence"]
        and all(isinstance(path, str) and resolve(path).exists() for path in claim["evidence"])
    ]
    forbidden_patterns: list[str] = []
    for claim in forbidden:
        if not isinstance(claim, dict):
            continue
        patterns = claim.get("forbidden_exact_patterns", [])
        if isinstance(patterns, list):
            forbidden_patterns.extend(str(pattern) for pattern in patterns if pattern)

    manuscript_text = text_for_paths([str(path) for path in audited_paths])
    forbidden_hits = [
        pattern for pattern in forbidden_patterns if pattern.lower() in manuscript_text.lower()
    ]
    audited_paths_exist = all(
        isinstance(path, str) and resolve(path).exists()
        for path in audited_paths
    )
    checklist_text = CHECKLIST_PATH.read_text(encoding="utf-8") if CHECKLIST_PATH.exists() else ""

    checks = [
        Check(
            key="claim_ledger_present",
            status=status(bool(ledger)),
            evidence=f"path={LEDGER_PATH}; exists={bool_word(LEDGER_PATH.exists())}",
            requirement="A machine-readable claim ledger must exist.",
            next_step="Create `research/configs/faro_claim_ledger.json`.",
        ),
        Check(
            key="claim_ledger_markdown_present",
            status=status(
                CHECKLIST_PATH.exists()
                and "Allowed Claims" in checklist_text
                and "Forbidden Claims" in checklist_text
            ),
            evidence=f"path={CHECKLIST_PATH}; exists={bool_word(CHECKLIST_PATH.exists())}",
            requirement="A human-readable claim ledger must exist.",
            next_step="Create or update `research/maintrack/CLAIM_LEDGER.md`.",
        ),
        Check(
            key="allowed_claims_have_evidence",
            status=status(len(allowed_with_evidence) == len(allowed) and len(allowed) >= 8),
            evidence=(
                f"allowed_claims={len(allowed)}; "
                f"with_existing_evidence={len(allowed_with_evidence)}"
            ),
            requirement="Every allowed claim must point to existing evidence artifacts.",
            next_step="Add evidence paths for every allowed claim or remove unsupported claims.",
        ),
        Check(
            key="forbidden_claims_registered",
            status=status(len(forbidden) >= 5 and len(forbidden_patterns) >= 5),
            evidence=f"forbidden_claims={len(forbidden)}; forbidden_patterns={len(forbidden_patterns)}",
            requirement="The ledger must register the most dangerous unsupported claims.",
            next_step="Add missing forbidden claims and exact manuscript patterns.",
        ),
        Check(
            key="audited_manuscript_paths_exist",
            status=status(audited_paths_exist and len(audited_paths) >= 4),
            evidence=f"audited_paths={audited_paths}; exist={bool_word(audited_paths_exist)}",
            requirement="The claim audit must scan the submission-facing manuscript files.",
            next_step="Add or repair audited manuscript paths in the claim ledger.",
        ),
        Check(
            key="forbidden_claims_absent_from_manuscript",
            status=status(not forbidden_hits),
            evidence=f"forbidden_hits={forbidden_hits}",
            requirement="Submission-facing manuscript files must not assert unsupported claims.",
            next_step="Remove or reframe the forbidden claims from the manuscript files.",
        ),
    ]

    fail_count = sum(1 for check in checks if check.status == "fail")
    report = {
        "name": "FARO Paper A claim ledger audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_ledger_ready": fail_count == 0,
        "pass_count": len(checks) - fail_count,
        "fail_count": fail_count,
        "checks": [asdict(check) for check in checks],
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with args.csv_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key", "status", "evidence", "requirement", "next_step"])
        writer.writeheader()
        for check in checks:
            writer.writerow(asdict(check))

    lines = [
        "# FARO Claim Ledger Audit",
        "",
        f"Generated at UTC: `{report['created_at_utc']}`",
        "",
        f"- `claim_ledger_ready`: {bool_word(report['claim_ledger_ready'])}",
        f"- `pass_count`: {report['pass_count']}",
        f"- `fail_count`: {report['fail_count']}",
        "",
        "| Check | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for check in checks:
        lines.append(f"| `{check.key}` | {check.status} | {check.evidence} |")
    args.md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("FARO claim ledger audit complete")
    print(f"claim_ledger_ready={str(report['claim_ledger_ready']).lower()}")
    print(f"fail_count={fail_count}")
    print(f"report={args.json_out}")
    if fail_count and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
