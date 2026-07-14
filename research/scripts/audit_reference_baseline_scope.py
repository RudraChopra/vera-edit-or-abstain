"""Audit whether VERA's reference/proxy baseline boundary is explicit.

This audit does not convert proxy stress tests into reference implementations.
It verifies the narrower claim needed for a protocol paper: official-code
MANCE++ is claim-grade on Waterbirds and Camelyon17, proxy erasure rows are
labeled as such, and the manuscripts do not claim universal state-of-the-art
erasure or reference parity for unrun methods.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
MAINTRACK_DIR = ROOT / "maintrack"

DEFAULT_JSON = ARTIFACT_DIR / "reference_baseline_scope_audit.json"
DEFAULT_MD = ARTIFACT_DIR / "reference_baseline_scope_audit.md"


@dataclass(frozen=True)
class Check:
    key: str
    status: str
    evidence: str


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    stat = path.stat()
    if stat.st_size > 0 and getattr(stat, "st_blocks", 1) == 0:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def load_text(path: Path) -> str:
    if not path.exists():
        return ""
    stat = path.stat()
    if stat.st_size > 0 and getattr(stat, "st_blocks", 1) == 0:
        return ""
    return path.read_text(encoding="utf-8")


def materialized(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    stat = path.stat()
    return not (stat.st_size > 0 and getattr(stat, "st_blocks", 1) == 0)


def passfail(key: str, passed: bool, evidence: str) -> Check:
    return Check(key=key, status="pass" if passed else "fail", evidence=evidence)


def collect_checks() -> list[Check]:
    baseline = load_json(ARTIFACT_DIR / "faro_baseline_fairness_report.json")
    mance = load_json(ARTIFACT_DIR / "mance_reference_status.json")
    claim = load_json(ARTIFACT_DIR / "claim_ledger_audit.json")
    upstream = load_json(ARTIFACT_DIR / "upstream_baseline_reference_inventory.json")
    hardening = load_text(MAINTRACK_DIR / "REFERENCE_BASELINE_HARDENING_PLAN.md")
    ledger = load_text(MAINTRACK_DIR / "CLAIM_LEDGER.md")
    manuscript = load_text(MAINTRACK_DIR / "faro_main.tex")
    iclr = load_text(
        MAINTRACK_DIR
        / "iclr2026_template"
        / "iclr2026"
        / "faro_iclr2026_draft.tex"
    )
    combined = "\n".join([hardening, ledger, manuscript, iclr])
    baseline_checks = baseline.get("checks", [])
    proxy_rows = [
        item
        for item in baseline_checks
        if isinstance(item, dict)
        and "proxy" in str(item.get("implementation_status", "")).lower()
    ]
    proxy_pass = proxy_rows and all(item.get("status") == "pass" for item in proxy_rows)
    waterbirds = mance.get("waterbirds", {}) if isinstance(mance, dict) else {}
    camelyon = mance.get("camelyon17", {}) if isinstance(mance, dict) else {}
    camelyon_receipt = ARTIFACT_DIR / "camelyon17_mancepp_reference_full_nocap_receipt.json"
    camelyon_scaling = load_json(ARTIFACT_DIR / "camelyon17_mance_scaling_feasibility.json")
    checks = [
        passfail(
            "baseline_fairness_ready",
            baseline.get("baseline_ready") is True and int(baseline.get("fail_count", 1)) == 0,
            f"baseline_ready={baseline.get('baseline_ready')}; fail_count={baseline.get('fail_count')}",
        ),
        passfail(
            "proxy_rows_labeled",
            bool(proxy_pass),
            f"proxy_rows={len(proxy_rows)}; proxy_statuses={[item.get('status') for item in proxy_rows]}",
        ),
        passfail(
            "mance_waterbirds_reference_ready",
            waterbirds.get("claim_grade_reference_row") is True
            and waterbirds.get("claim_grade_statistics") is True,
            (
                f"claim_grade_reference_row={waterbirds.get('claim_grade_reference_row')}; "
                f"claim_grade_statistics={waterbirds.get('claim_grade_statistics')}"
            ),
        ),
        passfail(
            "mance_camelyon_reference_ready",
            camelyon.get("claim_grade_reference_row") is True
            and materialized(camelyon_receipt)
            and "full no-cap" in str(camelyon.get("claim_boundary", "")).lower(),
            (
                f"claim_grade_reference_row={camelyon.get('claim_grade_reference_row')}; "
                f"full_nocap_receipt_materialized={materialized(camelyon_receipt)}; "
                f"claim_boundary={camelyon.get('claim_boundary')}"
            ),
        ),
        passfail(
            "mance_camelyon_scaling_documented",
            camelyon_scaling.get("full_no_cap_completed") is True
            and camelyon_scaling.get("compute_blocker") is False
            and camelyon_scaling.get("storage_blocker") is False
            and bool(camelyon_scaling.get("full_no_cap_observed_run"))
            and float(camelyon_scaling.get("linear_lower_bound_seconds", 0.0)) > 0.0
            and float(camelyon_scaling.get("recent_superlinear_estimate_seconds", 0.0)) > 0.0,
            (
                f"full_no_cap_completed={camelyon_scaling.get('full_no_cap_completed')}; "
                f"compute_blocker={camelyon_scaling.get('compute_blocker')}; "
                f"storage_blocker={camelyon_scaling.get('storage_blocker')}; "
                f"full_no_cap_observed_run={bool(camelyon_scaling.get('full_no_cap_observed_run'))}; "
                f"linear_lower_bound_seconds={camelyon_scaling.get('linear_lower_bound_seconds')}; "
                f"recent_superlinear_estimate_seconds={camelyon_scaling.get('recent_superlinear_estimate_seconds')}"
            ),
        ),
        passfail(
            "upstream_baseline_inventory_ready",
            upstream.get("inventory_ready") is True
            and int(upstream.get("fail_count", 1)) == 0
            and all(
                name in str(upstream)
                for name in ("MANCE++", "R-LACE", "TaCo", "LEACE", "SPLINCE/SPLICE")
            ),
            (
                f"inventory_ready={upstream.get('inventory_ready')}; "
                f"fail_count={upstream.get('fail_count')}; "
                "expected=MANCE++/R-LACE/TaCo/LEACE pinned plus SPLINCE boundary"
            ),
        ),
        passfail(
            "claim_ledger_ready",
            claim.get("claim_ledger_ready") is True and int(claim.get("fail_count", 1)) == 0,
            f"claim_ledger_ready={claim.get('claim_ledger_ready')}; fail_count={claim.get('fail_count')}",
        ),
        passfail(
            "no_universal_erasure_sota_claim",
            "not universal" in combined.lower()
            and "state-of-the-art erasure method" in combined.lower(),
            "manuscripts and claim ledger explicitly deny universal/SOTA erasure claims",
        ),
        passfail(
            "reference_parity_boundary_written",
            all(term in combined for term in ("SPLINCE", "R-LACE", "TaCo", "MANCE++"))
            and "reference parity" in combined.lower(),
            "close erasure baselines and reference-parity boundary are named in paper-facing docs",
        ),
    ]
    return checks


def write_markdown(path: Path, report: dict[str, object]) -> None:
    lines = [
        "# VERA Reference Baseline Scope Audit",
        "",
        f"Generated at UTC: `{report['created_at_utc']}`",
        f"Reference scope ready: `{report['reference_scope_ready']}`",
        f"Universal erasure SOTA claim allowed: `{report['universal_erasure_sota_claim_allowed']}`",
        "",
        "| Status | Check | Evidence |",
        "| --- | --- | --- |",
    ]
    for check in report["checks"]:
        lines.append(f"| {check['status']} | `{check['key']}` | {check['evidence']} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MD)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    checks = collect_checks()
    fail_count = sum(check.status == "fail" for check in checks)
    pass_count = sum(check.status == "pass" for check in checks)
    report = {
        "name": "VERA reference baseline scope audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_scope_ready": fail_count == 0,
        "universal_erasure_sota_claim_allowed": False,
        "full_reference_parity_claim_allowed": False,
        "remaining_reference_parity_gaps": [
            "SPLINCE/SPLICE exact upstream reference receipts",
            "R-LACE exact upstream reference receipts",
            "TaCo exact upstream reference receipts",
            "LEACE exact upstream reference receipts",
        ],
        "pass_count": pass_count,
        "fail_count": fail_count,
        "checks": [asdict(check) for check in checks],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.markdown_out, report)
    print("VERA reference baseline scope audit complete")
    print(f"reference_scope_ready={str(report['reference_scope_ready']).lower()}")
    print(f"fail_count={fail_count}")
    print(f"report={args.json_out}")
    return 0 if args.no_fail or fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
