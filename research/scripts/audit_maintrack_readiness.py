"""Audit whether the FARO main-track package has its required evidence.

The audit is intentionally artifact-driven: it checks the paper-facing specs,
claim-grade benchmark receipts, statistical reports, reproducibility packet,
baseline audit, MANCE reference evidence, and adversarial review result.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAINTRACK_DIR = ROOT / "maintrack"
ARTIFACT_DIR = ROOT / "artifacts"
EXTERNAL_ARTIFACT_DIR = Path("/Volumes/Backups/FARO/artifacts")

DEFAULT_JSON = ARTIFACT_DIR / "maintrack_readiness.json"
DEFAULT_MD = ARTIFACT_DIR / "maintrack_readiness.md"
DEFAULT_CSV = ARTIFACT_DIR / "maintrack_readiness.csv"


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


def file_text(path: Path) -> str:
    if not path.exists():
        return ""
    stat = path.stat()
    if stat.st_size > 0 and getattr(stat, "st_blocks", 1) == 0:
        return ""
    return path.read_text(encoding="utf-8")


def passfail(key: str, passed: bool, evidence: str) -> Check:
    return Check(key=key, status="pass" if passed else "fail", evidence=evidence)


def doc_check(relative: str, terms: tuple[str, ...]) -> Check:
    path = MAINTRACK_DIR / relative
    text = file_text(path)
    lowered = text.lower()
    missing = [term for term in terms if term.lower() not in lowered]
    return passfail(
        key=f"maintrack_{path.stem.lower()}_present",
        passed=path.exists() and not missing,
        evidence=f"exists={'yes' if path.exists() else 'no'}; missing_terms={missing}",
    )


def artifact_bool_check(path: Path, key: str, flag: str, label: str | None = None) -> Check:
    data = load_json(path)
    passed = bool(data.get(flag))
    return passfail(
        key=key,
        passed=passed,
        evidence=(
            f"{label or flag}={'yes' if passed else 'no'}; "
            f"report_exists={'yes' if bool(data) else 'no'}"
        ),
    )


def official_benchmark_checks() -> list[Check]:
    receipts = [
        ARTIFACT_DIR / "waterbirds_official_result_receipt.json",
        ARTIFACT_DIR / "camelyon17_wilds_official_result_receipt.json",
        ARTIFACT_DIR / "camelyon17_numpy_store_official_result_receipt.json",
        ARTIFACT_DIR / "camelyon17_full_numpy_store_result_receipt.json",
    ]
    statistics = [
        ARTIFACT_DIR / "waterbirds_official_statistical_report.json",
        ARTIFACT_DIR / "camelyon17_wilds_official_statistical_report.json",
        ARTIFACT_DIR / "camelyon17_numpy_store_official_statistical_report.json",
        ARTIFACT_DIR / "camelyon17_full_numpy_store_statistical_report.json",
    ]
    ready_receipts = [path for path in receipts if load_json(path).get("claim_gate_passed") is True]
    ready_stats = [path for path in statistics if load_json(path).get("claim_grade_statistics") is True]

    families = set()
    high_stakes = False
    for path in ready_receipts:
        data = load_json(path)
        text = json.dumps(data).lower()
        if "waterbirds" in text:
            families.add("spurious correlation / background shift")
        if "camelyon" in text:
            families.add("medical / hospital shift")
            high_stakes = True

    return [
        passfail(
            "official_result_count",
            len(ready_receipts) >= 2,
            f"official_ready_rows={len(ready_receipts)}; required=2; rows={[p.name for p in ready_receipts]}",
        ),
        passfail(
            "official_statistics_count",
            len(ready_stats) >= 2,
            f"claim_grade_statistical_reports={len(ready_stats)}; required=2; reports={[p.name for p in ready_stats]}",
        ),
        passfail(
            "official_family_count",
            len(families) >= 2,
            f"families={sorted(families)}; required=2",
        ),
        passfail(
            "high_stakes_official_result",
            high_stakes,
            f"high_stakes_ready={'yes' if high_stakes else 'no'}",
        ),
    ]


def camelyon_store_checks() -> list[Check]:
    store = EXTERNAL_ARTIFACT_DIR / "camelyon17_resnet18_torch_full_numpy_store"
    manifest = load_json(store / "manifest.json")
    n_examples = int(manifest.get("n_examples", 0) or 0)
    feature_count = int(manifest.get("feature_count", 0) or 0)
    arrays = manifest.get("arrays", {})
    array_names = arrays.values() if isinstance(arrays, dict) else []
    arrays_exist = all((store / str(name)).exists() for name in array_names)
    return [
        passfail(
            "camelyon17_full_numpy_store_present",
            n_examples >= 455_000 and feature_count == 512 and arrays_exist,
            (
                f"manifest_exists={'yes' if bool(manifest) else 'no'}; "
                f"n_examples={n_examples}; feature_count={feature_count}; arrays_exist={arrays_exist}"
            ),
        )
    ]


def mance_reference_check() -> Check:
    paths = [
        path
        for path in ARTIFACT_DIR.glob("*mance*")
        if "style" not in path.name.lower() and path.name.endswith("_receipt.json")
    ]
    claim_grade = []
    diagnostics = []
    for path in paths:
        data = load_json(path)
        if data.get("claim_grade_reference_row") is True:
            claim_grade.append(path.name)
        elif data:
            diagnostics.append(path.name)
    return passfail(
        "mance_reference_claim_grade_present",
        bool(claim_grade),
        f"claim_grade={claim_grade}; diagnostics={diagnostics[:5]}",
    )


def abstention_check() -> Check:
    path = ARTIFACT_DIR / "faro_synthetic_abstention_report.json"
    data = load_json(path)
    overlap = data.get("overlap_case", {}) if isinstance(data, dict) else {}
    nonoverlap = data.get("nonoverlap_case", {}) if isinstance(data, dict) else {}
    passed = (
        isinstance(overlap, dict)
        and overlap.get("decision") == "ABSTAIN"
        and isinstance(nonoverlap, dict)
        and nonoverlap.get("decision") == "EDIT"
    )
    return passfail(
        "faro_synthetic_abstention_certificate_passed",
        passed,
        (
            f"report_exists={'yes' if bool(data) else 'no'}; "
            f"overlap_decision={overlap.get('decision') if isinstance(overlap, dict) else None}; "
            f"nonoverlap_decision={nonoverlap.get('decision') if isinstance(nonoverlap, dict) else None}"
        ),
    )


def camelyon_frontier_certificate_check() -> Check:
    path = ARTIFACT_DIR / "camelyon17_faro_projection_certificate.json"
    data = load_json(path)
    selected = data.get("selected_candidate", {}) if isinstance(data, dict) else {}
    selected = selected if isinstance(selected, dict) else {}
    passed = (
        bool(data.get("claim_grade_frontier_certificate")) is True
        and data.get("decision") in {"EDIT", "ABSTAIN"}
        and bool(data.get("selection_uses_external_metrics")) is False
        and int(data.get("n_examples", 0) or 0) >= 455_000
    )
    return passfail(
        "camelyon17_full_frontier_certificate_present",
        passed,
        (
            f"report_exists={'yes' if bool(data) else 'no'}; "
            f"decision={data.get('decision')}; "
            f"safe_candidate_count={data.get('safe_candidate_count')}; "
            f"selected_strength={selected.get('strength')}; "
            f"external_source_interpretable={data.get('external_source_leakage_interpretable')}"
        ),
    )


def collect_checks() -> list[Check]:
    checks: list[Check] = [
        doc_check("README.md", ("FARO", "main-track")),
        doc_check("PROJECT_SPEC.md", ("FARO", "claim", "benchmark")),
        doc_check("PAPER_A_LOCK.md", ("Paper A", "FARO")),
        doc_check("NOVELTY_LOCK.md", ("LEACE", "RLACE", "INLP", "TaCo")),
        doc_check("NOVELTY_SWEEP_2026_UPDATE.md", ("MANCE", "SPLINCE")),
        doc_check("ALGORITHM_SPEC.md", ("frontier", "abstention")),
        doc_check("THEORY_TARGET.md", ("theorem", "abstention")),
        doc_check("BASELINE_PROTOCOL.md", ("LEACE", "RLACE", "TaCo", "MANCE")),
        doc_check("ABSTENTION_PROTOCOL.md", ("ABSTAIN", "calibration")),
        doc_check("STATISTICAL_INTEGRITY.md", ("seed", "confidence")),
        doc_check("REPRODUCIBILITY_CHECKLIST.md", ("artifact", "seed")),
        doc_check("CLAIM_LEDGER.md", ("claim", "evidence")),
        doc_check("CODE_AVAILABILITY.md", ("github", "anonymous")),
        doc_check("CODE_AVAILABILITY_ANONYMOUS.md", ("anonymous", "supplementary")),
        doc_check(
            "faro_main.tex",
            (
                "FARO",
                "Camelyon17",
                "Waterbirds",
                "False-acceptance control",
                "strongest reviewer objection",
                "Code Availability",
            ),
        ),
    ]
    checks.extend(official_benchmark_checks())
    checks.extend(camelyon_store_checks())
    checks.append(mance_reference_check())
    checks.extend(
        [
            abstention_check(),
            camelyon_frontier_certificate_check(),
            artifact_bool_check(
                ARTIFACT_DIR / "reproducibility_packet_audit.json",
                "reproducibility_packet_passed",
                "packet_ready",
            ),
            artifact_bool_check(
                ARTIFACT_DIR / "claim_ledger_audit.json",
                "claim_ledger_audit_passed",
                "claim_ledger_ready",
            ),
            artifact_bool_check(
                ARTIFACT_DIR / "faro_baseline_fairness_report.json",
                "baseline_fairness_audit_passed",
                "baseline_ready",
            ),
            artifact_bool_check(
                ARTIFACT_DIR / "upstream_baseline_reference_inventory.json",
                "upstream_baseline_inventory_passed",
                "inventory_ready",
            ),
            artifact_bool_check(
                ARTIFACT_DIR / "aaai2027_source_readiness.json",
                "aaai2027_source_readiness_passed",
                "source_ready",
            ),
            artifact_bool_check(
                ARTIFACT_DIR / "faro_adversarial_internal_review.json",
                "adversarial_internal_review_passed",
                "submission_ready",
            ),
        ]
    )
    return checks


def write_markdown(path: Path, report: dict[str, object]) -> None:
    lines = [
        "# FARO Main-Track Readiness Audit",
        "",
        f"Generated at UTC: `{report['created_at_utc']}`",
        f"Main-track ready: `{report['maintrack_ready']}`",
        "",
        "| Status | Check | Evidence |",
        "| --- | --- | --- |",
    ]
    for check in report["checks"]:
        lines.append(f"| {check['status']} | `{check['key']}` | {check['evidence']} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, report: dict[str, object]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["status", "key", "evidence"])
        writer.writeheader()
        for check in report["checks"]:
            writer.writerow(
                {
                    "status": check["status"],
                    "key": check["key"],
                    "evidence": check["evidence"],
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MD)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    checks = collect_checks()
    fail_count = sum(check.status == "fail" for check in checks)
    pass_count = sum(check.status == "pass" for check in checks)
    report = {
        "name": "FARO main-track readiness audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "maintrack_ready": fail_count == 0,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "checks": [asdict(check) for check in checks],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.markdown_out, report)
    write_csv(args.csv_out, report)
    print("FARO main-track readiness audit complete")
    print(f"maintrack_ready={str(report['maintrack_ready']).lower()}")
    print(f"pass_count={pass_count}")
    print(f"fail_count={fail_count}")
    print(f"report={args.json_out.resolve()}")
    return 0 if args.no_fail or fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
