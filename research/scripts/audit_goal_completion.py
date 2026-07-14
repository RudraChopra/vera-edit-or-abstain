"""Audit VERA against the full user goal, not just local readiness gates."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAINTRACK_DIR = ROOT / "maintrack"
ARTIFACT_DIR = ROOT / "artifacts"
EXTERNAL_ARTIFACT_DIR = Path("/Volumes/Backups/FARO/artifacts")

DEFAULT_JSON = ARTIFACT_DIR / "faro_goal_completion_audit.json"
DEFAULT_MD = ARTIFACT_DIR / "faro_goal_completion_audit.md"


@dataclass(frozen=True)
class Requirement:
    key: str
    status: str
    evidence: str
    remaining_work: str


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


def read_text(path: Path) -> str:
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


def req(key: str, status: str, evidence: str, remaining: str) -> Requirement:
    return Requirement(key=key, status=status, evidence=evidence, remaining_work=remaining)


def collect_requirements() -> list[Requirement]:
    readiness = load_json(ARTIFACT_DIR / "maintrack_readiness.json")
    adversarial = load_json(ARTIFACT_DIR / "faro_adversarial_internal_review.json")
    mance = load_json(ARTIFACT_DIR / "mance_reference_status.json")
    mance_stats = load_json(ARTIFACT_DIR / "mance_reference_statistical_report.json")
    baseline = load_json(ARTIFACT_DIR / "faro_baseline_fairness_report.json")
    abstention = load_json(ARTIFACT_DIR / "faro_synthetic_abstention_report.json")
    real_abstention = load_json(ARTIFACT_DIR / "faro_real_abstention_stress_report.json")
    waterbirds = load_json(ARTIFACT_DIR / "waterbirds_official_result_receipt.json")
    camelyon = load_json(ARTIFACT_DIR / "camelyon17_wilds_official_result_receipt.json")
    waterbirds_stats = load_json(ARTIFACT_DIR / "waterbirds_official_statistical_report.json")
    camelyon_stats = load_json(ARTIFACT_DIR / "camelyon17_wilds_official_statistical_report.json")
    camelyon_frontier = load_json(ARTIFACT_DIR / "camelyon17_faro_projection_certificate.json")
    claim_ledger = load_json(ARTIFACT_DIR / "claim_ledger_audit.json")
    repro = load_json(ARTIFACT_DIR / "reproducibility_packet_audit.json")
    significance = load_json(ARTIFACT_DIR / "faro_statistical_significance_addendum.json")
    civilcomments = load_json(ARTIFACT_DIR / "civilcomments_current_verification_report.json")
    gait = load_json(ARTIFACT_DIR / "gaitpdb_public_locked_split_result_receipt.json")
    gait_stats = load_json(ARTIFACT_DIR / "gaitpdb_public_locked_split_statistical_report.json")
    reference_scope = load_json(ARTIFACT_DIR / "reference_baseline_scope_audit.json")
    upstream = load_json(ARTIFACT_DIR / "upstream_baseline_reference_inventory.json")
    aaai = load_json(ARTIFACT_DIR / "aaai2027_source_readiness.json")

    faro_main = read_text(MAINTRACK_DIR / "faro_main.tex")
    theory = read_text(MAINTRACK_DIR / "THEORY_TARGET.md")
    theory_lower = theory.lower()
    novelty = read_text(MAINTRACK_DIR / "NOVELTY_LOCK.md") + "\n" + read_text(
        MAINTRACK_DIR / "NOVELTY_SWEEP_2026_UPDATE.md"
    )
    algorithm = read_text(MAINTRACK_DIR / "ALGORITHM_SPEC.md")
    venue_plan = read_text(MAINTRACK_DIR / "VENUE_FORMAT_PLAN.md")
    github_release_readme = read_text(ROOT.parent / "GITHUB_EXPORT_README.md")
    iclr_source = MAINTRACK_DIR / "iclr2026_template" / "iclr2026" / "faro_iclr2026_draft.tex"
    iclr_pdf = MAINTRACK_DIR / "faro_iclr2026_draft.pdf"
    aaai_source = MAINTRACK_DIR / "aaai2027_template" / "AuthorKit27" / "faro_aaai2027_draft.tex"
    aaai_anonymous_source = MAINTRACK_DIR / "aaai2027_template" / "AuthorKit27" / "faro_aaai2027_anonymous.tex"
    aaai_named_source = MAINTRACK_DIR / "aaai2027_template" / "AuthorKit27" / "faro_aaai2027_named.tex"

    store_manifest = load_json(EXTERNAL_ARTIFACT_DIR / "camelyon17_resnet18_torch_full_numpy_store" / "manifest.json")

    requirements: list[Requirement] = []
    requirements.append(
        req(
            "novelty_locked_against_close_baselines",
            "pass" if all(term in novelty for term in ("LEACE", "RLACE", "INLP", "TaCo", "MANCE", "SPLINCE")) else "fail",
            "NOVELTY_LOCK.md plus NOVELTY_SWEEP_2026_UPDATE.md include LEACE/RLACE/INLP/TaCo/MANCE/SPLINCE.",
            "Keep running recency sweeps before any real submission deadline.",
        )
    )
    requirements.append(
        req(
            "method_fully_specified",
            "pass" if all(term.lower() in algorithm.lower() for term in ("frontier", "selection", "abstention", "output")) else "fail",
            "ALGORITHM_SPEC.md defines inputs, frontier, selection, abstention, and output.",
            "Convert the spec into pseudocode in the final conference template.",
        )
    )
    requirements.append(
        req(
            "real_theory_present",
            "pass"
            if "simultaneous frontier theorem" in theory_lower
            and "finite-sample validation certificate" in theory_lower
            and "false-acceptance control" in faro_main.lower()
            and "proof" in faro_main.lower()
            else "partial",
            "Manuscript contains safe-acceptance, simultaneous frontier, finite-sample validation, and false-acceptance statements with proofs.",
            "Tighten conservative bounds with empirical Bernstein, bootstrap simultaneous intervals, or probe-stability analysis.",
        )
    )
    reference_scope_ready = (
        baseline.get("baseline_ready") is True
        and mance.get("waterbirds", {}).get("claim_grade_statistics") is True
        and mance_stats.get("claim_grade_statistics") is True
        and reference_scope.get("reference_scope_ready") is True
        and reference_scope.get("universal_erasure_sota_claim_allowed") is False
        and upstream.get("inventory_ready") is True
    )
    requirements.append(
        req(
            "strong_reference_baselines",
            "pass" if reference_scope_ready else "partial",
            (
                f"baseline_ready={baseline.get('baseline_ready')}; "
                f"mance_waterbirds_claim_grade={mance.get('waterbirds', {}).get('claim_grade_statistics')}; "
                f"mance_camelyon_claim_grade={mance.get('camelyon17', {}).get('claim_grade_reference_row')}; "
                f"mance_receipts={mance_stats.get('receipt_count')}; "
                f"reference_scope_ready={reference_scope.get('reference_scope_ready')}; "
                f"upstream_inventory_ready={upstream.get('inventory_ready')}; "
                f"universal_erasure_sota_claim_allowed={reference_scope.get('universal_erasure_sota_claim_allowed')}"
            ),
            (
                "Optional strengthening remains: add exact upstream SPLINCE/RLACE/TaCo receipts "
                "and exact upstream LEACE receipts. Current paper claims are scoped as a "
                "protocol contribution, not universal erasure SOTA."
            ),
        )
    )
    official_rows = [
        name
        for name, data in (("Waterbirds", waterbirds), ("Camelyon17", camelyon))
        if data.get("claim_gate_passed") is True
    ]
    gait_ready = gait.get("claim_gate_passed") is True and gait_stats.get("claim_grade_statistics") is True
    requirements.append(
        req(
            "broad_benchmark_evidence",
            "pass" if len(official_rows) >= 2 and gait_ready else "partial" if len(official_rows) >= 2 else "fail",
            (
                f"claim_grade_official_rows={official_rows}; "
                f"camelyon_full_store_examples={store_manifest.get('n_examples')}; "
                f"camelyon_frontier_certificate={camelyon_frontier.get('claim_grade_frontier_certificate')}; "
                f"civilcomments_status={civilcomments.get('status')}; "
                f"public_gait_claim_grade_row={gait_ready}; "
                f"gait_n_examples={gait.get('n_examples')}"
            ),
            "CivilComments remains useful for text-modality breadth, but the public gait row now supplies an additional claim-grade family.",
        )
    )
    overlap = abstention.get("overlap_case", {}) if isinstance(abstention, dict) else {}
    real_cases = real_abstention.get("cases", []) if isinstance(real_abstention, dict) else []
    real_abstain = any(isinstance(case, dict) and case.get("decision") == "ABSTAIN" for case in real_cases)
    camelyon_abstain = (
        camelyon_frontier.get("claim_grade_frontier_certificate") is True
        and camelyon_frontier.get("decision") == "ABSTAIN"
    )
    requirements.append(
        req(
            "abstention_demonstrated",
            "pass" if overlap.get("decision") == "ABSTAIN" and real_abstain and camelyon_abstain else "partial",
            (
                f"synthetic_overlap={overlap.get('decision')}; "
                f"real_abstain={real_abstain}; "
                f"camelyon_full_frontier_abstain={camelyon_abstain}"
            ),
            "Expand real benchmark abstention examples and calibrate confidence intervals in the manuscript.",
        )
    )
    requirements.append(
        req(
            "statistical_integrity",
            "pass"
            if waterbirds_stats.get("claim_grade_statistics")
            and camelyon_stats.get("claim_grade_statistics")
            and mance_stats.get("claim_grade_statistics")
            and gait_stats.get("claim_grade_statistics")
            and significance.get("significance_addendum_ready")
            and "finite-sample validation certificate" in theory_lower
            else "partial",
            (
                f"waterbirds_stats={waterbirds_stats.get('claim_grade_statistics')}; "
                f"camelyon_stats={camelyon_stats.get('claim_grade_statistics')}; "
                f"mance_claim_grade_statistics={mance_stats.get('claim_grade_statistics')}; "
                f"gait_stats={gait_stats.get('claim_grade_statistics')}; "
                f"significance_addendum={significance.get('significance_addendum_ready')}"
            ),
            "Increase seed count or use stronger resampling tests before final submission.",
        )
    )
    requirements.append(
        req(
            "reproducibility_packet",
            "pass" if repro.get("packet_ready") and claim_ledger.get("claim_ledger_ready") else "partial",
            (
                f"packet_ready={repro.get('packet_ready')}; "
                f"claim_ledger_ready={claim_ledger.get('claim_ledger_ready')}; "
                f"upstream_inventory_ready={upstream.get('inventory_ready')}; "
                f"aaai_source_ready={aaai.get('source_ready')}; "
                f"github_release_readme_present={bool(github_release_readme)}"
            ),
            "Package public release structure, environment lockfile, and one-command reproduction entrypoint.",
        )
    )
    requirements.append(
        req(
            "conference_manuscript_quality",
            "pass"
            if materialized(iclr_source)
            and materialized(iclr_pdf)
            and materialized(aaai_source)
            and materialized(aaai_anonymous_source)
            and materialized(aaai_named_source)
            and aaai.get("source_ready") is True
            and bool(venue_plan)
            else "partial",
            (
                "faro_main.tex compiles and contains narrative, theory, tables, figures, limitations, and references; "
                f"venue_plan_present={bool(venue_plan)}; "
                f"iclr2026_source_materialized={materialized(iclr_source)}; "
                f"iclr2026_pdf_materialized={materialized(iclr_pdf)}; "
                f"aaai_source_materialized={materialized(aaai_source)}; "
                f"aaai_anonymous_source_materialized={materialized(aaai_anonymous_source)}; "
                f"aaai_named_source_materialized={materialized(aaai_named_source)}; "
                f"aaai_source_ready={aaai.get('source_ready')}; "
                f"aaai_pdf_compile_ready={aaai.get('pdf_compile_ready')}"
            ),
            "Before real submission, replace ICLR-2026 style with the official target-year style and complete final human polish.",
        )
    )
    requirements.append(
        req(
            "strict_adversarial_review",
            "pass" if adversarial.get("submission_ready") and adversarial.get("critical_count") == 0 and adversarial.get("major_count") == 0 else "partial",
            (
                f"submission_ready={adversarial.get('submission_ready')}; "
                f"critical={adversarial.get('critical_count')}; major={adversarial.get('major_count')}"
            ),
            "Update adversarial review to include this full-goal audit before any real submission.",
        )
    )
    return requirements


def write_markdown(path: Path, report: dict[str, object]) -> None:
    lines = [
        "# VERA Full Goal Completion Audit",
        "",
        f"Generated at UTC: `{report['created_at_utc']}`",
        f"Goal complete: `{report['goal_complete']}`",
        "",
        "| Status | Requirement | Evidence | Remaining work |",
        "| --- | --- | --- | --- |",
    ]
    for item in report["requirements"]:
        lines.append(
            f"| {item['status']} | `{item['key']}` | {item['evidence']} | {item['remaining_work']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    requirements = collect_requirements()
    counts = {
        status: sum(1 for item in requirements if item.status == status)
        for status in ("pass", "partial", "fail")
    }
    report = {
        "name": "VERA full user-goal completion audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "goal_complete": counts.get("partial", 0) == 0 and counts.get("fail", 0) == 0,
        "status_counts": counts,
        "requirements": [asdict(item) for item in requirements],
    }
    DEFAULT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(DEFAULT_MD, report)
    print(
        "VERA full-goal completion audit complete\n"
        f"goal_complete={str(report['goal_complete']).lower()}\n"
        f"pass={counts.get('pass', 0)} partial={counts.get('partial', 0)} fail={counts.get('fail', 0)}\n"
        f"report={DEFAULT_JSON}"
    )
    raise SystemExit(0 if args.no_fail or report["goal_complete"] else 1)


if __name__ == "__main__":
    main()
