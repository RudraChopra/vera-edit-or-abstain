"""Run a strict internal reviewer audit for the VERA main-track package."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
MAINTRACK_DIR = ROOT / "maintrack"

DEFAULT_JSON = ARTIFACT_DIR / "faro_adversarial_internal_review.json"
DEFAULT_MD = ARTIFACT_DIR / "faro_adversarial_internal_review.md"
SELF_GOAL_KEY = "strict_adversarial_review"


@dataclass(frozen=True)
class Finding:
    severity: str
    title: str
    evidence: str
    reviewer_attack: str
    required_fix: str


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


def collect_findings() -> list[Finding]:
    findings: list[Finding] = []
    readiness = load_json(ARTIFACT_DIR / "maintrack_readiness.json")
    baseline = load_json(ARTIFACT_DIR / "faro_baseline_fairness_report.json")
    abstention = load_json(ARTIFACT_DIR / "faro_synthetic_abstention_report.json")
    real_abstention = load_json(ARTIFACT_DIR / "faro_real_abstention_stress_report.json")
    preflight = load_json(ARTIFACT_DIR / "camelyon17_full_store_preflight_report.json")
    full_goal = load_json(ARTIFACT_DIR / "faro_goal_completion_audit.json")

    if int(readiness.get("fail_count", 999)) > 0:
        blockers = [
            item.get("key", "")
            for item in readiness.get("checks", [])
            if isinstance(item, dict) and item.get("status") == "fail"
        ]
        non_self_blockers = [
            blocker for blocker in blockers if blocker != "adversarial_internal_review_passed"
        ]
        if non_self_blockers:
            findings.append(
                Finding(
                    severity="critical",
                    title="Main-track readiness audit still fails",
                    evidence=f"fail_count={readiness.get('fail_count')}; blockers={blockers}",
                    reviewer_attack="The paper is being submitted before its own gates are satisfied.",
                    required_fix="Clear every non-adversarial readiness blocker or explicitly move the manuscript to a workshop/preliminary track.",
                )
            )

    if int(baseline.get("fail_count", 999)) > 0:
        missing = [
            item.get("baseline_id", "")
            for item in baseline.get("checks", [])
            if isinstance(item, dict) and item.get("status") == "fail"
        ]
        findings.append(
            Finding(
                severity="critical",
                title="Baseline package is not yet reviewer-proof",
                evidence=f"fail_count={baseline.get('fail_count')}; failing_baselines={missing}",
                reviewer_attack="VERA is compared against weakened or incomplete erasure baselines.",
                required_fix="Implement SPLINCE/SPLICE, R-LACE, TaCo, and MANCE under matched conditions, or scope the claims so omitted baselines are not required.",
            )
        )

    ready_rows = [
        item
        for item in readiness.get("checks", [])
        if isinstance(item, dict) and item.get("key") == "official_result_count"
    ]
    ready_evidence = str(ready_rows[0].get("evidence", "")) if ready_rows else ""
    ready_count_match = re.search(r"official_ready_rows=(\d+)", ready_evidence)
    ready_count = int(ready_count_match.group(1)) if ready_count_match else 0
    if ready_rows and ready_count < 2:
        findings.append(
            Finding(
                severity="critical",
                title="Evidence is still single-family",
                evidence=ready_evidence,
                reviewer_attack="The method may be overfit to one modality or benchmark family.",
                required_fix="Finish Camelyon17-WILDS or another official second-family result with five seeds and paired statistics.",
            )
        )

    preflight_gate_rows = [
        item
        for item in readiness.get("checks", [])
        if isinstance(item, dict)
        and item.get("key") == "camelyon17_full_store_launch_preflight_passed"
    ]
    tar_stream_preflight_passed = bool(
        preflight_gate_rows and preflight_gate_rows[0].get("status") == "pass"
    )
    if preflight and not bool(preflight.get("ready_to_launch_full_store")) and not tar_stream_preflight_passed:
        image_scan = preflight.get("image_materialization_scan", {})
        image_scan = image_scan if isinstance(image_scan, dict) else {}
        findings.append(
            Finding(
                severity="major",
                title="Medical benchmark export is blocked by local preflight",
                evidence=(
                    f"blocking_checks={preflight.get('blocking_checks')}; "
                    f"free={preflight.get('disk_free_bytes')}; "
                    f"required={preflight.get('required_free_bytes')}; "
                    f"remaining_output={preflight.get('remaining_output_bytes')}; "
                    f"estimated_materialization={preflight.get('estimated_materialization_bytes')}; "
                    f"dataless_count={image_scan.get('dataless_count')}; "
                    f"projected_free_after={preflight.get('projected_free_after_bytes')}"
                ),
                reviewer_attack="The high-stakes benchmark plan is aspirational rather than executed.",
                required_fix=(
                    "Free enough disk for image hydration plus embeddings, materialize/download the local "
                    "Camelyon PNG placeholders, rerun preflight, then launch the full store."
                ),
            )
        )

    overlap = abstention.get("overlap_case", {}) if isinstance(abstention, dict) else {}
    real_cases = real_abstention.get("cases", []) if isinstance(real_abstention, dict) else []
    has_real_abstention = bool(real_abstention.get("real_benchmark")) and any(
        isinstance(case, dict) and case.get("decision") == "ABSTAIN"
        for case in real_cases
    )
    if overlap.get("decision") == "ABSTAIN" and not has_real_abstention:
        findings.append(
            Finding(
                severity="major",
                title="Abstention is demonstrated only synthetically so far",
                evidence=(
                    f"overlap_decision={overlap.get('decision')}; "
                    f"safe_candidate_count={overlap.get('safe_candidate_count')}"
                ),
                reviewer_attack="The abstention story may be a toy construction rather than a real benchmark behavior.",
                required_fix="Add a real frontier stress test where VERA abstains or narrowly rejects unsafe edits.",
            )
        )
    elif overlap.get("decision") != "ABSTAIN":
        findings.append(
            Finding(
                severity="critical",
                title="Abstention artifact is missing or invalid",
                evidence=f"overlap_case={overlap}",
                reviewer_attack="The differentiating feature is asserted but not demonstrated.",
                required_fix="Regenerate the synthetic abstention certificate and add a real benchmark stress test.",
            )
        )

    if (
        shutil.which("latexmk") is None
        and shutil.which("pdflatex") is None
        and shutil.which("tectonic") is None
    ):
        findings.append(
            Finding(
                severity="minor",
                title="Manuscript source has not been locally compiled",
                evidence="latexmk=pdflatex=tectonic=missing",
                reviewer_attack="The paper draft may contain LaTeX errors that are invisible to code tests.",
                required_fix="Install a TeX toolchain or compile in CI before treating the manuscript as submission-ready.",
            )
        )

    novelty_path = MAINTRACK_DIR / "NOVELTY_LOCK.md"
    novelty_sweep_path = MAINTRACK_DIR / "NOVELTY_SWEEP.md"
    novelty_update_path = MAINTRACK_DIR / "NOVELTY_SWEEP_2026_UPDATE.md"
    novelty_text = load_text(novelty_path)
    novelty_sweep_text = load_text(novelty_sweep_path)
    novelty_update_text = load_text(novelty_update_path)
    if "first representation-editing method" in novelty_text:
        findings.append(
            Finding(
                severity="major",
                title="Novelty sentence uses a strong first-claim",
                evidence="NOVELTY_LOCK.md contains 'first representation-editing method'",
                reviewer_attack="A reviewer may reject the contribution if one missed concurrent paper weakens the first-claim.",
                required_fix="Before submission, run an updated literature sweep and consider softening to 'we introduce the first framework we are aware of...'.",
            )
        )
    novelty_sweep_combined = "\n".join([novelty_sweep_text, novelty_update_text])
    if not (novelty_sweep_path.exists() or novelty_update_path.exists()) or not all(
        term in novelty_sweep_combined
        for term in ("SPLINCE", "MANCE", "Current Defensible Contribution Sentence")
    ):
        findings.append(
            Finding(
                severity="major",
                title="Novelty sweep is missing close 2025-2026 competitors",
                evidence=(
                    f"novelty_sweep_exists={novelty_sweep_path.exists()}; "
                    f"novelty_update_exists={novelty_update_path.exists()}"
                ),
                reviewer_attack="The paper may miss the strongest target-preserving erasure competitors.",
                required_fix="Update the novelty sweep with SPLINCE/SPLICE, MANCE/MANCE++, and the final contribution sentence.",
            )
        )

    mance_reference_paths = [
        path
        for path in ARTIFACT_DIR.glob("*mance*")
        if "style" not in path.name.lower()
        and path.name.endswith(("_receipt.json", "_statistical_report.json", "_statistics.json"))
    ]
    mance_claim_grade_artifacts = []
    mance_diagnostic_artifacts = []
    for path in mance_reference_paths:
        data = load_json(path)
        if data.get("claim_grade_reference_row") is True:
            mance_claim_grade_artifacts.append(path)
        else:
            mance_diagnostic_artifacts.append(path)
    has_mance_update = "MANCE++" in novelty_update_text or "Manifold Aware Concept Erasure" in novelty_update_text
    if has_mance_update and not mance_claim_grade_artifacts:
        findings.append(
            Finding(
                severity="major",
                title="Fresh MANCE++ baseline is not yet claim-grade",
                evidence=(
                    f"novelty_update_exists={novelty_update_path.exists()}; "
                    f"claim_grade_reference_artifacts={len(mance_claim_grade_artifacts)}; "
                    f"diagnostic_reference_artifacts={len(mance_diagnostic_artifacts)}; "
                    f"diagnostics={[path.name for path in mance_diagnostic_artifacts[:3]]}"
                ),
                reviewer_attack=(
                    "A reviewer can argue that a July 2026 manifold-aware nonlinear "
                    "erasure method is the closest baseline and that a diagnostic or MANCE-style "
                    "proxy is insufficient for a main-track comparison."
                ),
                required_fix=(
                    "Run a claim-grade reference MANCE/MANCE++ baseline under matched splits, or keep "
                    "the claim explicitly scoped as a protocol contribution without state-of-the-art "
                    "erasure claims; track the work in research/maintrack/REFERENCE_BASELINE_HARDENING_PLAN.md."
                ),
            )
        )

    if not full_goal:
        findings.append(
            Finding(
                severity="major",
                title="Full AAAI/NeurIPS goal audit is missing",
                evidence="research/artifacts/faro_goal_completion_audit.json is missing or unreadable",
                reviewer_attack=(
                    "The local readiness gates may be green while the broader submission objective "
                    "still has unverified requirements."
                ),
                required_fix="Run research/scripts/audit_goal_completion.py and resolve or explicitly scope every partial/failing item.",
            )
        )
    elif not bool(full_goal.get("goal_complete")):
        partial = [
            item.get("key", "")
            for item in full_goal.get("requirements", [])
            if isinstance(item, dict) and item.get("status") == "partial"
        ]
        failing = [
            item.get("key", "")
            for item in full_goal.get("requirements", [])
            if isinstance(item, dict) and item.get("status") == "fail"
        ]
        independent_partial = [key for key in partial if key != SELF_GOAL_KEY]
        if independent_partial or failing:
            findings.append(
                Finding(
                    severity="major",
                    title="Full AAAI/NeurIPS goal is not yet complete",
                    evidence=(
                        f"goal_complete={full_goal.get('goal_complete')}; "
                        f"status_counts={full_goal.get('status_counts')}; "
                        f"partial={partial}; fail={failing}; "
                        f"independent_partial={independent_partial}"
                    ),
                    reviewer_attack=(
                        "A reviewer-facing readiness gate is being conflated with the stronger user goal: "
                        "a main-track AAAI/NeurIPS paper with broad benchmarks, hardened baselines, mature "
                        "theory, statistical integrity, reproducibility, and final writing."
                    ),
                    required_fix=(
                        "Clear the independent partial/failing items in "
                        "research/artifacts/faro_goal_completion_audit.md or explicitly lower the "
                        "target from main-track submission readiness."
                    ),
                )
            )

    return findings


def write_markdown(path: Path, findings: list[Finding], report: dict[str, object]) -> None:
    lines = [
        "# VERA Adversarial Internal Review",
        "",
        f"Generated at UTC: `{report['created_at_utc']}`",
        "",
        "## Decision",
        "",
        "Do not submit while any critical finding remains unresolved.",
        "",
        "## Findings",
        "",
        "| Severity | Finding | Evidence | Required fix |",
        "| --- | --- | --- | --- |",
    ]
    for finding in findings:
        lines.append(
            f"| {finding.severity} | {finding.title} | "
            f"{finding.evidence} | {finding.required_fix} |"
        )
    lines.extend(
        [
            "",
            "## Reviewer Attacks",
            "",
        ]
    )
    for finding in findings:
        lines.append(f"- **{finding.title}:** {finding.reviewer_attack}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MD)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    findings = collect_findings()
    critical_count = sum(1 for finding in findings if finding.severity == "critical")
    major_count = sum(1 for finding in findings if finding.severity == "major")
    minor_count = sum(1 for finding in findings if finding.severity == "minor")
    report = {
        "name": "VERA adversarial internal review",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "submission_ready": critical_count == 0 and major_count == 0,
        "critical_count": critical_count,
        "major_count": major_count,
        "minor_count": minor_count,
        "findings": [asdict(finding) for finding in findings],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.markdown_out, findings, report)

    print("VERA adversarial internal review complete")
    print(f"submission_ready={str(report['submission_ready']).lower()}")
    print(f"critical_count={critical_count}")
    print(f"major_count={major_count}")
    print(f"report={args.json_out}")
    return 0 if args.no_fail or report["submission_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
