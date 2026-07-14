"""Audit the exact seven VERA paper gates and submission machinery.

This is a fail-closed audit.  It never infers scientific readiness from keyword
presence, old benchmark rows, or an internal review.  Downstream aggregate
reports must explicitly attest their preregistered pass conditions and expose
the counts used here.  Human-review and account checks require human evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MAINTRACK_DIR = ROOT / "maintrack"
ARTIFACT_DIR = ROOT / "artifacts"

DEFAULT_JSON = ARTIFACT_DIR / "faro_goal_completion_audit.json"
DEFAULT_MD = ARTIFACT_DIR / "faro_goal_completion_audit.md"

EXPECTED_DATASETS = {
    "Waterbirds",
    "Camelyon17-WILDS",
    "CivilComments-WILDS",
    "Bios",
    "GaitPDB",
}
EXPECTED_ERASERS = {"INLP", "RLACE", "LEACE", "MANCE++", "TaCo"}
EXPECTED_SEEDS = {0, 1, 2, 3, 4}
EXPECTED_REAL_FRACTIONS = {0.05, 0.1, 0.25, 0.5, 1.0}
EXPECTED_SYNTHETIC_SIZES = {250, 500, 1000, 2000, 5000, 10000}
EXPECTED_DELTAS = {0.01, 0.05, 0.1}


@dataclass(frozen=True)
class Gate:
    key: str
    title: str
    status: str
    evidence: str
    required_next: str


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


def sidecar_hash(path: Path) -> str:
    if not path.is_file():
        return ""
    fields = path.read_text(encoding="utf-8").strip().split()
    return fields[0] if fields else ""


def set_of(data: dict[str, Any], key: str) -> set[Any]:
    value = data.get(key, [])
    return set(value) if isinstance(value, list) else set()


def gate(key: str, title: str, passed: bool, evidence: str, required_next: str) -> Gate:
    return Gate(
        key=key,
        title=title,
        status="pass" if passed else "fail",
        evidence=evidence,
        required_next="None." if passed else required_next,
    )


def theory_gate() -> Gate:
    prereg = ROOT / "prereg.json"
    prereg_hash = ROOT / "prereg.sha256"
    verification = load_json(ARTIFACT_DIR / "vera_robust_synthetic_verification.json")
    synthetic = load_json(ARTIFACT_DIR / "vera_robust_synthetic_report.json")
    proof_path = MAINTRACK_DIR / "appendix_shift_robust_theory.tex"
    proof = proof_path.read_text(encoding="utf-8") if proof_path.is_file() else ""

    hash_ok = prereg.is_file() and sha256(prereg) == sidecar_hash(prereg_hash)
    cells = synthetic.get("cells", [])
    cells = cells if isinstance(cells, list) else []
    sizes = {int(c["n"]) for c in cells if isinstance(c, dict) and "n" in c}
    deltas = {float(c["delta"]) for c in cells if isinstance(c, dict) and "delta" in c}
    cells_ok = (
        len(cells) == 18
        and sizes == EXPECTED_SYNTHETIC_SIZES
        and deltas == EXPECTED_DELTAS
        and all(
            isinstance(c, dict)
            and int(c.get("replicates", 0)) == 1000
            and c.get("coverage_pass") is True
            for c in cells
        )
    )
    labels = {
        "robust_pair": "\\label{thm:robust-paired}",
        "shift_radius": "\\label{thm:shift-radius}",
        "worst_group": "\\label{cor:mixture}",
        "impossibility": "\\label{thm:unsupported}",
    }
    proof_blocks = proof.count("\\begin{proof}")
    proof_ok = all(label in proof for label in labels.values()) and proof_blocks >= 4
    verification_ok = (
        verification.get("verified") is True
        and int(verification.get("cell_count", 0)) == 18
        and verification.get("failures") == []
        and verification.get("prereg_sha256") == sidecar_hash(prereg_hash)
    )
    passed = hash_ok and proof_ok and cells_ok and verification_ok
    return gate(
        "goal_1_shift_aware_theory",
        "Shift-aware certification and impossibility",
        passed,
        (
            f"prereg_hash_valid={hash_ok}; proof_blocks={proof_blocks}; "
            f"required_labels_present={proof_ok}; synthetic_cells={len(cells)}; "
            f"synthetic_grid_valid={cells_ok}; independent_verification={verification_ok}"
        ),
        "Complete both proofs, valid preregistration, and independently verified 18-cell coverage simulation.",
    )


def theory_data_gate() -> Gate:
    report = load_json(ARTIFACT_DIR / "vera_real_theory_match_report.json")
    passed = (
        report.get("passed") is True
        and int(report.get("dataset_count", 0)) == 5
        and set_of(report, "datasets") == EXPECTED_DATASETS
        and set_of(report, "validation_fractions") == EXPECTED_REAL_FRACTIONS
        and int(report.get("datasets_tracking_predicted_band", 0)) >= 4
        and report.get("false_acceptance_below_delta_every_dataset_seed") is True
        and report.get("synthetic_overlay_verified") is True
        and report.get("real_overlay_figure_verified") is True
    )
    return gate(
        "goal_2_theory_matched_by_data",
        "Theory matched by synthetic and real data",
        passed,
        (
            f"report_present={bool(report)}; passed={report.get('passed')}; "
            f"dataset_count={report.get('dataset_count')}; "
            f"tracking={report.get('datasets_tracking_predicted_band')}; "
            f"all_false_acceptance_controlled={report.get('false_acceptance_below_delta_every_dataset_seed')}"
        ),
        "Run the locked real-data subsampling study and verify predicted/observed overlays on at least four datasets.",
    )


def killer_experiment_gate() -> Gate:
    report = load_json(ARTIFACT_DIR / "vera_deployment_rule_report.json")
    rules = set_of(report, "deployment_rules")
    passed = (
        report.get("passed") is True
        and set_of(report, "datasets") == EXPECTED_DATASETS
        and set_of(report, "seeds") == EXPECTED_SEEDS
        and int(report.get("eraser_count", 0)) >= 4
        and int(report.get("threshold_pair_count", 0)) >= 9
        and int(report.get("validation_size_count", 0)) >= 4
        and {"always_deploy", "point_selection", "vera", "oracle"}.issubset(rules)
        and int(report.get("datasets_with_naive_violation_at_least_20pct", 0)) == 5
        and float(report.get("vera_global_false_acceptance_upper", 1.0)) <= float(report.get("delta", 0.0))
        and int(report.get("holm_mcnemar_significant_dataset_count", 0)) >= 4
        and report.get("certification_tax_intervals_reported") is True
    )
    return gate(
        "goal_3_killer_experiment",
        "Deployment rules head to head",
        passed,
        (
            f"report_present={bool(report)}; rules={sorted(map(str, rules))}; "
            f"naive_failure_datasets={report.get('datasets_with_naive_violation_at_least_20pct')}; "
            f"vera_upper={report.get('vera_global_false_acceptance_upper')}; delta={report.get('delta')}; "
            f"significant_datasets={report.get('holm_mcnemar_significant_dataset_count')}"
        ),
        "Complete the preregistered four-rule grid, global false-acceptance analysis, McNemar tests, and retention intervals.",
    )


def baselines_gate() -> Gate:
    report = load_json(ARTIFACT_DIR / "official_eraser_receipt_audit.json")
    expected_receipts = len(EXPECTED_DATASETS) * len(EXPECTED_ERASERS) * len(EXPECTED_SEEDS)
    passed = (
        report.get("passed") is True
        and set_of(report, "datasets") == EXPECTED_DATASETS
        and set_of(report, "erasers") == EXPECTED_ERASERS
        and set_of(report, "seeds") == EXPECTED_SEEDS
        and int(report.get("official_run_receipt_count", 0)) >= expected_receipts
        and int(report.get("missing_run_receipt_count", -1)) == 0
        and int(report.get("proxy_row_count", -1)) == 0
        and int(report.get("invalid_receipt_count", -1)) == 0
        and report.get("all_upstream_commits_pinned") is True
        and report.get("shared_protocol_verified") is True
    )
    return gate(
        "goal_4_zero_proxy_baselines",
        "Official baselines on five datasets",
        passed,
        (
            f"report_present={bool(report)}; receipts={report.get('official_run_receipt_count')}/{expected_receipts}; "
            f"missing={report.get('missing_run_receipt_count')}; proxies={report.get('proxy_row_count')}; "
            f"invalid={report.get('invalid_receipt_count')}; pinned={report.get('all_upstream_commits_pinned')}"
        ),
        "Produce and validate all 125 official method/dataset/seed run receipts with zero proxy or missing rows.",
    )


def memorable_number_gate() -> Gate:
    report = load_json(ARTIFACT_DIR / "abstract_numbers_audit.json")
    x = report.get("point_selection_violation_rate")
    y = report.get("vera_violation_rate")
    z = report.get("safe_deployment_retention")
    numeric = all(isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0 for value in (x, y, z))
    gap = float(x) - float(y) if numeric else float("-inf")
    passed = (
        report.get("verified") is True
        and numeric
        and gap >= 0.15
        and report.get("sentence_matches_manuscript") is True
        and report.get("all_numbers_receipted") is True
    )
    return gate(
        "goal_5_memorable_number",
        "Receipted abstract result",
        passed,
        f"report_present={bool(report)}; X={x}; Y={y}; Z={z}; X_minus_Y={gap if numeric else None}; verified={report.get('verified')}",
        "Derive X/Y/Z from locked receipts and verify the exact abstract and introduction sentence with a gap of at least 15 points.",
    )


def presentation_gate() -> Gate:
    report = load_json(ARTIFACT_DIR / "presentation_readiness_audit.json")
    passed = (
        report.get("passed") is True
        and int(report.get("content_page_count", 0)) == 7
        and int(report.get("verified_reference_count", 0)) >= 40
        and report.get("figure_1_vector") is True
        and report.get("figure_1_colorblind_safe") is True
        and report.get("figure_1_readable_at_half_scale") is True
        and int(report.get("forbidden_name_hit_count", -1)) == 0
        and report.get("anonymous_pdf_clean") is True
        and report.get("named_pdf_clean") is True
        and report.get("pdf_metadata_clean") is True
        and report.get("abstract_figure1_sufficiency_reviewed") is True
    )
    return gate(
        "goal_6_presentation",
        "Top-conference presentation",
        passed,
        (
            f"report_present={bool(report)}; pages={report.get('content_page_count')}; "
            f"verified_references={report.get('verified_reference_count')}; "
            f"forbidden_hits={report.get('forbidden_name_hit_count')}; "
            f"anonymous_clean={report.get('anonymous_pdf_clean')}; named_clean={report.get('named_pdf_clean')}"
        ),
        "Finish and independently audit the seven-page paper, Figure 1, 40+ references, naming purge, both PDFs, and metadata.",
    )


def external_review_gate() -> Gate:
    report = load_json(ARTIFACT_DIR / "external_review_audit.json")
    passed = (
        report.get("passed") is True
        and int(report.get("completed_review_count", 0)) >= 2
        and int(report.get("ml_publisher_reviewer_count", 0)) >= 2
        and int(report.get("unresolved_critical_count", -1)) == 0
        and int(report.get("unresolved_major_count", -1)) == 0
        and int(report.get("reviewers_flagging_unaddressed_ltt_overlap", -1)) == 0
        and report.get("response_ledger_complete") is True
        and report.get("reviewer_identity_evidence_human_verified") is True
    )
    return gate(
        "goal_7_external_adversarial_review",
        "Two external cold reviews",
        passed,
        (
            f"report_present={bool(report)}; completed={report.get('completed_review_count')}; "
            f"ML_publishers={report.get('ml_publisher_reviewer_count')}; "
            f"unresolved_critical={report.get('unresolved_critical_count')}; "
            f"unresolved_major={report.get('unresolved_major_count')}; "
            f"unaddressed_LTT={report.get('reviewers_flagging_unaddressed_ltt_overlap')}"
        ),
        "Obtain two real cold reviews from ML publishers and close every critical/major item in a response ledger.",
    )


def submission_gate() -> Gate:
    report = load_json(ARTIFACT_DIR / "submission_machinery_audit.json")
    required = (
        "openreview_account_human_confirmed",
        "single_email_human_confirmed",
        "target_style_compiles",
        "exact_page_limit",
        "zero_formatting_hacks",
        "anonymization_complete",
        "anonymous_archive_reproduces_main_table",
        "reproducibility_checklist_complete",
        "supplement_ready",
        "deadlines_human_confirmed",
        "areas_and_keywords_selected",
    )
    passed = report.get("passed") is True and all(report.get(key) is True for key in required)
    missing = [key for key in required if report.get(key) is not True]
    return gate(
        "submission_machinery",
        "Venue submission machinery",
        passed,
        f"report_present={bool(report)}; missing_or_unconfirmed={missing}",
        "Complete the technical packaging and obtain human confirmation for account, email, and venue-deadline items.",
    )


def collect_gates() -> list[Gate]:
    return [
        theory_gate(),
        theory_data_gate(),
        killer_experiment_gate(),
        baselines_gate(),
        memorable_number_gate(),
        presentation_gate(),
        external_review_gate(),
        submission_gate(),
    ]


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# VERA Exact Goal Completion Audit",
        "",
        f"Generated at UTC: `{report['created_at_utc']}`",
        f"Goal complete: `{report['goal_complete']}`",
        "",
        "> This audit is fail-closed. It does not predict acceptance or substitute for peer review.",
        "",
        "| Status | Gate | Evidence | Required next |",
        "| --- | --- | --- | --- |",
    ]
    for item in report["gates"]:
        lines.append(
            f"| {item['status']} | `{item['key']}`: {item['title']} | "
            f"{item['evidence']} | {item['required_next']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    gates = collect_gates()
    pass_count = sum(item.status == "pass" for item in gates)
    fail_count = len(gates) - pass_count
    report = {
        "name": "VERA exact user-goal completion audit",
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "goal_complete": fail_count == 0,
        "paper_goals_complete": all(item.status == "pass" for item in gates[:7]),
        "acceptance_guaranteed": False,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "gates": [asdict(item) for item in gates],
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(DEFAULT_MD, report)
    print("VERA exact-goal completion audit complete")
    print(f"goal_complete={str(report['goal_complete']).lower()}")
    print(f"pass={pass_count} fail={fail_count}")
    print(f"report={DEFAULT_JSON}")
    return 0 if args.no_fail or report["goal_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
