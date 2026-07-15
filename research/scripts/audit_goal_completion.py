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
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
MAINTRACK_DIR = ROOT / "maintrack"
ARTIFACT_DIR = ROOT / "artifacts"
AUTHOR_KIT = MAINTRACK_DIR / "aaai2027_template" / "AuthorKit27"

DEFAULT_JSON = ARTIFACT_DIR / "vera_goal_completion_audit.json"
DEFAULT_MD = ARTIFACT_DIR / "vera_goal_completion_audit.md"

EXPECTED_DATASETS = {
    "Waterbirds",
    "Camelyon17-WILDS",
    "CivilComments-WILDS",
    "Bios",
    "GaitPDB",
}
EXPECTED_ERASERS = {"INLP", "RLACE", "LEACE", "MANCE++", "TaCo"}
EXPECTED_SEEDS = {5, 6, 7, 8, 9, 10, 11, 12}
INDEPENDENT_STRESS_SEEDS = set(range(13, 45))
EXPECTED_REAL_FRACTIONS = {0.05, 0.1, 0.25, 0.5, 1.0}
EXPECTED_SYNTHETIC_SIZES = {250, 500, 1000, 2000, 5000, 10000}
EXPECTED_DELTAS = {0.01, 0.05, 0.1}
EXPECTED_GAMMAS = {1.0, 1.01, 1.25}
REQUESTED_SEEDS = EXPECTED_SEEDS
EXPECTED_FAMILY_COUNTS = {5, 9, 13, 17}
EXPECTED_GROUP_COUNTS = {1, 3, 5}


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


def committed_file_matches(commit: str, path: Path) -> bool:
    """Return whether a commit contains the current bytes at a repository path."""
    if not commit or not path.is_file():
        return False
    try:
        relative = path.resolve().relative_to(REPOSITORY.resolve())
        stored = subprocess.run(
            ["git", "show", f"{commit}:{relative.as_posix()}"],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, ValueError, subprocess.CalledProcessError):
        return False
    return hashlib.sha256(stored).hexdigest() == sha256(path)


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
    prereg = ROOT / "prereg_confirmatory_balanced.json"
    prereg_hash = ROOT / "prereg_confirmatory_balanced.sha256"
    verification = load_json(ARTIFACT_DIR / "vera_exact_balanced_audit.json")
    synthetic = load_json(ARTIFACT_DIR / "vera_exact_balanced_report.json")
    theory_consistency = load_json(ARTIFACT_DIR / "vera_theory_consistency_audit.json")
    proof_path = MAINTRACK_DIR / "appendix_shift_robust_theory.tex"
    proof = proof_path.read_text(encoding="utf-8") if proof_path.is_file() else ""

    hash_ok = prereg.is_file() and sha256(prereg) == sidecar_hash(prereg_hash)
    cells = synthetic.get("cells", [])
    cells = cells if isinstance(cells, list) else []
    sizes = {int(c["n"]) for c in cells if isinstance(c, dict) and "n" in c}
    deltas = {float(c["delta"]) for c in cells if isinstance(c, dict) and "delta" in c}
    gammas = {float(c["gamma"]) for c in cells if isinstance(c, dict) and "gamma" in c}
    cells_ok = (
        len(cells) == 54
        and sizes == EXPECTED_SYNTHETIC_SIZES
        and deltas == EXPECTED_DELTAS
        and gammas == EXPECTED_GAMMAS
        and synthetic.get("claim_grade") is True
        and synthetic.get("all_cells_pass") is True
        and synthetic.get("prereg_sha256") == sidecar_hash(prereg_hash)
        and all(
            isinstance(c, dict)
            and int(c.get("replicates", 0)) == 2000
            and c.get("coverage_pass") is True
            and c.get("overlay_pass") is True
            for c in cells
        )
    )
    labels = {
        "robust_pair": "\\label{thm:robust-paired}",
        "shift_radius": "\\label{thm:shift-radius}",
        "worst_group": "\\label{cor:mixture}",
        "shift_envelope": "\\label{cor:shift-envelope}",
        "impossibility": "\\label{thm:unsupported}",
    }
    proof_blocks = proof.count("\\begin{proof}")
    proof_ok = all(label in proof for label in labels.values()) and proof_blocks >= 4
    verification_ok = (
        verification.get("passed") is True
        and int(verification.get("cells_replayed", 0)) == 54
        and int(verification.get("false_acceptances_replayed", -1)) == 0
        and verification.get("failures") == []
        and verification.get("prereg_sha256") == sidecar_hash(prereg_hash)
        and verification.get("report_sha256")
        == sha256(ARTIFACT_DIR / "vera_exact_balanced_report.json")
    )
    theory_consistency_ok = (
        theory_consistency.get("passed") is True
        and theory_consistency.get("formal_proof_verified") is False
        and theory_consistency.get("novelty_verified") is False
        and theory_consistency.get("prereg_sha256") == sidecar_hash(prereg_hash)
        and theory_consistency.get("group_shift_envelope", {}).get("passed") is True
        and theory_consistency.get("balanced_shift_envelope", {}).get("passed") is True
        and theory_consistency.get("theory_sha256") == sha256(proof_path)
    )
    passed = (
        hash_ok
        and proof_ok
        and cells_ok
        and verification_ok
        and theory_consistency_ok
    )
    return gate(
        "goal_1_shift_aware_theory",
        "Shift-aware certification and impossibility",
        passed,
        (
            f"prereg_hash_valid={hash_ok}; "
            f"proof_blocks={proof_blocks}; "
            f"required_labels_present={proof_ok}; synthetic_cells={len(cells)}; "
            f"synthetic_grid_valid={cells_ok}; independent_verification={verification_ok}"
            f"; theory_implementation_consistency={theory_consistency_ok}"
        ),
        "Complete both proofs, the balanced envelope implementation, valid preregistration, and independently replayed 54-cell coverage simulation.",
    )


def theory_data_gate() -> Gate:
    report = load_json(ARTIFACT_DIR / "vera_learning_curve_diagnostic.json")
    exact = load_json(ARTIFACT_DIR / "vera_exact_balanced_audit.json")
    confirmatory = load_json(
        ARTIFACT_DIR / "vera_confirmatory_balanced_report.json"
    )
    independent = load_json(
        ARTIFACT_DIR / "vera_confirmatory_analysis_audit.json"
    )
    records = report.get("records", {})
    records = records if isinstance(records, dict) else {}
    passed = (
        report.get("descriptive_calibration_only") is True
        and set(records) == EXPECTED_DATASETS
        and int(report.get("datasets_with_all_five_points_inside", 0)) >= 4
        and report.get("four_of_five_diagnostic_target_met") is True
        and all(
            isinstance(record, dict)
            and len(record.get("observed_abstention", [])) == 5
            and len(record.get("predicted_mean", [])) == 5
            and len(record.get("pointwise_95_lower", [])) == 5
            and len(record.get("pointwise_95_upper", [])) == 5
            for record in records.values()
        )
        and exact.get("passed") is True
        and confirmatory.get("pass_conditions", {}).get("vera_control") is True
        and confirmatory.get("strict_supported_dataset_seed_control") is True
        and independent.get("passed") is True
        and independent.get("strict_supported_dataset_seed_control") is True
        and int(independent.get("raw_npz_files_recomputed", 0)) == 480
        and int(independent.get("raw_candidate_rows_recomputed", 0)) == 25_920
        and int(independent.get("raw_candidate_mismatches", -1)) == 0
    )
    return gate(
        "goal_2_theory_matched_by_data",
        "Theory matched by synthetic and real data",
        passed,
        (
            f"report_present={bool(report)}; passed={report.get('passed')}; "
            f"dataset_count={len(records)}; "
            f"tracking={report.get('datasets_with_all_five_points_inside')}; "
            f"exact_replay={exact.get('passed')}; "
            f"confirmatory_vera_control="
            f"{confirmatory.get('pass_conditions', {}).get('vera_control')}; "
            f"strict_seed_control={confirmatory.get('strict_supported_dataset_seed_control')}; "
            f"raw_rows={independent.get('raw_candidate_rows_recomputed')}; "
            f"raw_mismatches={independent.get('raw_candidate_mismatches')}"
        ),
        "Run the locked real-data subsampling study and verify predicted/observed overlays on at least four datasets.",
    )


def killer_experiment_gate() -> Gate:
    stress_report = load_json(ARTIFACT_DIR / "vera_independent_stress_report.json")
    stress_audit = load_json(
        ARTIFACT_DIR / "vera_independent_stress_analysis_audit.json"
    )
    stress_receipts = load_json(
        ARTIFACT_DIR / "independent_stress_replication_receipt_audit.json"
    )
    if stress_report or stress_audit or stress_receipts:
        threshold_report = load_json(
            ARTIFACT_DIR / "vera_confirmatory_balanced_report.json"
        )
        threshold_audit = load_json(
            ARTIFACT_DIR / "vera_confirmatory_analysis_audit.json"
        )
        threshold_grid_valid = (
            set_of(threshold_report, "datasets") == EXPECTED_DATASETS
            and set_of(threshold_report, "confirmatory_seeds") == EXPECTED_SEEDS
            and int(threshold_report.get("threshold_pair_count", 0)) >= 3
            and len(set_of(threshold_report, "validation_fractions")) >= 3
            and {
                "always_deploy_balanced",
                "point_selection_balanced",
                "vera_balanced_iut",
                "external_balanced_oracle",
            }.issubset(set_of(threshold_report, "deployment_rules"))
            and threshold_audit.get("passed") is True
            and int(threshold_audit.get("raw_candidate_rows_recomputed", 0))
            == 25_920
            and int(threshold_audit.get("raw_candidate_mismatches", -1)) == 0
        )
        stress_rules = set_of(stress_report, "deployment_rules")
        stress_supported = set_of(stress_report, "supported_datasets")
        dataset_pass = stress_report.get("dataset_pass_conditions", {})
        dataset_pass = dataset_pass if isinstance(dataset_pass, dict) else {}
        supported_dataset_records = {
            dataset: record
            for dataset, record in dataset_pass.items()
            if dataset in stress_supported and isinstance(record, dict)
        }
        strict_dataset_count = sum(
            bool(record.get("passed_all_three"))
            for record in supported_dataset_records.values()
        )
        grid_valid = (
            set_of(stress_report, "datasets") == EXPECTED_DATASETS
            and stress_supported
            == {"Waterbirds", "CivilComments-WILDS", "Bios", "GaitPDB"}
            and set_of(stress_report, "erasers") == EXPECTED_ERASERS
            and set_of(stress_report, "replication_seeds")
            == INDEPENDENT_STRESS_SEEDS
            and int(stress_report.get("rule_row_count", 0))
            == len(EXPECTED_DATASETS) * len(INDEPENDENT_STRESS_SEEDS) * 5
            and int(stress_report.get("candidate_row_count", 0))
            == len(EXPECTED_DATASETS) * len(INDEPENDENT_STRESS_SEEDS) * 12
            and {
                "always_deploy_balanced",
                "point_selection_balanced",
                "vera_balanced_iut",
                "vera_balanced_envelope",
                "external_balanced_oracle",
            }.issubset(stress_rules)
        )
        strict_pass = (
            stress_report.get("passed") is True
            and stress_audit.get("passed") is True
            and stress_audit.get("confirmatory_passed") is True
            and stress_receipts.get("passed") is True
            and grid_valid
            and threshold_grid_valid
            and strict_dataset_count == 4
            and stress_report.get("global_vera_control") is True
            and stress_report.get("camelyon_forced_abstention") is True
            and stress_report.get("pass_conditions", {}).get(
                "four_supported_datasets_pass_all_three"
            )
            is True
            and stress_report.get("pass_conditions", {}).get(
                "global_vera_violation_rate_at_most_delta"
            )
            is True
            and stress_report.get("pass_conditions", {}).get(
                "camelyon_forced_abstention_all_registered_vera_rules"
            )
            is True
        )
        return gate(
            "goal_3_killer_experiment",
            "Deployment rules head to head",
            strict_pass,
            (
                f"independent_stress_report_present={bool(stress_report)}; "
                f"receipt_audit_pass={stress_receipts.get('passed')}; "
                f"analysis_audit_pass={stress_audit.get('passed')}; "
                f"grid_valid={grid_valid}; "
                f"threshold_grid_valid={threshold_grid_valid}; "
                f"supported_datasets_passing_all_three={strict_dataset_count}/4; "
                f"global_vera_control={stress_report.get('global_vera_control')}; "
                f"camelyon_forced_abstention={stress_report.get('camelyon_forced_abstention')}"
            ),
            "Complete the locked independent stress replication with all four supported datasets passing the naive-failure, VERA-control, and Holm-corrected paired-test endpoints.",
        )

    report = load_json(ARTIFACT_DIR / "vera_confirmatory_balanced_report.json")
    independent = load_json(
        ARTIFACT_DIR / "vera_confirmatory_analysis_audit.json"
    )
    rules = set_of(report, "deployment_rules")
    summaries = report.get("primary_summaries", {})
    vera = summaries.get("vera_balanced_iut", {})
    passed = (
        report.get("passed") is True
        and independent.get("passed") is True
        and independent.get("confirmatory_passed") is True
        and int(independent.get("raw_npz_files_recomputed", 0)) == 480
        and int(independent.get("raw_candidate_mismatches", -1)) == 0
        and set_of(report, "datasets") == EXPECTED_DATASETS
        and set_of(report, "confirmatory_seeds") == EXPECTED_SEEDS
        and len(set_of(report, "erasers")) >= 5
        and int(report.get("threshold_pair_count", 0)) >= 9
        and len(set_of(report, "validation_fractions")) >= 5
        and {
            "always_deploy_balanced",
            "point_selection_balanced",
            "vera_balanced_iut",
            "external_balanced_oracle",
        }.issubset(rules)
        and int(report.get("datasets_with_naive_violation_at_least_20pct", 0)) >= 1
        and float(vera.get("measured_external_violation_rate", 1.0))
        <= float(report.get("delta", 0.0))
        and report.get("strict_supported_dataset_seed_control") is True
        and int(report.get("seed_blocked_significant_dataset_count", 0)) >= 1
        and report.get("certification_tax_intervals_reported") is True
        and report.get("pass_conditions", {}).get("abstract_gap") is True
    )
    return gate(
        "goal_3_killer_experiment",
        "Deployment rules head to head",
        passed,
        (
            f"report_present={bool(report)}; rules={sorted(map(str, rules))}; "
            f"naive_failure_datasets={report.get('datasets_with_naive_violation_at_least_20pct')}; "
            f"vera_observed_rate={vera.get('measured_external_violation_rate')}; "
            f"delta={report.get('delta')}; "
            f"seed_blocked_significant_datasets={report.get('seed_blocked_significant_dataset_count')}"
        ),
        "Complete the preregistered rule grid, observed false-acceptance analysis, seed-blocked paired tests, and retention intervals.",
    )


def baselines_gate() -> Gate:
    stress_report = load_json(
        ARTIFACT_DIR / "independent_stress_replication_receipt_audit.json"
    )
    stress_expected = (
        len(EXPECTED_DATASETS)
        * len(EXPECTED_ERASERS)
        * len(INDEPENDENT_STRESS_SEEDS)
    )
    if stress_report:
        stress_pass = (
            stress_report.get("passed") is True
            and set_of(stress_report, "datasets") == EXPECTED_DATASETS
            and set_of(stress_report, "erasers") == EXPECTED_ERASERS
            and set_of(stress_report, "seeds") == INDEPENDENT_STRESS_SEEDS
            and int(stress_report.get("official_run_receipt_count", 0))
            == stress_expected
            and int(stress_report.get("missing_run_receipt_count", -1)) == 0
            and int(stress_report.get("proxy_row_count", -1)) == 0
            and int(stress_report.get("invalid_receipt_count", -1)) == 0
            and stress_report.get("all_upstream_commits_pinned") is True
            and stress_report.get("shared_protocol_verified") is True
        )
        return gate(
            "goal_4_zero_proxy_baselines",
            "Official baselines on five datasets",
            stress_pass,
            (
                f"independent_receipt_count={stress_report.get('official_run_receipt_count')}/{stress_expected}; "
                f"seeds={sorted(set_of(stress_report, 'seeds'))}; "
                f"missing={stress_report.get('missing_run_receipt_count')}; "
                f"proxies={stress_report.get('proxy_row_count')}; "
                f"invalid={stress_report.get('invalid_receipt_count')}; "
                f"pinned={stress_report.get('all_upstream_commits_pinned')}"
            ),
            "Complete and audit every independent stress dataset/eraser/seed receipt with zero proxies.",
        )

    report = load_json(
        ARTIFACT_DIR / "confirmatory_balanced_receipt_audit.json"
    )
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
        "Produce and validate all 200 untouched-seed official method/dataset/seed run receipts with zero proxy or missing rows.",
    )


def memorable_number_gate() -> Gate:
    stress_report = load_json(ARTIFACT_DIR / "vera_independent_stress_report.json")
    stress_abstract = load_json(
        ARTIFACT_DIR / "vera_independent_stress_abstract_numbers.json"
    )
    stress_audit = load_json(
        ARTIFACT_DIR / "vera_independent_stress_analysis_audit.json"
    )
    stress_package = load_json(
        ARTIFACT_DIR / "vera_independent_stress_package_audit.json"
    )
    if stress_report or stress_abstract or stress_audit or stress_package:
        x = stress_abstract.get("point_selection_violation_rate")
        y = stress_abstract.get("vera_iut_violation_rate")
        z = stress_abstract.get("safe_retention")
        numeric = all(
            isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0
            for value in (x, y, z)
        )
        gap = float(x) - float(y) if numeric else float("-inf")
        passed = (
            stress_report.get("passed") is True
            and stress_abstract.get("verified") is True
            and stress_abstract.get("registered_pass_conditions_met") is True
            and numeric
            and gap >= 0.15
            and stress_audit.get("passed") is True
            and stress_audit.get("abstract_verified") is True
            and stress_package.get("passed") is True
            and stress_package.get("confirmatory_passed") is True
        )
        return gate(
            "goal_5_memorable_number",
            "Receipted abstract result",
            passed,
            (
                f"independent_report_passed={stress_report.get('passed')}; "
                f"verified={stress_abstract.get('verified')}; X={x}; Y={y}; Z={z}; "
                f"X_minus_Y={gap if numeric else None}; "
                f"package_passed={stress_package.get('passed')}; "
                f"registered_pass_conditions_met={stress_abstract.get('registered_pass_conditions_met')}"
            ),
            "Derive X/Y/Z from the independent stress receipts and package the audited abstract sentence.",
        )

    report = load_json(ARTIFACT_DIR / "vera_confirmatory_abstract_numbers.json")
    independent = load_json(
        ARTIFACT_DIR / "vera_confirmatory_analysis_audit.json"
    )
    package = load_json(
        ARTIFACT_DIR / "vera_confirmatory_results_package_audit.json"
    )
    x = report.get("point_selection_violation_rate")
    y = report.get("vera_iut_violation_rate")
    z = report.get("safe_retention")
    numeric = all(isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0 for value in (x, y, z))
    gap = float(x) - float(y) if numeric else float("-inf")
    macros = (
        AUTHOR_KIT / "vera_results_macros.tex"
    ).read_text(encoding="utf-8", errors="replace")
    descriptive_caveat_present = (
        "external-oracle opportunities" in macros
        and ("Holm" in macros or "seed" in macros.lower())
    )
    passed = (
        report.get("verified") is True
        and numeric
        and gap >= 0.15
        and int(report.get("stress_configuration_count", 0)) == 32
        and descriptive_caveat_present
        and independent.get("passed") is True
        and independent.get("abstract_verified") is True
        and package.get("passed") is True
        and package.get("abstract_numbers_sha256")
        == sha256(ARTIFACT_DIR / "vera_confirmatory_abstract_numbers.json")
    )
    return gate(
        "goal_5_memorable_number",
        "Receipted abstract result",
        passed,
        f"report_present={bool(report)}; X={x}; Y={y}; Z={z}; X_minus_Y={gap if numeric else None}; verified={report.get('verified')}; descriptive_caveat={descriptive_caveat_present}",
        "Derive X/Y/Z from locked receipts and place the descriptive effect and failed Holm result together in the abstract.",
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
        and report.get("ai_assistance_disclosure_present") is True
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
        and int(
            report.get(
                "reviewers_flagging_unaddressed_prompt_risk_control_overlap", -1
            )
        )
        == 0
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
            f"unaddressed_LTT={report.get('reviewers_flagging_unaddressed_ltt_overlap')}; "
            f"unaddressed_PRC={report.get('reviewers_flagging_unaddressed_prompt_risk_control_overlap')}"
        ),
        "Obtain two real cold reviews from ML publishers and close every critical/major item in a response ledger.",
    )


def requested_theory_gate(protocol: Gate) -> Gate:
    """Audit the literal theory/simulation bar, including m and |G| variation."""
    prereg = ROOT / "prereg_exact_family_grid.json"
    report = load_json(ARTIFACT_DIR / "vera_exact_balanced_report.json")
    family_report = load_json(ARTIFACT_DIR / "vera_exact_family_grid_report.json")
    family_audit = load_json(ARTIFACT_DIR / "vera_exact_family_grid_audit.json")
    config = report.get("config", {})
    config = config if isinstance(config, dict) else {}
    cells = family_report.get("cells", [])
    cells = cells if isinstance(cells, list) else []
    candidate_counts = {
        int(cell["candidate_count"])
        for cell in cells
        if isinstance(cell, dict) and "candidate_count" in cell
    }
    group_counts = {
        int(cell["environment_count"])
        for cell in cells
        if isinstance(cell, dict) and "environment_count" in cell
    }
    candidate_counts.update(
        int(value) for value in family_report.get("candidate_counts_tested", [])
    )
    group_counts.update(
        int(value) for value in family_report.get("group_counts_tested", [])
    )
    run_commit = str(family_report.get("git_commit", ""))
    prereg_committed = committed_file_matches(run_commit, prereg)
    explicit_grid = (
        candidate_counts == EXPECTED_FAMILY_COUNTS
        and group_counts == EXPECTED_GROUP_COUNTS
    )
    family_verified = (
        family_report.get("all_cells_pass") is True
        and int(family_report.get("cell_count", 0)) == 216
        and family_audit.get("passed") is True
        and int(family_audit.get("cells_replayed", 0)) == 216
        and family_audit.get("failures") == []
    )
    identity_included = (
        family_report.get("config", {}).get("identity_candidate_included") is True
        and int(family_report.get("config", {}).get("identity_candidate_index", -1))
        == 0
    )
    passed = (
        protocol.status == "pass"
        and explicit_grid
        and prereg_committed
        and family_verified
        and identity_included
    )
    return gate(
        "requested_goal_1_theory",
        "Requested theory grid and proofs",
        passed,
        (
            f"registered_theory_pass={protocol.status == 'pass'}; "
            f"candidate_counts_tested={sorted(candidate_counts)}; "
            f"group_counts_tested={sorted(group_counts)}; "
            f"explicit_m_and_group_variation={explicit_grid}; "
            f"run_prereg_bytes_committed={prereg_committed}; "
            f"family_grid_independently_verified={family_verified}; "
            f"identity_candidate_included={identity_included}; "
            f"configured_m={config.get('candidate_count')}; "
            f"configured_groups={config.get('environment_count')}"
        ),
        "Preregister and run an additional coverage grid that varies candidate count and validated-group count, then replay it independently.",
    )


def requested_theory_data_gate(protocol: Gate) -> Gate:
    return gate(
        "requested_goal_2_theory_data",
        "Requested synthetic and real curve overlay",
        protocol.status == "pass",
        f"registered_theory_data_pass={protocol.status == 'pass'}; {protocol.evidence}",
        protocol.required_next,
    )


def requested_killer_experiment_gate() -> Gate:
    """Audit the original no-partial-credit deployment-rule bar."""
    stress_report = load_json(ARTIFACT_DIR / "vera_independent_stress_report.json")
    stress_audit = load_json(
        ARTIFACT_DIR / "vera_independent_stress_analysis_audit.json"
    )
    stress_receipts = load_json(
        ARTIFACT_DIR / "independent_stress_replication_receipt_audit.json"
    )
    if stress_report or stress_audit or stress_receipts:
        threshold_report = load_json(
            ARTIFACT_DIR / "vera_confirmatory_balanced_report.json"
        )
        threshold_audit = load_json(
            ARTIFACT_DIR / "vera_confirmatory_analysis_audit.json"
        )
        threshold_grid_valid = (
            set_of(threshold_report, "datasets") == EXPECTED_DATASETS
            and set_of(threshold_report, "confirmatory_seeds") == EXPECTED_SEEDS
            and int(threshold_report.get("threshold_pair_count", 0)) >= 3
            and len(set_of(threshold_report, "validation_fractions")) >= 3
            and {
                "always_deploy_balanced",
                "point_selection_balanced",
                "vera_balanced_iut",
                "external_balanced_oracle",
            }.issubset(set_of(threshold_report, "deployment_rules"))
            and threshold_audit.get("passed") is True
            and int(threshold_audit.get("raw_candidate_rows_recomputed", 0))
            == 25_920
            and int(threshold_audit.get("raw_candidate_mismatches", -1)) == 0
        )
        stress_rules = set_of(stress_report, "deployment_rules")
        stress_supported = set_of(stress_report, "supported_datasets")
        dataset_pass = stress_report.get("dataset_pass_conditions", {})
        dataset_pass = dataset_pass if isinstance(dataset_pass, dict) else {}
        supported_dataset_records = {
            dataset: record
            for dataset, record in dataset_pass.items()
            if dataset in stress_supported and isinstance(record, dict)
        }
        strict_dataset_count = sum(
            bool(record.get("passed_all_three"))
            for record in supported_dataset_records.values()
        )
        grid_valid = (
            set_of(stress_report, "datasets") == EXPECTED_DATASETS
            and stress_supported
            == {
                "Waterbirds",
                "CivilComments-WILDS",
                "Bios",
                "GaitPDB",
            }
            and set_of(stress_report, "erasers") == EXPECTED_ERASERS
            and set_of(stress_report, "replication_seeds")
            == INDEPENDENT_STRESS_SEEDS
            and int(stress_report.get("rule_row_count", 0))
            == len(EXPECTED_DATASETS) * len(INDEPENDENT_STRESS_SEEDS) * 5
            and int(stress_report.get("candidate_row_count", 0))
            == len(EXPECTED_DATASETS) * len(INDEPENDENT_STRESS_SEEDS) * 12
            and {
                "always_deploy_balanced",
                "point_selection_balanced",
                "vera_balanced_iut",
                "vera_balanced_envelope",
                "external_balanced_oracle",
            }.issubset(stress_rules)
        )
        strict_pass = (
            stress_report.get("passed") is True
            and stress_audit.get("passed") is True
            and stress_audit.get("confirmatory_passed") is True
            and stress_receipts.get("passed") is True
            and grid_valid
            and threshold_grid_valid
            and strict_dataset_count == 4
            and stress_report.get("global_vera_control") is True
            and stress_report.get("camelyon_forced_abstention") is True
            and stress_report.get("pass_conditions", {}).get(
                "four_supported_datasets_pass_all_three"
            )
            is True
            and stress_report.get("pass_conditions", {}).get(
                "global_vera_violation_rate_at_most_delta"
            )
            is True
            and stress_report.get("pass_conditions", {}).get(
                "camelyon_forced_abstention_all_registered_vera_rules"
            )
            is True
        )
        return gate(
            "requested_goal_3_killer_experiment",
            "Requested strict false-acceptance study",
            strict_pass,
            (
                f"independent_stress_report_present={bool(stress_report)}; "
                f"receipt_audit_pass={stress_receipts.get('passed')}; "
                f"analysis_audit_pass={stress_audit.get('passed')}; "
                f"grid_valid={grid_valid}; "
                f"threshold_grid_valid={threshold_grid_valid}; "
                f"supported_datasets_passing_all_three={strict_dataset_count}/4; "
                f"global_vera_control={stress_report.get('global_vera_control')}; "
                f"camelyon_forced_abstention={stress_report.get('camelyon_forced_abstention')}"
            ),
            "Complete the locked independent stress replication: all four supported datasets must hit point-selection >=20%, VERA <= delta, and Holm-corrected McNemar <= .05, with Camelyon17 forced abstention.",
        )

    report = load_json(ARTIFACT_DIR / "vera_confirmatory_balanced_report.json")
    independent = load_json(ARTIFACT_DIR / "vera_confirmatory_analysis_audit.json")
    rules = set_of(report, "deployment_rules")
    summaries = report.get("primary_summaries", {})
    summaries = summaries if isinstance(summaries, dict) else {}
    vera = summaries.get("vera_balanced_iut", {})
    vera = vera if isinstance(vera, dict) else {}
    grid_ok = (
        set_of(report, "datasets") == EXPECTED_DATASETS
        and len(set_of(report, "erasers")) >= 4
        and int(report.get("threshold_pair_count", 0)) >= 9
        and len(set_of(report, "confirmatory_seeds")) >= 5
        and len(set_of(report, "validation_fractions")) >= 4
        and {
            "always_deploy_balanced",
            "point_selection_balanced",
            "vera_balanced_iut",
            "external_balanced_oracle",
        }.issubset(rules)
    )
    naive_failure_present = int(
        report.get("datasets_with_naive_violation_at_least_20pct", 0)
    ) >= 1
    vera_control = (
        float(vera.get("measured_external_violation_rate", 1.0))
        <= float(report.get("delta", 0.0))
        and report.get("pass_conditions", {}).get("vera_control") is True
        and report.get("strict_supported_dataset_seed_control") is True
    )
    paired_diagnostics_ok = (
        int(report.get("seed_blocked_significant_dataset_count", 0)) >= 1
        and len(report.get("mcnemar_discordant_counts", {})) == 4
        and report.get("mcnemar_discordant_counts_reported") is True
    )
    retention_ok = report.get("certification_tax_intervals_reported") is True
    passed = (
        report.get("passed") is True
        and independent.get("passed") is True
        and grid_ok
        and naive_failure_present
        and vera_control
        and paired_diagnostics_ok
        and retention_ok
    )
    return gate(
        "requested_goal_3_killer_experiment",
        "Requested strict false-acceptance study",
        passed,
        (
            f"grid_valid={grid_ok}; "
            f"naive_failure_datasets={report.get('datasets_with_naive_violation_at_least_20pct')} (required >=1); "
            f"vera_control={vera_control}; "
            f"seed_blocked_significant={report.get('seed_blocked_significant_dataset_count')}; "
            f"discordant_counts={report.get('mcnemar_discordant_counts_reported')}; "
            f"retention_intervals={retention_ok}"
        ),
        "Meet the prespecified >=20% naive-failure, strict VERA-control, seed-blocked significance, and retention conditions without post-hoc tuning.",
    )


def requested_baselines_gate() -> Gate:
    stress_report = load_json(
        ARTIFACT_DIR / "independent_stress_replication_receipt_audit.json"
    )
    stress_expected = (
        len(EXPECTED_DATASETS)
        * len(EXPECTED_ERASERS)
        * len(INDEPENDENT_STRESS_SEEDS)
    )
    if stress_report:
        stress_pass = (
            stress_report.get("passed") is True
            and set_of(stress_report, "datasets") == EXPECTED_DATASETS
            and set_of(stress_report, "erasers") == EXPECTED_ERASERS
            and set_of(stress_report, "seeds") == INDEPENDENT_STRESS_SEEDS
            and int(stress_report.get("official_run_receipt_count", 0))
            == stress_expected
            and int(stress_report.get("missing_run_receipt_count", -1)) == 0
            and int(stress_report.get("proxy_row_count", -1)) == 0
            and int(stress_report.get("invalid_receipt_count", -1)) == 0
            and stress_report.get("all_upstream_commits_pinned") is True
            and stress_report.get("shared_protocol_verified") is True
        )
        return gate(
            "requested_goal_4_zero_proxy_baselines",
            "Requested official baselines on untouched seeds",
            stress_pass,
            (
                f"independent_receipt_count={stress_report.get('official_run_receipt_count')}/{stress_expected}; "
                f"seeds={sorted(set_of(stress_report, 'seeds'))}; "
                f"proxies={stress_report.get('proxy_row_count')}; "
                f"invalid={stress_report.get('invalid_receipt_count')}; "
                f"pinned={stress_report.get('all_upstream_commits_pinned')}"
            ),
            "Complete and audit every independent stress dataset/eraser/seed receipt with zero proxies.",
        )

    report = load_json(ARTIFACT_DIR / "confirmatory_balanced_receipt_audit.json")
    requested_receipts = (
        len(EXPECTED_DATASETS) * len(EXPECTED_ERASERS) * len(REQUESTED_SEEDS)
    )
    passed = (
        report.get("passed") is True
        and set_of(report, "datasets") == EXPECTED_DATASETS
        and set_of(report, "erasers") == EXPECTED_ERASERS
        and set_of(report, "seeds") == REQUESTED_SEEDS
        and int(report.get("official_run_receipt_count", 0)) >= requested_receipts
        and int(report.get("missing_run_receipt_count", -1)) == 0
        and int(report.get("proxy_row_count", -1)) == 0
        and int(report.get("invalid_receipt_count", -1)) == 0
        and report.get("all_upstream_commits_pinned") is True
        and report.get("shared_protocol_verified") is True
    )
    return gate(
        "requested_goal_4_zero_proxy_baselines",
        "Requested official baselines on untouched seeds",
        passed,
        (
            f"receipt_count={report.get('official_run_receipt_count')}/{requested_receipts}; "
            f"seeds={sorted(set_of(report, 'seeds'))}; requested_seeds={sorted(REQUESTED_SEEDS)}; "
            f"proxies={report.get('proxy_row_count')}; invalid={report.get('invalid_receipt_count')}"
        ),
        "Complete and audit every official dataset/eraser/untouched-seed receipt with zero proxies.",
    )


def requested_memorable_number_gate(protocol: Gate) -> Gate:
    stress_abstract = load_json(
        ARTIFACT_DIR / "vera_independent_stress_abstract_numbers.json"
    )
    stress_report = load_json(ARTIFACT_DIR / "vera_independent_stress_report.json")
    if stress_abstract or stress_report:
        return gate(
            "requested_goal_5_memorable_number",
            "Requested receipted headline or theory lead",
            protocol.status == "pass",
            (
                "independent_stress_present=True; "
                f"registered_memorable_number_pass={protocol.status == 'pass'}; "
                f"{protocol.evidence}"
            ),
            "Meet the independent stress X/Y/Z headline exactly; the older theory-only fallback is disabled once the independent strict replication exists.",
        )

    report = load_json(ARTIFACT_DIR / "vera_confirmatory_abstract_numbers.json")
    alternative = (
        report.get("theory_forced_abstention_lead_verified") is True
        and report.get("unsupported_camelyon_abstention_verified") is True
    )
    passed = protocol.status == "pass" or alternative
    return gate(
        "requested_goal_5_memorable_number",
        "Requested receipted headline or theory lead",
        passed,
        f"X_Y_Z_pass={protocol.status == 'pass'}; theory_forced_abstention_alternative={alternative}; {protocol.evidence}",
        "Verify X/Y/Z with at least a 15-point gap, or verify the preregistered theory plus forced-abstention alternative against receipts.",
    )


def requested_presentation_gate(protocol: Gate) -> Gate:
    report = load_json(ARTIFACT_DIR / "presentation_readiness_audit.json")
    three_panels = int(report.get("figure_1_panel_count", 0)) == 3
    broad_naming_clean = (
        int(report.get("repository_forbidden_name_hit_count", -1)) == 0
        and report.get("repository_naming_oversized_unscanned") == []
    )
    passed = protocol.status == "pass" and three_panels and broad_naming_clean
    return gate(
        "requested_goal_6_presentation",
        "Requested seven-page presentation package",
        passed,
        f"registered_presentation_pass={protocol.status == 'pass'}; figure_1_panel_count={report.get('figure_1_panel_count')}; repository_forbidden_hits={report.get('repository_forbidden_name_hit_count')}; oversized_unscanned={len(report.get('repository_naming_oversized_unscanned', []))}; {protocol.evidence}",
        "Pass the full presentation audit, including an explicit three-panel Figure 1 check and zero forbidden-name hits.",
    )


def requested_external_review_gate(protocol: Gate) -> Gate:
    return gate(
        "requested_goal_7_external_review",
        "Requested two human cold reviews",
        protocol.status == "pass",
        f"registered_external_review_pass={protocol.status == 'pass'}; {protocol.evidence}",
        protocol.required_next,
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
        "scientific_content_human_verified",
        "authorship_criteria_human_confirmed",
        "ai_assistance_disclosure_human_confirmed",
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


def collect_registered_gates() -> list[Gate]:
    return [
        theory_gate(),
        theory_data_gate(),
        killer_experiment_gate(),
        baselines_gate(),
        memorable_number_gate(),
        presentation_gate(),
        external_review_gate(),
    ]


def collect_requested_gates(registered: list[Gate]) -> list[Gate]:
    by_key = {item.key: item for item in registered}
    return [
        requested_theory_gate(by_key["goal_1_shift_aware_theory"]),
        requested_theory_data_gate(by_key["goal_2_theory_matched_by_data"]),
        requested_killer_experiment_gate(),
        requested_baselines_gate(),
        requested_memorable_number_gate(by_key["goal_5_memorable_number"]),
        requested_presentation_gate(by_key["goal_6_presentation"]),
        requested_external_review_gate(
            by_key["goal_7_external_adversarial_review"]
        ),
    ]


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# VERA Exact Goal Completion Audit",
        "",
        f"Generated at UTC: `{report['created_at_utc']}`",
        f"Goal complete: `{report['goal_complete']}`",
        f"Literal requested bar complete: `{report['requested_bar_complete']}`",
        f"Registered protocol complete: `{report['registered_protocol_complete']}`",
        "",
        "> This audit is fail-closed. A stronger replacement is recorded separately; it does not silently check a literal requested box. This audit does not predict acceptance or substitute for peer review.",
        "",
        "## Literal Requested Bar",
        "",
        "| Status | Gate | Evidence | Required next |",
        "| --- | --- | --- | --- |",
    ]
    for item in report["requested_bar_gates"]:
        lines.append(
            f"| {item['status']} | `{item['key']}`: {item['title']} | "
            f"{item['evidence']} | {item['required_next']} |"
        )
    lines.extend(
        [
            "",
            "## Registered Scientific Protocol",
            "",
            "| Status | Gate | Evidence | Required next |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in report["registered_protocol_gates"]:
        lines.append(
            f"| {item['status']} | `{item['key']}`: {item['title']} | "
            f"{item['evidence']} | {item['required_next']} |"
        )
    submission = report["submission_gate"]
    lines.extend(
        [
            "",
            "## Submission Machinery",
            "",
            f"- **{submission['status']}** `{submission['key']}`: {submission['evidence']}",
            "",
            "## Declared Replacements",
            "",
        ]
    )
    for item in report["declared_replacements"]:
        lines.append(f"- `{item['requested']}` -> `{item['registered']}`: {item['reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    registered = collect_registered_gates()
    requested = collect_requested_gates(registered)
    submission = submission_gate()
    all_gates = requested + registered + [submission]
    pass_count = sum(item.status == "pass" for item in all_gates)
    fail_count = len(all_gates) - pass_count
    requested_complete = all(item.status == "pass" for item in requested)
    registered_complete = all(item.status == "pass" for item in registered)
    goal_complete = requested_complete and registered_complete and submission.status == "pass"
    report = {
        "name": "VERA exact user-goal completion audit",
        "schema_version": 3,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "goal_complete": goal_complete,
        "paper_goals_complete": requested_complete and registered_complete,
        "requested_bar_complete": requested_complete,
        "registered_protocol_complete": registered_complete,
        "acceptance_guaranteed": False,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "requested_bar_gates": [asdict(item) for item in requested],
        "registered_protocol_gates": [asdict(item) for item in registered],
        "submission_gate": asdict(submission),
        "gates": [asdict(item) for item in all_gates],
        "declared_replacements": [
            {
                "requested": "at least five claim-grade seeds",
                "registered": "seeds 5-12 as untouched confirmatory runs",
                "reason": "Eight untouched seeds exceed the requested minimum; seeds 0-4 informed protocol design and are exploratory.",
            },
        ],
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(DEFAULT_MD, report)
    print("VERA exact-goal completion audit complete")
    print(f"goal_complete={str(goal_complete).lower()}")
    print(f"requested_bar_complete={str(requested_complete).lower()}")
    print(f"registered_protocol_complete={str(registered_complete).lower()}")
    print(f"pass={pass_count} fail={fail_count}")
    print(f"report={DEFAULT_JSON}")
    return 0 if args.no_fail or report["goal_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
