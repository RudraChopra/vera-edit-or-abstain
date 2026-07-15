"""Fail closed when the theorem-to-code contract is incomplete or stale."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
DEFAULT_MAPPING = ROOT / "theory_code_mapping.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "vera_theory_code_mapping_audit.json"
REQUIRED_COLUMNS = (
    "component",
    "theorem_variable",
    "code_variable",
    "data_source",
    "range",
    "confidence_level",
    "candidate_multiplicity",
    "environment_multiplicity",
    "attacker_multiplicity",
    "source_class_multiplicity",
    "exact_interval_implementation",
    "missing_cell_treatment",
    "identity_treatment",
    "receipt_output_field",
)
EXPECTED_COMPONENTS = {
    "paired_target_harm",
    "balanced_attacker_leakage",
    "fixed_profile_candidate_iut",
    "vector_shift_envelope",
    "controlled_shift_membership",
}
REQUIRED_SOURCE_FRAGMENTS = {
    "paired_target_harm": (
        "exact_discrete_risk_certificate",
        "target_environment_radii",
    ),
    "balanced_attacker_leakage": (
        "exact_balanced_leakage_profile_certificate",
        "class_{source_class}_probability_upper",
    ),
    "fixed_profile_candidate_iut": (
        "certify_balanced_iut_profile",
        "candidate_alpha = delta / candidate_count",
    ),
    "vector_shift_envelope": (
        "certify_balanced_shift_envelope",
        "balanced_profile_in_envelope",
        "simultaneous_curve_parameters",
    ),
    "controlled_shift_membership": (
        "design_controlled_shift_from_fold",
        "conditional_density_ratio_profile",
        "membership_verified",
    ),
}
REQUIRED_THEORY_LABELS = (
    "\\label{thm:iut}",
    "\\label{thm:shift-envelope}",
    "\\label{cor:common-radius}",
    "\\label{thm:sample-complexity-upper}",
    "\\label{thm:sample-complexity-lower}",
    "\\label{thm:unsupported}",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_mapping(
    mapping: Mapping[str, Any],
    *,
    theory_text: str,
    implementation_text: str,
    analysis_text: str,
) -> list[str]:
    failures: list[str] = []
    if mapping.get("schema_version") != 1:
        failures.append("schema_version must equal 1")
    rows = mapping.get("rows")
    if not isinstance(rows, list):
        return failures + ["rows must be a list"]
    components = [row.get("component") for row in rows if isinstance(row, dict)]
    if set(components) != EXPECTED_COMPONENTS or len(components) != len(EXPECTED_COMPONENTS):
        failures.append("component set is incomplete or duplicated")
    source_text = implementation_text + "\n" + analysis_text
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            failures.append(f"row {index} is not an object")
            continue
        for column in REQUIRED_COLUMNS:
            if column not in row or row[column] in (None, "", []):
                failures.append(f"row {index} missing {column}")
        component = row.get("component")
        for fragment in REQUIRED_SOURCE_FRAGMENTS.get(str(component), ()):
            if fragment not in source_text:
                failures.append(f"{component} source fragment absent: {fragment}")
    for label in REQUIRED_THEORY_LABELS:
        if label not in theory_text:
            failures.append(f"theory label absent: {label}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    theory_path = REPOSITORY / str(mapping["theory_source"])
    implementation_path = REPOSITORY / str(mapping["implementation_source"])
    analysis_path = REPOSITORY / str(mapping["analysis_source"])
    controlled_shift_path = REPOSITORY / str(mapping["controlled_shift_source"])
    failures = audit_mapping(
        mapping,
        theory_text=theory_path.read_text(encoding="utf-8"),
        implementation_text=implementation_path.read_text(encoding="utf-8"),
        analysis_text=(
            analysis_path.read_text(encoding="utf-8")
            + "\n"
            + controlled_shift_path.read_text(encoding="utf-8")
        ),
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "name": "VERA theorem-to-code mapping audit",
        "passed": not failures,
        "git_commit": head,
        "mapping_sha256": sha256(args.mapping),
        "theory_sha256": sha256(theory_path),
        "implementation_sha256": sha256(implementation_path),
        "analysis_sha256": sha256(analysis_path),
        "controlled_shift_sha256": sha256(controlled_shift_path),
        "row_count": len(mapping.get("rows", [])),
        "required_column_count": len(REQUIRED_COLUMNS),
        "failures": failures,
        "formal_proof_verified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
