"""Build a deterministic, byte-faithful VERA anonymous artifact archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import zipfile
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
RESEARCH = REPOSITORY / "research"
DEFAULT_OUTPUT = (
    Path(os.environ.get("VERA_RELEASE_DIR", REPOSITORY / "dist"))
    / "vera_anonymous_submission.zip"
)
ZIP_TIME = (2026, 7, 14, 0, 0, 0)
CORE_FILES = (
    "README.md",
    "research/prereg_confirmatory_balanced.json",
    "research/prereg_confirmatory_balanced.sha256",
    "research/prereg_exact_family_grid.json",
    "research/prereg_exact_family_grid.sha256",
    "research/prereg_real_learning_curve_diagnostic.json",
    "research/prereg_real_learning_curve_diagnostic.sha256",
    "research/prereg_confirmatory_secondary_ablations.json",
    "research/prereg_confirmatory_secondary_ablations.sha256",
    "research/prereg_independent_stress_replication.json",
    "research/prereg_independent_stress_replication.sha256",
    "research/reference_manifest.json",
    "research/maintrack/appendix_shift_robust_theory.tex",
    "research/maintrack/references_verified.bib",
    "research/maintrack/CODE_AVAILABILITY_ANONYMOUS.md",
    "research/maintrack/CLAIM_LEDGER.md",
    "research/maintrack/NOVELTY_LOCK.md",
    "research/maintrack/STATISTICAL_INTEGRITY.md",
    "research/maintrack/REPRODUCIBILITY_CHECKLIST.md",
)

SCRIPT_FILES = (
    "research/scripts/reproduce_vera_submission.py",
    "research/scripts/audit_exact_balanced_simulation.py",
    "research/scripts/run_exact_family_grid_simulation.py",
    "research/scripts/audit_exact_family_grid_simulation.py",
    "research/scripts/audit_official_eraser_receipts.py",
    "research/scripts/analyze_vera_attacker_ablation.py",
    "research/scripts/analyze_vera_balanced_existing.py",
    "research/scripts/analyze_vera_confirmatory_balanced.py",
    "research/scripts/audit_vera_confirmatory_analysis.py",
    "research/scripts/audit_vera_confirmatory_compact.py",
    "research/scripts/analyze_vera_learning_curve_diagnostic.py",
    "research/scripts/analyze_vera_confirmatory_ablations.py",
    "research/scripts/analyze_vera_secondary_ablations.py",
    "research/scripts/analyze_vera_real_study.py",
    "research/scripts/run_parallel_real_study_matrix.py",
    "research/scripts/analyze_vera_independent_stress_replication.py",
    "research/scripts/audit_vera_independent_stress_replication.py",
    "research/scripts/audit_vera_independent_stress_compact.py",
    "research/scripts/build_vera_independent_stress_package.py",
    "research/scripts/build_vera_confirmatory_results.py",
    "research/scripts/audit_frozen_references.py",
    "research/scripts/official_eraser_adapters.py",
    "research/scripts/run_official_eraser_frontier.py",
    "research/scripts/run_real_study_matrix.py",
    "research/scripts/vera_robust_certificate.py",
    "research/tests/test_vera_analysis.py",
    "research/tests/test_vera_robust_certificate.py",
)

ARTIFACT_FILES = (
    "research/artifacts/confirmatory_balanced_receipt_audit.json",
    "research/artifacts/vera_exact_balanced_report.json",
    "research/artifacts/vera_exact_balanced_audit.json",
    "research/artifacts/vera_exact_family_grid_report.json",
    "research/artifacts/vera_exact_family_grid_audit.json",
    "research/artifacts/vera_confirmatory_balanced_rule_rows.csv",
    "research/artifacts/vera_confirmatory_balanced_candidate_rows.csv",
    "research/artifacts/vera_confirmatory_balanced_report.json",
    "research/artifacts/vera_confirmatory_abstract_numbers.json",
    "research/artifacts/vera_confirmatory_analysis_audit.json",
    "research/artifacts/vera_confirmatory_results_package_audit.json",
    "research/artifacts/vera_learning_curve_diagnostic.json",
    "research/artifacts/vera_confirmatory_ablation_rows.csv",
    "research/artifacts/vera_confirmatory_ablation_report.json",
    "research/artifacts/independent_stress_replication_receipt_audit.json",
    "research/artifacts/vera_independent_stress_rule_rows.csv",
    "research/artifacts/vera_independent_stress_candidate_rows.csv",
    "research/artifacts/vera_independent_stress_report.json",
    "research/artifacts/vera_independent_stress_abstract_numbers.json",
    "research/artifacts/vera_independent_stress_analysis_audit.json",
    "research/artifacts/vera_independent_stress_compact_audit.json",
    "research/artifacts/vera_independent_stress_package_audit.json",
    "research/artifacts/reference_verification_report.json",
)

PAPER_FILES = (
    "research/maintrack/aaai2027_template/AuthorKit27/vera_aaai2027_anonymous.tex",
    "research/maintrack/aaai2027_template/AuthorKit27/vera_aaai2027_anonymous.pdf",
    "research/maintrack/aaai2027_template/AuthorKit27/vera_aaai2027_supplement_anonymous.tex",
    "research/maintrack/aaai2027_template/AuthorKit27/vera_aaai2027_supplement_anonymous.pdf",
    "research/maintrack/aaai2027_template/AuthorKit27/vera_paper_body.tex",
    "research/maintrack/aaai2027_template/AuthorKit27/vera_supplement_body.tex",
    "research/maintrack/aaai2027_template/AuthorKit27/vera_results_macros.tex",
    "research/maintrack/aaai2027_template/AuthorKit27/vera_main_results_table.tex",
    "research/maintrack/aaai2027_template/AuthorKit27/vera_main_results_narrative.tex",
    "research/maintrack/aaai2027_template/AuthorKit27/vera_supplement_results.tex",
    "research/maintrack/aaai2027_template/AuthorKit27/vera_ablation_results.tex",
    "research/maintrack/aaai2027_template/AuthorKit27/vera_family_grid_results.tex",
    "research/maintrack/aaai2027_template/AuthorKit27/vera_independent_stress_results.tex",
    "research/maintrack/aaai2027_template/AuthorKit27/aaai2027.sty",
    "research/maintrack/aaai2027_template/AuthorKit27/aaai2027.bst",
)

FIGURE_FILES = (
    "research/maintrack/figures/vera_method_overview.pdf",
    "research/maintrack/figures/vera_method_overview.png",
    "research/maintrack/figures/vera_exact_theory_match.pdf",
    "research/maintrack/figures/vera_exact_theory_match.png",
    "research/maintrack/figures/vera_deployment_rules.pdf",
    "research/maintrack/figures/vera_deployment_rules.png",
    "research/maintrack/figures/vera_real_learning_curve.pdf",
    "research/maintrack/figures/vera_real_learning_curve.png",
    "research/maintrack/figures/vera_independent_stress_replication.pdf",
    "research/maintrack/figures/vera_independent_stress_replication.png",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def required_paths() -> list[Path]:
    relative = list(CORE_FILES + SCRIPT_FILES + ARTIFACT_FILES + PAPER_FILES + FIGURE_FILES)
    receipts = sorted(
        (RESEARCH / "artifacts" / "confirmatory_balanced_receipts").glob("*.json")
    )
    if len(receipts) != 200:
        raise RuntimeError(f"expected 200 confirmatory receipts, found {len(receipts)}")
    independent_receipts = sorted(
        (RESEARCH / "artifacts" / "independent_stress_replication_receipts").glob(
            "*.json"
        )
    )
    if len(independent_receipts) != 800:
        raise RuntimeError(
            f"expected 800 independent stress receipts, found {len(independent_receipts)}"
        )
    paths = [REPOSITORY / value for value in relative] + receipts + independent_receipts
    missing = [str(path.relative_to(REPOSITORY)) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("anonymous archive inputs are missing:\n" + "\n".join(missing))
    return sorted(set(paths), key=lambda path: path.relative_to(REPOSITORY).as_posix())


def generated_files() -> dict[str, bytes]:
    readme = """# VERA Anonymous Reproduction Archive

This archive contains the locked preregistrations, the independently replayed
216-cell theorem coverage grid, 200 design-stage official-code receipts, the
800-run disjoint-seed independent stress replication, frozen candidate- and
decision-level rows, independent audits, anonymous paper, supplement, and
figures.

OpenAI Codex assisted extensively with research ideation, literature discovery,
theorem and proof drafting, implementation, experiment orchestration,
statistical analysis, figures, and manuscript drafting. It is not an author or
a citable source; human authors retain responsibility for complete independent
verification and policy compliance.

Run the compact, no-dataset replay from the archive root:

```bash
python research/scripts/reproduce_vera_submission.py
```

The compact replay requires Python 3.10+ with NumPy, SciPy, and Matplotlib. The
raw third-party datasets and large per-example arrays are intentionally excluded.
Compact mode independently replays every candidate selection, aggregate, and
headline from the frozen rows. Candidate metrics are anchored by the included
full raw-array audit and receipt hashes; the raw arrays themselves are excluded.
To use `--full`, mount those arrays at the immutable paths in the locked
preregistration. Those historical paths are retained byte-for-byte so the
preregistration hash remains verifiable; they are not author identifiers.
"""
    requirements = "matplotlib\nnumpy\nscipy\n"
    return {
        "ANONYMOUS_README.md": readme.encode("utf-8"),
        "requirements.txt": requirements.encode("utf-8"),
    }


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def build(output: Path) -> dict[str, object]:
    payload: dict[str, bytes] = {
        path.relative_to(REPOSITORY).as_posix(): path.read_bytes()
        for path in required_paths()
    }
    payload.update(generated_files())
    entries = [
        {"path": name, "bytes": len(data), "sha256": sha256(data)}
        for name, data in sorted(payload.items())
    ]
    manifest = {
        "archive": "VERA anonymous submission artifact",
        "format_version": 1,
        "source_commit": source_commit(),
        "payload_file_count": len(entries),
        "confirmatory_receipt_count": sum(
            entry["path"].startswith(
                "research/artifacts/confirmatory_balanced_receipts/"
            )
            for entry in entries
        ),
        "independent_stress_receipt_count": sum(
            entry["path"].startswith(
                "research/artifacts/independent_stress_replication_receipts/"
            )
            for entry in entries
        ),
        "entries": entries,
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        for name, data in sorted(payload.items()):
            archive.writestr(zip_info(name), data)
        archive.writestr(zip_info("MANIFEST.json"), manifest_bytes)
    return {
        "passed": True,
        "output": str(output),
        "archive_sha256": sha256(output.read_bytes()),
        "payload_file_count": len(entries),
        "confirmatory_receipt_count": manifest["confirmatory_receipt_count"],
        "independent_stress_receipt_count": manifest[
            "independent_stress_receipt_count"
        ],
        "source_commit": manifest["source_commit"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    report = build(parse_args().output.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
