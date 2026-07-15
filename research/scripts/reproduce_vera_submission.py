"""Reproduce and audit VERA's submission-facing result package in one command."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
OUTPUT = ROOT / "artifacts" / "vera_one_command_reproduction.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(script: str, *arguments: str) -> dict[str, object]:
    command = [sys.executable, str(ROOT / "scripts" / script), *arguments]
    result = subprocess.run(
        command,
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{script} failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return {
        "script": script,
        "arguments": list(arguments),
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-2000:],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Recompute candidate/rule rows from external per-example arrays before "
            "building the package. The default compact mode replays and audits "
            "the frozen derived rows shipped in the anonymous archive."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    steps: list[dict[str, object]] = []
    steps.append(run("audit_exact_balanced_simulation.py"))
    steps.append(run("audit_exact_family_grid_simulation.py"))
    if args.full:
        steps.append(
            run(
                "audit_official_eraser_receipts.py",
                "--prereg",
                "research/prereg_confirmatory_balanced.json",
                "--hash-file",
                "research/prereg_confirmatory_balanced.sha256",
                "--receipt-dir",
                "research/artifacts/confirmatory_balanced_receipts",
                "--output",
                "research/artifacts/confirmatory_balanced_receipt_audit.json",
            )
        )
        steps.append(run("analyze_vera_confirmatory_balanced.py"))
    steps.append(
        run(
            "audit_vera_confirmatory_analysis.py"
            if args.full
            else "audit_vera_confirmatory_compact.py"
        )
    )
    if args.full:
        steps.append(
            run(
                "audit_official_eraser_receipts.py",
                "--prereg",
                "research/prereg_independent_stress_replication.json",
                "--hash-file",
                "research/prereg_independent_stress_replication.sha256",
                "--receipt-dir",
                "research/artifacts/independent_stress_replication_receipts",
                "--output",
                "research/artifacts/independent_stress_replication_receipt_audit.json",
            )
        )
        steps.append(run("analyze_vera_independent_stress_replication.py"))
        steps.append(run("audit_vera_independent_stress_replication.py"))
    steps.append(run("audit_vera_independent_stress_compact.py"))
    if args.full:
        steps.append(run("analyze_vera_learning_curve_diagnostic.py"))
        steps.append(run("analyze_vera_confirmatory_ablations.py"))
    steps.append(run("build_vera_confirmatory_results.py"))
    steps.append(run("build_vera_independent_stress_package.py"))
    steps.append(run("audit_frozen_references.py"))

    expected = (
        ROOT
        / "maintrack"
        / "aaai2027_template"
        / "AuthorKit27"
        / "vera_main_results_table.tex"
    )
    if not expected.is_file():
        raise RuntimeError("main result table was not generated")
    report = {
        "name": "VERA one-command submission reproduction",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "mode": "full_external_arrays" if args.full else "compact_frozen_rows",
        "steps": steps,
        "main_table_sha256": sha256(expected),
    }
    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
