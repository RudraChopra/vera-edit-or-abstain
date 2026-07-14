"""Inventory upstream erasure-baseline reference repositories for FARO.

This audit does not claim that every upstream implementation has been run under
matched FARO splits. It records which exact official-code repositories are
available locally and which baselines remain proxy or paper-only comparisons.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"

DEFAULT_JSON = ARTIFACT_DIR / "upstream_baseline_reference_inventory.json"
DEFAULT_MD = ARTIFACT_DIR / "upstream_baseline_reference_inventory.md"


@dataclass(frozen=True)
class RepoSpec:
    baseline: str
    status_role: str
    path: Path
    expected_remote_fragment: str
    key_files: tuple[str, ...]
    allowed_claim: str


@dataclass(frozen=True)
class Check:
    baseline: str
    status: str
    evidence: str
    allowed_claim: str


REPOS = [
    RepoSpec(
        baseline="MANCE++",
        status_role="official upstream reference implementation",
        path=Path("/Volumes/Backups/FARO/external/mance"),
        expected_remote_fragment="MatanAvitan/mance",
        key_files=("mance/erasure.py", "mance/tangent.py", "mance/scorer.py"),
        allowed_claim=(
            "Official-code FARO adapter exists; Waterbirds is claim-grade, "
            "and Camelyon17 has a full no-cap claim-grade receipt."
        ),
    ),
    RepoSpec(
        baseline="R-LACE",
        status_role="official upstream reference implementation available",
        path=Path("/Volumes/Backups/FARO/external/rlace-icml"),
        expected_remote_fragment="shauli-ravfogel/rlace-icml",
        key_files=("rlace.py", "debias.py", "classifier.py"),
        allowed_claim=(
            "Official upstream code is pinned locally; current FARO tables may "
            "only claim R-LACE-style proxy stress tests until matched receipts exist."
        ),
    ),
    RepoSpec(
        baseline="TaCo",
        status_role="official upstream reference implementation available",
        path=Path("/Volumes/Backups/FARO/external/TaCo"),
        expected_remote_fragment="fanny-jourdan/TaCo",
        key_files=("TaCo/TaCo.py", "TaCo/concept_removal.py", "TaCo/run_TaCo.py"),
        allowed_claim=(
            "Official upstream code is pinned locally; current FARO tables may "
            "only claim TaCo-style proxy stress tests until matched receipts exist."
        ),
    ),
    RepoSpec(
        baseline="LEACE",
        status_role="official upstream reference implementation available",
        path=Path("/Volumes/Backups/FARO/external/concept-erasure"),
        expected_remote_fragment="EleutherAI/concept-erasure",
        key_files=("concept_erasure/leace.py", "tests/test_leace.py", "pyproject.toml"),
        allowed_claim=(
            "Official upstream code is pinned locally; current FARO tables may "
            "only claim LEACE-style proxy stress tests until matched receipts exist."
        ),
    ),
]


PAPER_ONLY_BASELINES = [
    {
        "baseline": "SPLINCE/SPLICE",
        "status_role": "paper-only in current local packet",
        "evidence": (
            "No official upstream repository is pinned in "
            "/Volumes/Backups/FARO/external as of this audit."
        ),
        "allowed_claim": (
            "Only SPLINCE/SPLICE-style proxy stress-test language is allowed "
            "until an exact upstream implementation is identified and run."
        ),
    }
]


def materialized_file(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    stat = path.stat()
    return not (stat.st_size > 0 and getattr(stat, "st_blocks", 1) == 0)


def git_value(path: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def inspect_repo(spec: RepoSpec) -> tuple[Check, dict[str, Any]]:
    commit = git_value(spec.path, "rev-parse", "HEAD")
    remote = git_value(spec.path, "config", "--get", "remote.origin.url")
    missing_files = [
        rel for rel in spec.key_files if not materialized_file(spec.path / rel)
    ]
    path_ok = spec.path.exists() and spec.path.is_dir()
    remote_ok = spec.expected_remote_fragment.lower() in remote.lower()
    commit_ok = bool(commit)
    files_ok = not missing_files
    passed = path_ok and remote_ok and commit_ok and files_ok
    evidence = (
        f"path={spec.path}; commit={commit or '<missing>'}; remote={remote or '<missing>'}; "
        f"missing_key_files={missing_files}"
    )
    check = Check(
        baseline=spec.baseline,
        status="pass" if passed else "fail",
        evidence=evidence,
        allowed_claim=spec.allowed_claim,
    )
    record = {
        "baseline": spec.baseline,
        "status_role": spec.status_role,
        "path": str(spec.path),
        "commit": commit,
        "remote": remote,
        "key_files": list(spec.key_files),
        "missing_key_files": missing_files,
        "official_upstream_available": passed,
        "allowed_claim": spec.allowed_claim,
    }
    return check, record


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Upstream Baseline Reference Inventory",
        "",
        f"Generated at UTC: `{report['created_at_utc']}`",
        f"Inventory ready: `{report['inventory_ready']}`",
        "",
        "## Official-Code Repositories",
        "",
        "| Status | Baseline | Commit | Remote | Allowed claim |",
        "| --- | --- | --- | --- | --- |",
    ]
    for repo in report["repositories"]:
        status = "pass" if repo["official_upstream_available"] else "fail"
        lines.append(
            f"| {status} | {repo['baseline']} | `{repo['commit'] or '<missing>'}` | "
            f"{repo['remote'] or '<missing>'} | {repo['allowed_claim']} |"
        )
    lines.extend(
        [
            "",
            "## Paper-Only or Proxy Baselines",
            "",
            "| Baseline | Evidence | Allowed claim |",
            "| --- | --- | --- |",
        ]
    )
    for item in report["paper_only_or_proxy_baselines"]:
        lines.append(
            f"| {item['baseline']} | {item['evidence']} | {item['allowed_claim']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MD)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    checks: list[Check] = []
    repos: list[dict[str, Any]] = []
    for spec in REPOS:
        check, record = inspect_repo(spec)
        checks.append(check)
        repos.append(record)

    fail_count = sum(check.status == "fail" for check in checks)
    report = {
        "name": "FARO upstream baseline reference inventory",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inventory_ready": fail_count == 0,
        "full_reference_parity_claim_allowed": False,
        "pass_count": sum(check.status == "pass" for check in checks),
        "fail_count": fail_count,
        "checks": [asdict(check) for check in checks],
        "repositories": repos,
        "paper_only_or_proxy_baselines": PAPER_ONLY_BASELINES,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.markdown_out, report)
    print("FARO upstream baseline reference inventory complete")
    print(f"inventory_ready={str(report['inventory_ready']).lower()}")
    print(f"fail_count={fail_count}")
    print(f"report={args.json_out}")
    return 0 if args.no_fail or fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
