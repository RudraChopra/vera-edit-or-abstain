"""Audit the VERA Paper A reproducibility packet."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
CONFIG_PATH = ROOT / "configs" / "faro_paper_a_reproducibility.json"
CHECKLIST_PATH = ROOT / "maintrack" / "REPRODUCIBILITY_CHECKLIST.md"
DEFAULT_JSON = ARTIFACT_DIR / "reproducibility_packet_audit.json"
DEFAULT_MD = ARTIFACT_DIR / "reproducibility_packet_audit.md"
DEFAULT_CSV = ARTIFACT_DIR / "reproducibility_packet_audit.csv"


@dataclass(frozen=True)
class Check:
    key: str
    status: str
    evidence: str
    requirement: str
    next_step: str


def status(condition: bool) -> str:
    return "pass" if condition else "fail"


def bool_word(value: bool) -> str:
    return "yes" if value else "no"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if not is_materialized_file(path):
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def resolve(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT.parent / candidate


def is_materialized_file(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    stat = path.stat()
    if stat.st_size > 0 and getattr(stat, "st_blocks", 1) == 0:
        return False
    return True


def artifact_exists(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_dir():
        return True
    return is_materialized_file(path)


def script_paths_from_argv(argv: list[Any]) -> list[Path]:
    paths: list[Path] = []
    for token in argv:
        if not isinstance(token, str):
            continue
        if token.startswith("research/scripts/") or token.startswith("research/maintrack/"):
            paths.append(resolve(token))
    return paths


def command_has_existing_references(command: dict[str, Any]) -> bool:
    argv = command.get("argv", [])
    if not isinstance(argv, list) or not argv:
        return False
    paths = script_paths_from_argv(argv)
    return all(artifact_exists(path) for path in paths)


def missing_command_references(command: dict[str, Any]) -> list[str]:
    argv = command.get("argv", [])
    if not isinstance(argv, list) or not argv:
        return ["<empty argv>"]
    return [str(path) for path in script_paths_from_argv(argv) if not artifact_exists(path)]


def expected_outputs_exist(command: dict[str, Any]) -> bool:
    outputs = command.get("expected_outputs", [])
    if not isinstance(outputs, list) or not outputs:
        return False
    return all(isinstance(item, str) and artifact_exists(resolve(item)) for item in outputs)


def missing_expected_outputs(command: dict[str, Any]) -> list[str]:
    outputs = command.get("expected_outputs", [])
    if not isinstance(outputs, list) or not outputs:
        return ["<no expected outputs>"]
    return [
        str(resolve(item))
        for item in outputs
        if not isinstance(item, str) or not artifact_exists(resolve(item))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    manifest = load_json(CONFIG_PATH)
    checklist_text = CHECKLIST_PATH.read_text(encoding="utf-8") if CHECKLIST_PATH.exists() else ""
    commands = manifest.get("reproduction_commands", [])
    claim_rows = manifest.get("claim_ready_rows", [])
    blocked_rows = manifest.get("blocked_required_rows", [])
    core_artifacts = manifest.get("core_artifacts", [])
    seed_policy = manifest.get("seed_policy", {})
    release_policy = manifest.get("public_release_policy", {})

    if not isinstance(commands, list):
        commands = []
    if not isinstance(claim_rows, list):
        claim_rows = []
    if not isinstance(blocked_rows, list):
        blocked_rows = []
    if not isinstance(core_artifacts, list):
        core_artifacts = []

    command_keys = {
        str(command.get("key", ""))
        for command in commands
        if isinstance(command, dict)
    }
    required_command_keys = {
        "benchmark_claim_audit",
        "maintrack_figures",
        "maintrack_readiness",
        "adversarial_internal_review",
        "manuscript_compile",
    }
    all_command_refs_ok = all(
        isinstance(command, dict) and command_has_existing_references(command)
        for command in commands
    )
    all_command_outputs_present = all(
        isinstance(command, dict) and expected_outputs_exist(command)
        for command in commands
    )
    missing_script_refs = {
        str(command.get("key", "")): missing_command_references(command)
        for command in commands
        if isinstance(command, dict) and missing_command_references(command)
    }
    missing_outputs = {
        str(command.get("key", "")): missing_expected_outputs(command)
        for command in commands
        if isinstance(command, dict) and missing_expected_outputs(command)
    }

    row_receipts_present = all(
        isinstance(row, dict)
        and artifact_exists(resolve(str(row.get("receipt", ""))))
        and artifact_exists(resolve(str(row.get("statistics", ""))))
        for row in claim_rows
    )
    missing_claim_receipts = [
        {
            "benchmark": str(row.get("benchmark", "")),
            "receipt": str(resolve(str(row.get("receipt", "")))),
            "statistics": str(resolve(str(row.get("statistics", "")))),
        }
        for row in claim_rows
        if isinstance(row, dict)
        and (
            not artifact_exists(resolve(str(row.get("receipt", ""))))
            or not artifact_exists(resolve(str(row.get("statistics", ""))))
        )
    ]
    blocked_rows_scoped = all(
        isinstance(row, dict)
        and bool(row.get("current_blocker"))
        and bool(row.get("next_action"))
        and artifact_exists(resolve(str(row.get("preflight_report", ""))))
        for row in blocked_rows
    )
    core_artifacts_present = all(
        isinstance(path, str) and artifact_exists(resolve(path))
        for path in core_artifacts
    )

    checks = [
        Check(
            key="manifest_present",
            status=status(bool(manifest)),
            evidence=f"path={CONFIG_PATH}; exists={bool_word(CONFIG_PATH.exists())}",
            requirement="A machine-readable reproducibility manifest must exist.",
            next_step="Create `research/configs/faro_paper_a_reproducibility.json`.",
        ),
        Check(
            key="checklist_present",
            status=status(
                CHECKLIST_PATH.exists()
                and "Reproduction Commands" in checklist_text
                and "Pre-Submission Checklist" in checklist_text
            ),
            evidence=f"path={CHECKLIST_PATH}; exists={bool_word(CHECKLIST_PATH.exists())}",
            requirement="A human-readable reproducibility checklist must exist.",
            next_step="Create or update `research/maintrack/REPRODUCIBILITY_CHECKLIST.md`.",
        ),
        Check(
            key="seed_policy_locked",
            status=status(
                seed_policy.get("official_seed_list") == [0, 1, 2, 3, 4]
                and int(seed_policy.get("minimum_official_seeds", 0)) == 5
                and abs(float(seed_policy.get("confidence_level", 0.0)) - 0.95) < 1e-9
                and bool(seed_policy.get("paired_statistics_required")) is True
            ),
            evidence=(
                f"official_seed_list={seed_policy.get('official_seed_list')}; "
                f"minimum_official_seeds={seed_policy.get('minimum_official_seeds')}; "
                f"confidence_level={seed_policy.get('confidence_level')}; "
                f"paired_statistics_required={seed_policy.get('paired_statistics_required')}"
            ),
            requirement="Official rows need locked five-seed, 95 percent interval, paired-statistics policy.",
            next_step="Update the seed policy in the reproducibility manifest.",
        ),
        Check(
            key="reproduction_commands_complete",
            status=status(required_command_keys.issubset(command_keys) and all_command_refs_ok),
            evidence=(
                f"command_keys={sorted(command_keys)}; "
                f"missing={sorted(required_command_keys - command_keys)}; "
                f"script_refs_ok={bool_word(all_command_refs_ok)}; "
                f"missing_script_refs={missing_script_refs}"
            ),
            requirement="The manifest must list the core reproduction commands and reference existing scripts.",
            next_step="Add missing commands or repair stale script paths in the manifest.",
        ),
        Check(
            key="reproduction_outputs_present",
            status=status(all_command_outputs_present),
            evidence=(
                f"all_expected_outputs_present={bool_word(all_command_outputs_present)}; "
                f"missing_outputs={missing_outputs}"
            ),
            requirement="Core reproduction commands should have current output artifacts.",
            next_step="Run the missing reproduction commands and regenerate their outputs.",
        ),
        Check(
            key="claim_rows_have_receipts",
            status=status(len(claim_rows) >= 2 and row_receipts_present),
            evidence=(
                f"claim_ready_rows={len(claim_rows)}; "
                f"receipts_present={bool_word(row_receipts_present)}; "
                f"missing_claim_receipts={missing_claim_receipts}"
            ),
            requirement="The packet must include at least two durable claim-ready rows, each with existing receipts and statistical reports.",
            next_step="Finish, rehydrate, or regenerate another official benchmark row with materialized receipt and statistical report.",
        ),
        Check(
            key="blocked_rows_scoped",
            status=status(len(blocked_rows) >= 1 and blocked_rows_scoped),
            evidence=f"blocked_rows={len(blocked_rows)}; scoped={bool_word(blocked_rows_scoped)}",
            requirement="Required but incomplete rows must have an explicit blocker and next action.",
            next_step="Add blocker, preflight report, and next action for every required incomplete row.",
        ),
        Check(
            key="core_artifacts_present",
            status=status(core_artifacts_present),
            evidence=f"core_artifact_count={len(core_artifacts)}; present={bool_word(core_artifacts_present)}",
            requirement="The reproducibility packet must point to all core method, paper, and audit artifacts.",
            next_step="Regenerate or relink missing core artifacts.",
        ),
        Check(
            key="release_policy_scoped",
            status=status(
                bool(release_policy.get("release_code_configs_and_small_artifacts")) is True
                and bool(release_policy.get("release_large_raw_data")) is False
                and bool(release_policy.get("anonymization_required_for_review")) is True
            ),
            evidence=(
                f"release_code_configs_and_small_artifacts="
                f"{release_policy.get('release_code_configs_and_small_artifacts')}; "
                f"release_large_raw_data={release_policy.get('release_large_raw_data')}; "
                f"anonymization_required_for_review={release_policy.get('anonymization_required_for_review')}"
            ),
            requirement="Public release policy must avoid committing third-party data and preserve review anonymity.",
            next_step="Update the public release policy in the manifest.",
        ),
    ]

    fail_count = sum(1 for check in checks if check.status == "fail")
    report = {
        "name": "VERA Paper A reproducibility packet audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "packet_ready": fail_count == 0,
        "pass_count": len(checks) - fail_count,
        "fail_count": fail_count,
        "checks": [asdict(check) for check in checks],
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with args.csv_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key", "status", "evidence", "requirement", "next_step"])
        writer.writeheader()
        for check in checks:
            writer.writerow(asdict(check))

    lines = [
        "# VERA Reproducibility Packet Audit",
        "",
        f"Generated at UTC: `{report['created_at_utc']}`",
        "",
        f"- `packet_ready`: {bool_word(report['packet_ready'])}",
        f"- `pass_count`: {report['pass_count']}",
        f"- `fail_count`: {report['fail_count']}",
        "",
        "| Check | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for check in checks:
        lines.append(f"| `{check.key}` | {check.status} | {check.evidence} |")
    args.md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("VERA reproducibility packet audit complete")
    print(f"packet_ready={str(report['packet_ready']).lower()}")
    print(f"fail_count={fail_count}")
    print(f"report={args.json_out}")
    if fail_count and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
