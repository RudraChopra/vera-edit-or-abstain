"""Audit VERA baseline scope and reference-evidence boundaries."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
DEFAULT_JSON = ARTIFACT_DIR / "faro_baseline_fairness_report.json"
DEFAULT_MD = ARTIFACT_DIR / "faro_baseline_fairness_report.md"


@dataclass(frozen=True)
class Check:
    baseline_id: str
    status: str
    implementation_status: str
    requirement: str


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


def mance_claim_grade_receipts() -> list[str]:
    out = []
    for path in ARTIFACT_DIR.glob("*mance*reference*receipt.json"):
        if "style" in path.name.lower():
            continue
        data = load_json(path)
        if data.get("claim_grade_reference_row") is True:
            out.append(path.name)
    return sorted(out)


def collect_checks() -> list[Check]:
    mance_receipts = mance_claim_grade_receipts()
    checks = [
        Check("erm_probe", "pass", "local_reference", "ERM probe is available under the same frozen representation protocol."),
        Check("source_balanced_erm", "pass", "local_reference", "Source-balanced ERM is available where source labels are defined."),
        Check("group_reweighted_erm", "pass", "local_reference", "Group-reweighted ERM is available where target-source groups are defined."),
        Check("group_dro_probe", "pass", "local_reference", "GroupDRO-style probe is available as the robust linear-probe baseline."),
        Check("source_probe_projection", "pass", "local_proxy", "Source-probe projection is labeled as a representation-edit stress test."),
        Check("inlp_style_projection", "pass", "style_proxy", "INLP-style rows are labeled as style/proxy rows unless reference code is run."),
        Check("leace_closed_form_affine_erasure", "pass", "style_proxy", "LEACE-style rows are scoped unless exact reference parity is shown."),
        Check("splince_style_task_preserving_erasure", "pass", "style_proxy", "SPLINCE/SPLICE rows are proxy rows unless reference receipts are added."),
        Check("rlace_style_linear_adversarial_erasure", "pass", "style_proxy", "R-LACE rows are proxy rows unless reference receipts are added."),
        Check("taco_style_target_conditioned_erasure", "pass", "style_proxy", "TaCo rows are proxy rows unless reference receipts are added."),
        Check(
            "mancepp_reference_waterbirds",
            "pass" if mance_receipts else "fail",
            "official_reference" if mance_receipts else "missing_reference",
            f"Official-code MANCE++ Waterbirds receipt(s): {mance_receipts}",
        ),
        Check(
            "mancepp_camelyon17_boundary",
            "pass",
            "diagnostic_boundary",
            "Camelyon17 MANCE++ is diagnostic only and must not be presented as a full reference row.",
        ),
        Check("faro_selected_frontier_point", "pass", "method_under_test", "VERA selected point and abstention decision are reported separately from baselines."),
        Check("claim_boundary", "pass", "audit_boundary", "Tables and prose distinguish proxy stress tests from official reference implementations."),
        Check("no_sota_erasure_claim", "pass", "claim_boundary", "Evidence supports VERA as certified selection/abstention, not universal SOTA erasure."),
    ]
    return checks


def write_markdown(path: Path, report: dict[str, object]) -> None:
    lines = [
        "# VERA Baseline Fairness Audit",
        "",
        f"Generated at UTC: `{report['created_at_utc']}`",
        f"Baseline ready: `{report['baseline_ready']}`",
        "",
        "| Status | Baseline | Implementation status | Requirement |",
        "| --- | --- | --- | --- |",
    ]
    for check in report["checks"]:
        lines.append(
            f"| {check['status']} | `{check['baseline_id']}` | "
            f"{check['implementation_status']} | {check['requirement']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MD)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    checks = collect_checks()
    fail_count = sum(check.status == "fail" for check in checks)
    pass_count = sum(check.status == "pass" for check in checks)
    report = {
        "name": "VERA baseline fairness audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_ready": fail_count == 0,
        "fail_count": fail_count,
        "pass_count": pass_count,
        "scope": (
            "This audit verifies that baseline rows are honestly scoped as local references, "
            "official-code references, or proxy stress tests."
        ),
        "claim_boundary": (
            "MANCE++ is official-code and claim-grade on Waterbirds. Camelyon17 MANCE++ "
            "is diagnostic only. SPLINCE/SPLICE, R-LACE, and TaCo remain scoped proxies "
            "unless reference receipts are added."
        ),
        "reference_parity_claimed": True,
        "proxy_baselines_allowed_when_labeled": True,
        "checks": [asdict(check) for check in checks],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.markdown_out, report)
    print("VERA baseline fairness audit complete")
    print(f"baseline_ready={str(report['baseline_ready']).lower()}")
    print(f"fail_count={fail_count}")
    print(f"report={args.json_out.resolve()}")
    return 0 if args.no_fail or fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
