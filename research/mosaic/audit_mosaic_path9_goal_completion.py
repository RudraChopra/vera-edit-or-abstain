#!/usr/bin/env python3
"""Audit every requested path-to-9 scientific objective from frozen evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "research/artifacts"
DEFAULT_CAM_SUMMARY = (
    ARTIFACTS / "mosaic_camelyon_streamed_confirmation_v1/summary.json"
)
DEFAULT_CAM_AUDIT = (
    ARTIFACTS / "mosaic_camelyon_streamed_confirmation_audit_v1.json"
)
DEFAULT_OUTPUT = ARTIFACTS / "mosaic_path9_goal_completion_audit.json"


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def audit_passed(payload: dict[str, Any]) -> bool:
    return bool(payload.get("pass") is True or payload.get("passed") is True)


def collect_checks(
    *,
    cam_summary_path: Path = DEFAULT_CAM_SUMMARY,
    cam_audit_path: Path = DEFAULT_CAM_AUDIT,
) -> dict[str, bool]:
    acs = load(ARTIFACTS / "mosaic_acs_scalar_confirmation_v1.json")
    acs_audit = load(
        ARTIFACTS / "mosaic_acs_scalar_confirmation_audit_v1.json"
    )
    qwen = load(
        ARTIFACTS / "mosaic_qwen_powered_confirmation_v1/summary.json"
    )
    qwen_audit = load(
        ARTIFACTS / "mosaic_qwen_powered_confirmation_audit_v1.json"
    )
    residual = load(ARTIFACTS / "mosaic_residual_sharpness_v1.json")
    residual_audit = load(
        ARTIFACTS / "mosaic_residual_sharpness_audit_v1.json"
    )
    proxy = load(
        ARTIFACTS / "mosaic_real_proxy_mass_confirmation_v1.json"
    )
    proxy_audit = load(
        ARTIFACTS / "mosaic_real_proxy_mass_confirmation_audit_v1.json"
    )
    local_dp = load(ARTIFACTS / "mosaic_local_dp_baseline_v1.json")
    local_dp_audit = load(
        ARTIFACTS / "mosaic_local_dp_baseline_audit_v1.json"
    )
    original_breadth = load(
        ARTIFACTS / "mosaic_bridge_confirmation_manifest_v1.json"
    )
    cam = load(cam_summary_path)
    cam_audit = load(cam_audit_path)

    acs_summary = acs["summary"]
    residual_summary = residual["summary"]
    proxy_curve = proxy["calibration_curve"]
    local_dp_summary = local_dp["summary"]
    return {
        "natural_acs_audit_passes": audit_passed(acs_audit),
        "natural_acs_direct_deployed_mosaic_abstained": all(
            row["direct_decision_2018"] == "deploy"
            and row["mosaic_decision_2018"] == "abstain"
            for row in acs["rows"]
        ),
        "natural_acs_has_familywise_confirmed_failure": (
            acs_summary["familywise_confirmed_2023_utility_violations"] >= 1
        ),
        "qwen_audit_passes": audit_passed(qwen_audit),
        "qwen_gate_passes": qwen["main_paper_inclusion_gate_pass"] is True,
        "qwen_releases_five_of_five": (
            qwen["primary_releases"] == 5
            and qwen["registered_jobs"] == 5
        ),
        "qwen_zero_heldout_and_operational_violations": (
            qwen["heldout_primary_violations"] == 0
            and qwen["operational_primary_trials"] == 500
            and qwen["operational_primary_violations"] == 0
        ),
        "residual_audit_passes": audit_passed(residual_audit),
        "residual_floor_dominates_sampling_in_all_jobs": (
            residual_summary["jobs"] == 35
            and residual_summary[
                "jobs_residual_exceeds_sampling_for_source"
            ]
            == 35
            and residual_summary[
                "jobs_residual_exceeds_sampling_for_utility"
            ]
            == 35
        ),
        "proxy_audit_passes": audit_passed(proxy_audit),
        "proxy_curve_abstains_then_releases": (
            [row["decision"] for row in proxy_curve]
            == ["abstain", "abstain", "deploy", "deploy"]
        ),
        "proxy_full_certificate_is_safe_and_nonconstant": (
            proxy["gates"]["full_calibration_certificate_releases"] is True
            and proxy["gates"]["released_interface_is_diagnostically_safe"]
            is True
            and proxy["release"]["release_channel"][0]
            != proxy["release"]["release_channel"][1]
        ),
        "local_dp_audit_passes": audit_passed(local_dp_audit),
        "local_dp_matched_comparison_covers_all_jobs": (
            local_dp_summary["jobs"] == 35
            and local_dp_summary["local_dp_deployments"] == 0
            and local_dp_summary["mosaic_deployments"] == 35
            and local_dp_summary["mosaic_strictly_lower_error_jobs"] == 35
        ),
        "camelyon_audit_passes": audit_passed(cam_audit),
        "camelyon_gate_passes": (
            cam["main_paper_inclusion_gate_passed"] is True
            and cam["primary_release_count"] >= 3
            and cam["primary_heldout_violation_count"] == 0
            and cam["operational_violation_count"] == 0
        ),
        "breadth_reaches_three_release_domains": (
            qwen["primary_releases"] > 0
            and cam["primary_release_count"] > 0
            and original_breadth["deployment_by_dataset_and_threshold"][
                "BiasBios-Clinical"
            ]["0.40"]["deployments"]
            > 0
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cam-summary", type=Path, default=DEFAULT_CAM_SUMMARY)
    parser.add_argument("--cam-audit", type=Path, default=DEFAULT_CAM_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checks = collect_checks(
        cam_summary_path=args.cam_summary,
        cam_audit_path=args.cam_audit,
    )
    failures = [name for name, passed in checks.items() if not passed]
    report = {
        "name": "MOSAIC path-to-9 objective completion audit",
        "checks": checks,
        "failures": failures,
        "passed": not failures,
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
