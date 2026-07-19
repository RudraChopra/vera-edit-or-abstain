#!/usr/bin/env python3
"""Audit that MOSAIC's disclosed v2 correction stayed within its locked scope."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from mosaic_real import sha256
from run_mosaic_official_frontier_exact_confirmation import atomic_json_dump


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _has_structural_zero(transform: object) -> bool:
    matrix = np.asarray(transform, dtype=np.float64)
    return bool(np.any(np.all(matrix == 0.0, axis=0)))


def compare_pair(
    v1: dict[str, object], v2: dict[str, object]
) -> tuple[list[str], dict[str, object]]:
    failures: list[str] = []
    for key in (
        "dataset",
        "seed",
        "protocol",
        "original_receipt_sha256",
        "preregistration_sha256",
    ):
        if v1.get(key) != v2.get(key):
            failures.append(f"immutable top-level field changed: {key}")
    v1_rows = {str(row["candidate"]): row for row in v1.get("results", [])}
    v2_rows = {str(row["candidate"]): row for row in v2.get("results", [])}
    if set(v1_rows) != set(v2_rows):
        failures.append("candidate set changed")

    changed_labels = 0
    structural_zero_labels = 0
    restored_labels = 0
    compared_labels = 0
    unchanged_rows_with_identical_release = 0
    maximum_retained_gain = 0.0
    for candidate in sorted(set(v1_rows) & set(v2_rows)):
        row1 = v1_rows[candidate]
        row2 = v2_rows[candidate]
        for key in ("candidate", "method", "strength", "provenance"):
            if row1.get(key) != row2.get(key):
                failures.append(f"{candidate}: candidate metadata changed: {key}")
        membership1 = row1.get("bridge_membership")
        membership2 = row2.get("bridge_membership")
        if not isinstance(membership1, dict) or not isinstance(membership2, dict):
            failures.append(f"{candidate}: missing bridge membership")
            continue
        labels1 = membership1.get("labels", [])
        labels2 = membership2.get("labels", [])
        if len(labels1) != len(labels2):
            failures.append(f"{candidate}: label count changed")
            continue
        row_changed = False
        for label_index, (label1, label2) in enumerate(
            zip(labels1, labels2, strict=True)
        ):
            compared_labels += 1
            transform = label1.get("transform")
            structural_zero = _has_structural_zero(transform)
            structural_zero_labels += int(structural_zero)
            for key in ("transform", "optimal_retained_mass_upper", "transform_trace"):
                if label1.get(key) != label2.get(key):
                    failures.append(
                        f"{candidate}: label {label_index}: base bridge field changed: {key}"
                    )
            retained1 = float(label1["retained_mass"])
            retained2 = float(label2["retained_mass"])
            if retained2 < retained1:
                failures.append(
                    f"{candidate}: label {label_index}: v2 retained mass decreased"
                )
            if retained2 != retained1:
                row_changed = True
                changed_labels += 1
                maximum_retained_gain = max(
                    maximum_retained_gain, retained2 - retained1
                )
                if not structural_zero:
                    failures.append(
                        f"{candidate}: label {label_index}: retained mass changed "
                        "without a structural-zero output"
                    )
                restored_labels += int(retained1 == 0.0 and retained2 > 0.0)
        if not row_changed and row1.get("release_l2") != row2.get("release_l2"):
            failures.append(f"{candidate}: unchanged bridge changed the L=2 release")
        if not row_changed and row1.get("release_l2") == row2.get("release_l2"):
            unchanged_rows_with_identical_release += 1

    policy1 = v1.get("numerical_policy", {})
    policy2 = v2.get("numerical_policy", {})
    for key in (
        "bridge_feasibility_guard",
        "release_optimization_guard",
        "reported_value_guard",
        "decision_tolerance",
    ):
        if policy1.get(key) != policy2.get(key):
            failures.append(f"numerical guard changed: {key}")
    return failures, {
        "dataset": v2.get("dataset"),
        "seed": v2.get("seed"),
        "candidate_rows": len(v2_rows),
        "labels_compared": compared_labels,
        "structural_zero_labels": structural_zero_labels,
        "changed_labels": changed_labels,
        "zero_to_positive_labels": restored_labels,
        "maximum_retained_mass_gain": maximum_retained_gain,
        "unchanged_rows_with_identical_l2_release": (
            unchanged_rows_with_identical_release
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-dir", required=True, type=Path)
    parser.add_argument("--v2-dir", required=True, type=Path)
    parser.add_argument("--v1-audit", required=True, type=Path)
    parser.add_argument("--v2-audit", required=True, type=Path)
    parser.add_argument("--v2-rational-audit", required=True, type=Path)
    parser.add_argument("--v2-amendment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    amendment_hash = sha256(args.v2_amendment)
    amendment_sidecar = args.v2_amendment.with_suffix(
        args.v2_amendment.suffix + ".sha256"
    )
    if amendment_sidecar.read_text(encoding="utf-8").strip() != amendment_hash:
        raise ValueError("v2 amendment sidecar mismatch")

    audit_paths = {
        "v1_strict": args.v1_audit,
        "v2_strict": args.v2_audit,
        "v2_rational": args.v2_rational_audit,
    }
    failures: list[str] = []
    audit_receipts: dict[str, object] = {}
    for name, path in audit_paths.items():
        audit = _load(path)
        if audit.get("passed") is not True:
            failures.append(f"required audit did not pass: {name}")
        audit_receipts[name] = {"path": str(path), "sha256": sha256(path)}

    v1_paths = {path.name: path for path in args.v1_dir.glob("*.json")}
    v2_paths = {path.name: path for path in args.v2_dir.glob("*.json")}
    if len(v1_paths) != 100 or set(v1_paths) != set(v2_paths):
        failures.append("v1 and v2 directories do not contain the same 100 receipts")
    summaries: list[dict[str, object]] = []
    for name in sorted(set(v1_paths) & set(v2_paths)):
        pair_failures, summary = compare_pair(
            _load(v1_paths[name]), _load(v2_paths[name])
        )
        failures.extend(f"{name}: {failure}" for failure in pair_failures)
        summaries.append(summary)

    deployment_counts: dict[str, dict[str, int]] = {
        "v1": Counter(),
        "v2": Counter(),
    }
    v2_false_acceptances: Counter[str] = Counter()
    for version, paths in (("v1", v1_paths), ("v2", v2_paths)):
        for path in paths.values():
            payload = _load(path)
            for threshold, selection in payload[
                "selection_by_utility_threshold"
            ].items():
                deployment_counts[version][str(threshold)] += int(
                    selection.get("decision") == "deploy"
                )
                if version == "v2":
                    v2_false_acceptances[str(threshold)] += int(
                        bool(selection.get("false_acceptance"))
                    )

    report: dict[str, object] = {
        "name": "MOSAIC v2 structural-zero correction scope audit",
        "passed": not failures,
        "v2_amendment_sha256": amendment_hash,
        "audit_receipts": audit_receipts,
        "files_compared": len(summaries),
        "candidate_rows_compared": sum(
            int(summary["candidate_rows"]) for summary in summaries
        ),
        "labels_compared": sum(
            int(summary["labels_compared"]) for summary in summaries
        ),
        "structural_zero_labels": sum(
            int(summary["structural_zero_labels"]) for summary in summaries
        ),
        "changed_labels": sum(
            int(summary["changed_labels"]) for summary in summaries
        ),
        "zero_to_positive_labels": sum(
            int(summary["zero_to_positive_labels"]) for summary in summaries
        ),
        "maximum_retained_mass_gain": max(
            (float(summary["maximum_retained_mass_gain"]) for summary in summaries),
            default=0.0,
        ),
        "v1_deployments_by_threshold": dict(deployment_counts["v1"]),
        "v2_deployments_by_threshold": dict(deployment_counts["v2"]),
        "v2_false_acceptances_by_threshold": dict(v2_false_acceptances),
        "failures": failures,
        "files": summaries,
    }
    atomic_json_dump(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
