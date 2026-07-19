#!/usr/bin/env python3
"""Independent structural and exact-rational audit of natural-shift receipts."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from mosaic_rational_certificate import (
    RationalBridgeAudit,
    RationalBridgeLabelAudit,
    audit_bridge_exact,
    audit_release_exact,
    normalized_fraction_rows,
)
from mosaic_real import sha256


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
DEFAULT_PREREG = ROOT / "prereg_mosaic_acs_natural_shift_v1.json"
DEFAULT_DATA_LOCK = ROOT / "prereg_mosaic_acs_natural_shift_data_v1.json"
DEFAULT_RECEIPTS = REPOSITORY / "research/artifacts/mosaic_acs_natural_shift_v1_receipts"
DEFAULT_OUTPUT = REPOSITORY / "research/artifacts/mosaic_acs_natural_shift_v1_audit.json"


def key(payload: dict[str, Any]) -> tuple[str, str, int]:
    return str(payload["task"]), str(payload["target_state"]), int(payload["seed"])


def identity_bridge(token_count: int) -> RationalBridgeAudit:
    transform = normalized_fraction_rows(
        [[int(row == column) for column in range(token_count)] for row in range(token_count)]
    )
    label = RationalBridgeLabelAudit(
        retained_mass=Fraction(1),
        contamination=Fraction(0),
        minimum_membership_slack=Fraction(0),
        transform=transform,
    )
    return RationalBridgeAudit(labels=(label, label), minimum_membership_slack=Fraction(0))


def outward(release: dict[str, Any], exact: Any) -> list[str]:
    failures = []
    stored_source = [Fraction(str(float(value))) for value in release["certified_source_advantage_upper"]]
    stored_utility = Fraction(str(float(release["certified_worst_conditional_error_upper"])))
    for label, (stored, verified) in enumerate(zip(stored_source, exact.source_advantages, strict=True)):
        if stored < verified:
            failures.append(f"source bound {label} rounds inward")
    if stored_utility < exact.worst_conditional_error:
        failures.append("utility bound rounds inward")
    return failures


def audit_selection(rows: list[dict[str, Any]], alphabet: dict[str, Any], rule: str, threshold: str) -> list[str]:
    release_key = f"{rule}_release"
    eligible = [
        row
        for row in rows
        if isinstance(row.get(release_key), dict)
        and bool(row[release_key]["threshold_decisions"][threshold]["deployed"])
    ]
    stored = alphabet["selection_by_rule_and_threshold"][rule][threshold]
    if not eligible:
        return [] if stored["decision"] == "abstain" and stored["candidate"] is None else ["stored release should abstain"]
    selected = min(
        eligible,
        key=lambda row: (
            float(row[release_key]["certified_worst_conditional_error_upper"]),
            str(row["candidate"]),
        ),
    )
    failures = []
    if stored["decision"] != "deploy" or stored["candidate"] != selected["candidate"]:
        failures.append(f"selection mismatch: expected {selected['candidate']}, stored {stored.get('candidate')}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--data-lock", type=Path, default=DEFAULT_DATA_LOCK)
    parser.add_argument("--receipts", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    prereg = json.loads(args.prereg.read_text(encoding="utf-8"))
    prereg_sha = sha256(args.prereg)
    data_lock = json.loads(args.data_lock.read_text(encoding="utf-8"))
    if data_lock["preregistration_sha256"] != prereg_sha:
        raise ValueError("data lock references the wrong preregistration")
    expected = {
        (str(row["task"]), str(row["target_state"]), int(row["seed"]))
        for row in prereg["jobs"]
    }
    receipts = sorted(args.receipts.glob("ACS-*.json"))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in receipts]
    observed = {key(payload) for payload in payloads}
    failures: list[str] = []
    if len(payloads) != len(expected) or observed != expected:
        failures.append("receipt set differs from registered jobs")
    bridges_replayed = 0
    releases_replayed = 0
    selections_replayed = 0
    candidate_rows = 0
    optimization_errors = 0
    thresholds = [f"{float(value):.2f}" for value in prereg["protocol"]["utility_thresholds"]]
    for payload in payloads:
        prefix = ":".join(map(str, key(payload)))
        if payload.get("preregistration_sha256") != prereg_sha:
            failures.append(f"{prefix}: wrong preregistration hash")
        partition = payload["puma_partition"]
        if not partition["disjoint"] or set(partition["bridge"]) & set(partition["diagnostic"]):
            failures.append(f"{prefix}: PUMA partition overlaps")
        locked_store = data_lock["stores"][f"{payload['task']}:CA->{payload['target_state']}"]
        if payload["store_manifest_sha256"] != locked_store["manifest_sha256"]:
            failures.append(f"{prefix}: store manifest differs from data lock")
        for alphabet_name, alphabet in payload["alphabets"].items():
            token_count = int(alphabet_name)
            rows = alphabet["rows"]
            candidate_rows += len(rows)
            if len(rows) != int(prereg["protocol"]["frontier_candidate_count"]):
                failures.append(f"{prefix}:K={token_count}: incomplete candidate frontier")
            for rule in ("mosaic", "direct"):
                for threshold in thresholds:
                    for failure in audit_selection(rows, alphabet, rule, threshold):
                        failures.append(f"{prefix}:K={token_count}:{rule}:{threshold}: {failure}")
                    selections_replayed += 1
            for row in rows:
                candidate = str(row["candidate"])
                if "optimization_error" in row:
                    optimization_errors += 1
                    continue
                try:
                    bridge = audit_bridge_exact(
                        row["reference_table"]["token_counts"],
                        row["bridge_table"]["token_counts"],
                        reference_l1_radii=row["reference_table"]["l1_radii"],
                        bridge_l1_radii=row["bridge_table"]["l1_radii"],
                        serialized_labels=row["bridge_membership"]["labels"],
                    )
                    bridges_replayed += 1
                    if bridge.minimum_membership_slack < 0:
                        failures.append(f"{prefix}:K={token_count}:{candidate}: negative bridge slack")
                    mosaic_exact = audit_release_exact(
                        row["reference_table"]["token_counts"],
                        reference_l1_radii=row["reference_table"]["l1_radii"],
                        bridge=bridge,
                        release_channel=row["mosaic_release"]["release_channel"],
                        decoder=row["mosaic_release"]["decoder"],
                    )
                    releases_replayed += 1
                    for failure in outward(row["mosaic_release"], mosaic_exact):
                        failures.append(f"{prefix}:K={token_count}:{candidate}:mosaic: {failure}")
                    direct_exact = audit_release_exact(
                        row["bridge_table"]["token_counts"],
                        reference_l1_radii=row["bridge_table"]["l1_radii"],
                        bridge=identity_bridge(token_count),
                        release_channel=row["direct_release"]["release_channel"],
                        decoder=row["direct_release"]["decoder"],
                    )
                    releases_replayed += 1
                    for failure in outward(row["direct_release"], direct_exact):
                        failures.append(f"{prefix}:K={token_count}:{candidate}:direct: {failure}")
                except (KeyError, TypeError, ValueError) as error:
                    failures.append(f"{prefix}:K={token_count}:{candidate}: audit error: {error}")
    report = {
        "name": "MOSAIC multi-state ACS exact-rational and structural audit",
        "passed": not failures,
        "preregistration_sha256": prereg_sha,
        "data_lock_sha256": sha256(args.data_lock),
        "receipts_replayed": len(payloads),
        "candidate_rows": candidate_rows,
        "optimization_errors": optimization_errors,
        "bridges_replayed": bridges_replayed,
        "releases_replayed": releases_replayed,
        "selections_replayed": selections_replayed,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
