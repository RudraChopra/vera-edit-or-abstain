#!/usr/bin/env python3
"""Replay every bridge and outward release bound with exact rational arithmetic."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mosaic_rational_certificate import audit_bridge_exact, audit_release_exact, fraction_decimal
from replay_common import finish, sha256


def audit_pair(original_path: Path, strict_path: Path) -> tuple[list[str], dict[str, object]]:
    original = json.loads(original_path.read_text(encoding="utf-8"))
    strict = json.loads(strict_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if strict.get("original_receipt_sha256") != sha256(original_path):
        failures.append("original receipt hash mismatch")
    original_rows = {row["candidate"]: row for row in original.get("results", [])}
    strict_rows = {row["candidate"]: row for row in strict.get("results", [])}
    if set(original_rows) != set(strict_rows):
        failures.append("candidate sets differ")
    bridge_count = 0
    release_count = 0
    minimum_slack: Fraction | None = None
    for candidate, row in strict_rows.items():
        source_row = original_rows.get(candidate)
        if source_row is None or "optimization_error" in source_row:
            failures.append(f"{candidate}: missing valid original token tables")
            continue
        membership = row.get("bridge_membership")
        if not isinstance(membership, dict):
            failures.append(f"{candidate}: missing strict bridge membership")
            continue
        bridge = audit_bridge_exact(
            source_row["reference_token_counts"],
            source_row["bridge_token_counts"],
            reference_l1_radii=source_row["reference_l1_radii"],
            bridge_l1_radii=source_row["bridge_l1_radii"],
            serialized_labels=membership["labels"],
        )
        bridge_count += 1
        minimum_slack = bridge.minimum_membership_slack if minimum_slack is None else min(minimum_slack, bridge.minimum_membership_slack)
        if bridge.minimum_membership_slack < 0:
            failures.append(f"{candidate}: negative exact bridge slack")
        for release_key in ("release_l2", "release_l4"):
            release = row.get(release_key)
            if not isinstance(release, dict):
                continue
            exact = audit_release_exact(
                source_row["reference_token_counts"],
                reference_l1_radii=source_row["reference_l1_radii"],
                bridge=bridge,
                release_channel=release["release_channel"],
                decoder=release["decoder"],
            )
            release_count += 1
            stored_source = tuple(Fraction(str(float(value))) for value in release["certified_source_advantage_upper"])
            stored_utility = Fraction(str(float(release["certified_worst_conditional_error_upper"])))
            if any(stored < verified for stored, verified in zip(stored_source, exact.source_advantages, strict=True)):
                failures.append(f"{candidate}: {release_key}: source bound rounds inward")
            if stored_utility < exact.worst_conditional_error:
                failures.append(f"{candidate}: {release_key}: utility bound rounds inward")
    return failures, {
        "file": strict_path.name,
        "bridges_replayed": bridge_count,
        "releases_replayed": release_count,
        "minimum_exact_membership_slack": fraction_decimal(minimum_slack) if minimum_slack is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1200, choices=range(1200, 1220))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("workers must be positive")
    original_dir = ROOT / "artifacts" / "certificates" / "original"
    strict_dir = ROOT / "artifacts" / "certificates" / "strict"
    originals = sorted(original_dir.glob("*.json"))
    strict = {path.name: path for path in strict_dir.glob("*.json")}
    failures = []
    summaries: list[dict[str, object] | None] = [None] * len(originals)
    if set(strict) != {path.name for path in originals}:
        failures.append("original and strict receipt sets differ")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        pending = {
            executor.submit(audit_pair, original, strict[original.name]): index
            for index, original in enumerate(originals)
            if original.name in strict
        }
        for future in as_completed(pending):
            index = pending[future]
            pair_failures, summary = future.result()
            failures.extend(f"{originals[index].name}: {failure}" for failure in pair_failures)
            summaries[index] = summary
    files = [summary for summary in summaries if summary is not None]
    bridges = sum(int(row["bridges_replayed"]) for row in files)
    releases = sum(int(row["releases_replayed"]) for row in files)
    if bridges != 1300 or releases != 1400:
        failures.append(f"expected 1300 bridges and 1400 releases, found {bridges} and {releases}")
    if failures:
        raise SystemExit("exact-rational audit failed\n" + "\n".join(failures))
    claims = {
        "expected": "1,300 bridge certificates and 1,400 outward release bounds",
        "files_replayed": len(files),
        "bridges_replayed": bridges,
        "releases_replayed": releases,
        "failures": [],
    }
    finish("exact_rational_audit", args.seed, claims, args.output)


if __name__ == "__main__":
    main()
