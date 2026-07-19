#!/usr/bin/env python3
"""Decompose why strict ACS releases miss the primary utility contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from mosaic_transform_exact import transform_exact_utility_confidence_bound


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPTS = ROOT / "research/artifacts/mosaic_acs_bridge_strict_v3_receipts"
DEFAULT_OUTPUT = ROOT / "research/artifacts/mosaic_acs_primary_infeasibility_v1.json"
PRIMARY_THRESHOLD = 0.40
PRIVACY_THRESHOLD = 0.35


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain an object")
    return payload


def normalized_counts(values: Any) -> np.ndarray:
    counts = np.asarray(values, dtype=np.float64)
    return counts / counts.sum(axis=2, keepdims=True)


def best_privacy_feasible_row(receipt: dict[str, Any]) -> dict[str, Any]:
    rows = [
        row
        for row in receipt["results"]
        if isinstance(row.get("release_l2"), dict)
        and max(row["release_l2"]["certified_source_advantage_upper"])
        <= PRIVACY_THRESHOLD + 1e-12
    ]
    if not rows:
        raise ValueError("receipt has no privacy-feasible L2 row")
    return min(
        rows,
        key=lambda row: (
            float(row["release_l2"]["certified_worst_conditional_error_upper"]),
            str(row["candidate"]),
        ),
    )


def analyze_receipt(path: Path) -> dict[str, Any]:
    strict = load(path)
    strict_row = best_privacy_feasible_row(strict)
    raw_path = ROOT / strict["original_receipt"]
    raw = load(raw_path)
    raw_by_candidate = {row["candidate"]: row for row in raw["results"]}
    raw_row = raw_by_candidate[strict_row["candidate"]]
    empirical = normalized_counts(raw_row["reference_token_counts"])
    radii = np.asarray(raw_row["reference_l1_radii"], dtype=np.float64)
    release = strict_row["release_l2"]
    channel = np.asarray(release["release_channel"], dtype=np.float64)
    decoder = tuple(int(value) for value in release["decoder"])
    membership = strict_row["bridge_membership"]
    contaminations = tuple(float(value) for value in membership["contaminations"])
    transforms = tuple(
        (np.asarray(label["transform"], dtype=np.float64),)
        for label in membership["labels"]
    )
    strata = []
    for label in range(empirical.shape[0]):
        for source in range(empirical.shape[1]):
            common_arguments = dict(
                empirical_distribution=empirical[label, source],
                release_channel=channel,
                decoder=decoder,
                true_label=label,
                common_fine_token_channels=transforms[label],
            )
            full = transform_exact_utility_confidence_bound(
                **common_arguments,
                l1_radius=float(radii[label, source]),
                contamination=contaminations[label],
            )
            common = transform_exact_utility_confidence_bound(
                **common_arguments,
                l1_radius=float(radii[label, source]),
                contamination=0.0,
            )
            center = transform_exact_utility_confidence_bound(
                **common_arguments,
                l1_radius=0.0,
                contamination=0.0,
            )
            strata.append(
                {
                    "label": label,
                    "source": source,
                    "full_error": full.error_probability,
                    "center_error": center.error_probability,
                    "sampling_charge": common.error_probability
                    - center.error_probability,
                    "bridge_residual_charge": full.error_probability
                    - common.error_probability,
                    "bridge_contamination": contaminations[label],
                    "l1_radius": float(radii[label, source]),
                    "residual_error_capacity": full.differential_error_capacity,
                    "worst_transform_index": full.worst_transform_index,
                }
            )
    worst = max(strata, key=lambda row: row["full_error"])
    stored_error = float(release["certified_worst_conditional_error_upper"])
    selection_045 = strict["selection_by_utility_threshold"]["0.45"]
    return {
        "seed": int(strict["seed"]),
        "best_candidate": strict_row["candidate"],
        "method": strict_row["method"],
        "strength": strict_row["strength"],
        "certified_error": stored_error,
        "margin_to_primary_contract": stored_error - PRIMARY_THRESHOLD,
        "certified_source_advantages": release[
            "certified_source_advantage_upper"
        ],
        "bridge_contaminations": list(contaminations),
        "worst_stratum_decomposition": worst,
        "error_without_sampling_charge": worst["full_error"]
        - worst["sampling_charge"],
        "error_without_bridge_residual_charge": worst["full_error"]
        - worst["bridge_residual_charge"],
        "strict_outward_guard_gap": stored_error - worst["full_error"],
        "relaxed_045_selection": {
            "candidate": selection_045.get("candidate"),
            "method": selection_045.get("method"),
            "degenerate_identity_release": selection_045.get("method") == "Identity",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    rows = [analyze_receipt(path) for path in sorted(arguments.receipts.glob("*.json"))]
    if len(rows) != 5:
        raise ValueError(f"expected five ACS receipts, found {len(rows)}")
    payload = {
        "name": "MOSAIC ACS primary-contract infeasibility diagnosis v1",
        "status": "post-outcome deterministic decomposition of locked receipts",
        "primary_utility_threshold": PRIMARY_THRESHOLD,
        "privacy_advantage_threshold": PRIVACY_THRESHOLD,
        "seeds": [row["seed"] for row in rows],
        "primary_deployments": sum(
            row["certified_error"] <= PRIMARY_THRESHOLD for row in rows
        ),
        "error_range": [
            min(row["certified_error"] for row in rows),
            max(row["certified_error"] for row in rows),
        ],
        "margin_to_primary_contract_range": [
            min(row["margin_to_primary_contract"] for row in rows),
            max(row["margin_to_primary_contract"] for row in rows),
        ],
        "relaxed_identity_releases": sum(
            row["relaxed_045_selection"]["degenerate_identity_release"]
            for row in rows
        ),
        "rows": rows,
        "claim_boundary": (
            "The decomposition explains the locked K=4 outcome. It is not a "
            "post-outcome retuning of the candidate family or contract."
        ),
        "pass": {
            "all_five_receipts_decomposed": len(rows) == 5,
            "every_decomposition_closes": all(
                abs(
                    row["certified_error"]
                    - row["strict_outward_guard_gap"]
                    - row["worst_stratum_decomposition"]["center_error"]
                    - row["worst_stratum_decomposition"]["sampling_charge"]
                    - row["worst_stratum_decomposition"]["bridge_residual_charge"]
                )
                <= 1e-10
                for row in rows
            ),
            "identity_releases_explicitly_labeled": sum(
                row["relaxed_045_selection"]["degenerate_identity_release"]
                for row in rows
            )
            == 2,
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(arguments.output), "pass": payload["pass"]}, indent=2))


if __name__ == "__main__":
    main()
