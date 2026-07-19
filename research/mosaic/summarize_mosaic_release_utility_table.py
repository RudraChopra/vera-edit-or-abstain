#!/usr/bin/env python3
"""Produce manuscript-ready numerical summaries of released-interface utility."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "research/artifacts/mosaic_release_utility_v2.json"
DEFAULT_OUTPUT = ROOT / "research/artifacts/mosaic_release_utility_table_v1.json"
METRICS = (
    ("released_interface", "expected_balanced_accuracy", "Released interface"),
    ("four_bin_tokenizer_before_channel", "balanced_accuracy", "4-bin tokenizer"),
    (
        "full_feature_classifier_on_selected_edit",
        "balanced_accuracy",
        "Full edited features",
    ),
    (
        "full_feature_classifier_on_unedited_representation",
        "balanced_accuracy",
        "Unedited features",
    ),
)


def summarize(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    if array.size == 1:
        half_width = 0.0
        standard_deviation = 0.0
    else:
        standard_deviation = float(np.std(array, ddof=1))
        half_width = float(
            t.ppf(0.975, df=array.size - 1)
            * standard_deviation
            / np.sqrt(array.size)
        )
    return {
        "n": int(array.size),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "mean_t95_interval": [max(0.0, mean - half_width), min(1.0, mean + half_width)],
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    report = json.loads(arguments.input.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in report["results"]:
        if row["reconstruction"]["diagnostic_token_count_receipt_match"] is not True:
            raise ValueError("utility row does not match its locked diagnostic token table")
        grouped[str(row["dataset"])].append(row)
    datasets: dict[str, Any] = {}
    for dataset, rows in sorted(grouped.items()):
        datasets[dataset] = {
            "releases": len(rows),
            "metrics": {
                label: summarize([float(row[section][key]) for row in rows])
                for section, key, label in METRICS
            },
        }
    payload = {
        "name": "MOSAIC diagnostic released-interface numerical utility table v1",
        "status": "post-outcome numerical summary of the locked-table-matched v2 audit",
        "datasets": datasets,
        "interval": "two-sided 95% t interval over released jobs; descriptive, not a population guarantee",
        "claim_boundary": report["claim_boundary"],
        "pass": {
            "all_23_releases_summarized": sum(
                row["releases"] for row in datasets.values()
            )
            == 23,
            "two_datasets": len(datasets) == 2,
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(arguments.output), "pass": payload["pass"]}, indent=2))


if __name__ == "__main__":
    main()
