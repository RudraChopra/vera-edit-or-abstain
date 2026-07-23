#!/usr/bin/env python3
"""Build the auditor's sample-size chart for a binary-source MOSAIC contract."""

from __future__ import annotations

import csv
import json
from math import ceil, log
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_JSON = ROOT / "research/artifacts/mosaic_designer_chart_v1.json"
OUTPUT_CSV = ROOT / "research/artifacts/mosaic_designer_chart_v1.csv"
OUTPUT_FIGURE = (
    ROOT / "research/maintrack/mosaic_aaai2027/"
    "mosaic_designer_chart_v1.pdf"
)
ALPHABETS = (2, 4, 8, 16, 32, 64)
CONTRACTS = (0.20, 0.30, 0.40)
SHIFT_BUDGETS = (0.00, 0.05, 0.10, 0.15, 0.20)
NOMINAL_ADVANTAGE = 0.10
FAMILY_FAILURE = 0.05
REGISTERED_STRATA = 4


def weissman_required_n(
    alphabet_size: int,
    statistical_margin: float,
    row_failure_probability: float,
) -> int | None:
    """Sufficient per-row sample size for L1 radius at most the margin."""

    if statistical_margin <= 0.0:
        return None
    log_factor = log(
        (2.0**alphabet_size - 2.0) / row_failure_probability
    )
    return int(ceil(2.0 * log_factor / statistical_margin**2))


def main() -> None:
    rows = []
    row_delta = FAMILY_FAILURE / REGISTERED_STRATA
    for contract in CONTRACTS:
        for shift in SHIFT_BUDGETS:
            statistical_margin = contract - NOMINAL_ADVANTAGE - shift
            for alphabet in ALPHABETS:
                required = weissman_required_n(
                    alphabet, statistical_margin, row_delta
                )
                rows.append(
                    {
                        "contract": contract,
                        "nominal_source_advantage": NOMINAL_ADVANTAGE,
                        "shift_budget": shift,
                        "statistical_margin": statistical_margin,
                        "alphabet_size": alphabet,
                        "required_labels_per_source_label_stratum": required,
                        "family_failure_probability": FAMILY_FAILURE,
                        "registered_strata": REGISTERED_STRATA,
                    }
                )
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "name": "MOSAIC designer sample-size chart v1",
                "interpretation": (
                    "For a binary-source certificate with empirical source "
                    "advantage .10, the table gives a distribution-uniform "
                    "sufficient n per source-label row after charging a shift "
                    "budget rho. Blank cells are population-infeasible because "
                    "tau <= .10 + rho."
                ),
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(
        1, len(CONTRACTS), figsize=(7.0, 2.35), sharey=True
    )
    for axis, contract in zip(axes, CONTRACTS, strict=True):
        matrix = np.full((len(ALPHABETS), len(SHIFT_BUDGETS)), np.nan)
        for row in rows:
            if row["contract"] != contract:
                continue
            i = ALPHABETS.index(row["alphabet_size"])
            j = SHIFT_BUDGETS.index(row["shift_budget"])
            value = row["required_labels_per_source_label_stratum"]
            if value is not None:
                matrix[i, j] = value
        image = axis.imshow(
            np.log10(matrix),
            aspect="auto",
            origin="lower",
            cmap="viridis_r",
            vmin=3.0,
            vmax=5.5,
        )
        axis.set_title(rf"$\tau={contract:.2f}$")
        axis.set_xticks(range(len(SHIFT_BUDGETS)))
        axis.set_xticklabels(
            [f"{value:.2f}" for value in SHIFT_BUDGETS], rotation=45
        )
        axis.set_yticks(range(len(ALPHABETS)))
        axis.set_yticklabels(ALPHABETS)
        axis.set_xlabel(r"shift budget $\rho$")
        for i in range(len(ALPHABETS)):
            for j in range(len(SHIFT_BUDGETS)):
                value = matrix[i, j]
                label = "no" if np.isnan(value) else f"{int(value):,}"
                axis.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="white" if not np.isnan(value) and value > 5000 else "black",
                )
    axes[0].set_ylabel("fine alphabet K")
    colorbar = figure.colorbar(image, ax=axes, fraction=0.025, pad=0.03)
    colorbar.set_label(r"$\log_{10}$ labels per source-label row")
    figure.suptitle(
        "Evidence needed after nominal advantage .10 and shift charge",
        fontsize=9,
    )
    figure.subplots_adjust(left=0.08, right=0.90, bottom=0.22, top=0.78, wspace=0.16)
    figure.savefig(OUTPUT_FIGURE, bbox_inches="tight")
    print(
        json.dumps(
            {
                "json": str(OUTPUT_JSON),
                "csv": str(OUTPUT_CSV),
                "figure": str(OUTPUT_FIGURE),
                "rows": len(rows),
            }
        )
    )


if __name__ == "__main__":
    main()
