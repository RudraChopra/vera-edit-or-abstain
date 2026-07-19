#!/usr/bin/env python3
"""Render the MOSAIC finite-alphabet scaling study."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[2]
REPORT = REPOSITORY / "research/artifacts/mosaic_scaling_study_v1.json"
OUTPUT = ROOT / "figures/figure5_mosaic_scaling"

INK = "#202124"
MUTED = "#5F6368"
GRID = "#DADCE0"
GREEN = "#009E73"
BLUE = "#0072B2"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
COLORS = {2: BLUE, 3: ORANGE, 4: GREEN}
MARKERS = {2: "o", 3: "s", 4: "D"}


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=8.4,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
        bbox={"boxstyle": "square,pad=0.18", "facecolor": INK, "edgecolor": INK},
    )


def style(ax: plt.Axes, label: str) -> None:
    panel_label(ax, label)
    ax.grid(axis="y", color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)


def indexed(report: dict) -> dict[tuple[int, int], dict]:
    return {
        (int(row["token_count"]), int(row["source_count"])): row
        for row in report["summary"]
    }


def runtime_panel(ax: plt.Axes, cells: dict[tuple[int, int], dict]) -> None:
    tokens = (4, 8, 16, 32, 64)
    for sources in (2, 3, 4):
        values = [cells[(token, sources)]["wall_clock_seconds_median"] for token in tokens]
        ax.plot(
            tokens,
            values,
            color=COLORS[sources],
            marker=MARKERS[sources],
            linewidth=1.5,
            markersize=4,
            label=f"G={sources}",
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(tokens, [str(value) for value in tokens])
    ax.set_xlabel("Fine-token alphabet K")
    ax.set_ylabel("Median solve time (seconds)")
    ax.set_title("Exact certification remains subsecond", loc="left")
    ax.legend(frameon=False, fontsize=5.8, ncol=3, loc="upper left")


def slack_panel(ax: plt.Axes, cells: dict[tuple[int, int], dict]) -> None:
    tokens = (4, 8, 16, 32, 64)
    for sources in (2, 3, 4):
        values = [cells[(token, sources)]["utility_slack_median"] for token in tokens]
        ax.plot(
            tokens,
            values,
            color=COLORS[sources],
            marker=MARKERS[sources],
            linewidth=1.5,
            markersize=4,
            label=f"G={sources}",
        )
    ax.axhline(0.0, color=INK, linewidth=0.7)
    ax.set_xscale("log", base=2)
    ax.set_xticks(tokens, [str(value) for value in tokens])
    ax.set_ylim(-0.01, 0.33)
    ax.set_xlabel("Fine-token alphabet K")
    ax.set_ylabel("Median utility slack")
    ax.set_title("Sampling cost grows smoothly with K", loc="left")


def retention_panel(ax: plt.Axes, cells: dict[tuple[int, int], dict]) -> None:
    tokens = (4, 8, 16, 32, 64)
    thresholds = (0.10, 0.12, 0.15, 0.20, 0.40)
    matrix = np.asarray(
        [
            [
                np.mean(
                    [
                        cells[(token, sources)]["retention_by_utility_threshold"][
                            f"{threshold:.2f}"
                        ]
                        for sources in (2, 3, 4)
                    ]
                )
                for token in tokens
            ]
            for threshold in thresholds
        ]
    )
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="cividis", aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(
                column,
                row,
                f"{100 * value:.0f}",
                ha="center",
                va="center",
                fontsize=5.8,
                color="white" if value < 0.52 else INK,
            )
    ax.set_xticks(np.arange(len(tokens)), [str(value) for value in tokens])
    ax.set_yticks(np.arange(len(thresholds)), [f"{value:.2f}" for value in thresholds])
    ax.set_xlabel("Fine-token alphabet K")
    ax.set_ylabel("Utility contract")
    ax.set_title("Retention across contract strength", loc="left")
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    colorbar.set_ticks([0, 0.5, 1])
    colorbar.set_ticklabels(["0", "50", "100"])
    colorbar.set_label("Deployed (%)", fontsize=6.3)


def sample_panel(ax: plt.Axes, cells: dict[tuple[int, int], dict], report: dict) -> None:
    tokens = (4, 8, 16, 32, 64)
    rows = [cells[(token, 4)] for token in tokens]
    radii = [row["l1_radius"] for row in rows]
    sample_sizes = [row["required_n_for_k4_radius"] for row in rows]
    ax.plot(tokens, radii, color=PURPLE, marker="o", linewidth=1.5, markersize=4)
    ax.set_xscale("log", base=2)
    ax.set_xticks(tokens, [str(value) for value in tokens])
    ax.set_ylim(0.06, 0.24)
    ax.set_xlabel("Fine-token alphabet K")
    ax.set_ylabel("Weissman L1 radius", color=PURPLE)
    ax.tick_params(axis="y", colors=PURPLE)
    twin = ax.twinx()
    twin.plot(tokens, sample_sizes, color=BLUE, marker="s", linewidth=1.3, markersize=3.8)
    twin.set_ylabel("n for K=4 radius", color=BLUE)
    twin.tick_params(axis="y", colors=BLUE)
    twin.spines[["top", "left"]].set_visible(False)
    hard = report["hard_constraint_generation_point"]
    ax.set_title("More states require more table support", loc="left")
    ax.text(
        0.02,
        0.93,
        f"Column generation: {hard['active_attacker_assignments']}/{hard['full_attacker_assignments']} active",
        transform=ax.transAxes,
        fontsize=5.8,
        color=MUTED,
        va="top",
    )


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if not all(report["pass"].values()):
        raise ValueError("scaling report has a failed gate")
    cells = indexed(report)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.5,
            "axes.titlesize": 8.2,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.08, 4.75), facecolor="white")
    runtime_panel(axes[0, 0], cells)
    slack_panel(axes[0, 1], cells)
    retention_panel(axes[1, 0], cells)
    sample_panel(axes[1, 1], cells, report)
    for axis, label in zip(axes.flat, "abcd"):
        style(axis, label)
    figure.subplots_adjust(
        left=0.09, right=0.92, top=0.94, bottom=0.12, wspace=0.42, hspace=0.52
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    figure.savefig(
        OUTPUT.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
