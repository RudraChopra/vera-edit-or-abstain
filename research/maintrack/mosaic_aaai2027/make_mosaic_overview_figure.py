#!/usr/bin/env python3
"""Render the three-step MOSAIC overview figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


OUTPUT = Path(__file__).resolve().parent / "figures" / "figure1_mosaic_overview"
BLUE = "#0072B2"
GREEN = "#009E73"
VERMILION = "#D55E00"
INK = "#202124"
MUTED = "#667085"
LIGHT = "#F5F7FA"
BORDER = "#B8C0CC"


def title(ax: plt.Axes, step: str, heading: str, detail: str) -> None:
    ax.text(0.05, 0.89, step, fontsize=8.3, fontweight="bold", color="white", ha="left", va="center", bbox={"boxstyle": "square,pad=0.23", "facecolor": INK, "edgecolor": INK})
    ax.text(0.18, 0.89, heading, fontsize=8.7, fontweight="bold", color=INK, ha="left", va="center")
    ax.text(0.05, 0.74, detail, fontsize=6.5, color=MUTED, ha="left", va="top", linespacing=1.2)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, linewidth=1.1, color=MUTED))


def evidence_panel(ax: plt.Axes) -> None:
    title(ax, "1", "Certify the evidence", "Reference and labeled target-bridge tables\nshare one fixed confidence event.")
    for x, label, color in ((0.08, "reference\ntable", BLUE), (0.56, "target bridge\ntable", GREEN)):
        ax.add_patch(Rectangle((x, 0.30), 0.31, 0.25, facecolor=LIGHT, edgecolor=color, linewidth=1.2))
        for row in range(2):
            for column in range(3):
                ax.add_patch(Rectangle((x + 0.045 + column * 0.071, 0.355 + row * 0.075), 0.055, 0.050, facecolor=mpl.colors.to_rgba(color, 0.22 + 0.16 * (row + column)), edgecolor="white", linewidth=0.5))
        ax.text(x + 0.155, 0.20, label, fontsize=6.7, color=INK, ha="center", va="center")
    arrow(ax, (0.39, 0.425), (0.55, 0.425))
    ax.text(0.50, 0.09, "missing support stops the release", fontsize=6.3, color=VERMILION, ha="center", va="center", fontweight="bold")


def design_panel(ax: plt.Axes) -> None:
    title(ax, "2", "Design the interface", "Search a stochastic channel and decoder\ninside that one shared event.")
    ax.text(0.14, 0.42, "fine\ntoken", fontsize=7.0, color=INK, ha="center", va="center")
    ax.add_patch(Rectangle((0.31, 0.31), 0.25, 0.23, facecolor=mpl.colors.to_rgba(BLUE, 0.11), edgecolor=BLUE, linewidth=1.2))
    ax.text(0.435, 0.425, "channel\n+ decoder", fontsize=7.1, color=INK, ha="center", va="center", fontweight="bold")
    ax.text(0.79, 0.42, "public\ntoken", fontsize=7.0, color=INK, ha="center", va="center")
    arrow(ax, (0.21, 0.42), (0.30, 0.42))
    arrow(ax, (0.57, 0.42), (0.70, 0.42))
    ax.text(0.50, 0.12, "one event covers every channel searched", fontsize=6.5, color=GREEN, ha="center", va="center", fontweight="bold")


def gate_panel(ax: plt.Axes) -> None:
    title(ax, "3", "Gate the release", "Check bridge membership, source distinguishability,\nand worst-stratum task utility.")
    ax.add_patch(Rectangle((0.08, 0.31), 0.35, 0.25, facecolor=mpl.colors.to_rgba(GREEN, 0.10), edgecolor=GREEN, linewidth=1.2))
    ax.text(0.255, 0.435, "all checks pass", fontsize=7.0, color=INK, ha="center", va="center", fontweight="bold")
    ax.text(0.255, 0.355, "persist one token", fontsize=6.5, color=GREEN, ha="center", va="center")
    ax.add_patch(Rectangle((0.57, 0.31), 0.35, 0.25, facecolor=mpl.colors.to_rgba(VERMILION, 0.08), edgecolor=VERMILION, linewidth=1.2))
    ax.text(0.745, 0.435, "any check fails", fontsize=7.0, color=INK, ha="center", va="center", fontweight="bold")
    ax.text(0.745, 0.355, "abstain", fontsize=6.6, color=VERMILION, ha="center", va="center", fontweight="bold")
    ax.text(0.50, 0.12, "the released object is a small, auditable interface", fontsize=6.3, color=MUTED, ha="center", va="center")


def main() -> None:
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axes = plt.subplots(1, 3, figsize=(7.08, 1.72))
    for axis in axes:
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.axis("off")
        axis.add_patch(Rectangle((0.006, 0.006), 0.988, 0.988, facecolor="white", edgecolor=BORDER, linewidth=0.7))
    evidence_panel(axes[0])
    design_panel(axes[1])
    gate_panel(axes[2])
    figure.subplots_adjust(left=0.006, right=0.994, top=0.98, bottom=0.04, wspace=0.055)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.015)
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.015)
    plt.close(figure)


if __name__ == "__main__":
    main()
