#!/usr/bin/env python3
"""Render the reviewer-requested MOSAIC revision evidence figure."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[2]
ARTIFACTS = REPOSITORY / "research" / "artifacts"
EVIDENCE = ARTIFACTS / "mosaic_revision_evidence_v1.json"
MISSPEC = ARTIFACTS / "mosaic_bridge_misspecification_v1.json"
ADMITTED_SHIFT = ARTIFACTS / "mosaic_admitted_shift_stress_v1.json"
OUTPUT = ROOT / "figures" / "figure4_mosaic_revision_evidence"

INK = "#202124"
MUTED = "#5F6368"
GRID = "#DADCE0"
GREEN = "#009E73"
BLUE = "#0072B2"
ORANGE = "#E69F00"
RED = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


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


def direct_target_panel(ax: plt.Axes, stress: dict[str, Any]) -> None:
    primary = stress["primary"]
    names = ["Direct target\ntable", "Strict\nMOSAIC"]
    deployments = [
        int(primary["direct_deployments"]),
        int(primary["mosaic_deployments"]),
    ]
    violations = [
        int(primary["direct_contract_violations"]),
        int(primary["mosaic_contract_violations"]),
    ]
    colors = [BLUE, GREEN]
    positions = np.arange(len(names))
    bars = ax.bar(positions, deployments, color=colors, width=0.58, zorder=2)
    for bar, deployed, violation_count in zip(bars, deployments, violations):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            deployed + 1.5,
            f"{deployed} deployed",
            ha="center",
            va="bottom",
            fontsize=6.7,
            fontweight="bold",
        )
        if violation_count:
            ax.bar(
                bar.get_x() + bar.get_width() / 2,
                violation_count,
                color=RED,
                width=bar.get_width(),
                zorder=3,
            )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(1.0, violation_count / 2),
            f"{violation_count} violations",
            ha="center",
            va="center",
            fontsize=5.9,
            color="white" if violation_count else INK,
            fontweight="bold" if violation_count else "normal",
        )
    ax.set_xticks(positions, names)
    ax.set_ylim(0, 42)
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_ylabel("Real-table selected jobs")
    ax.set_title("The bridge blocks harmful admitted drift", loc="left")
    ax.text(
        0.0,
        -0.31,
        "Worst laws lie inside each learned bridge class but outside the direct table region.",
        transform=ax.transAxes,
        fontsize=5.65,
        color=MUTED,
        wrap=True,
    )


def stratum_panel(ax: plt.Axes, evidence: dict[str, Any], misspec: dict[str, Any]) -> None:
    cells = [
        cell
        for cell in misspec["cells"]
        if cell["scenario"] == "compatible_common_transform"
    ]
    cells.sort(key=lambda row: int(row["sample_size_per_stratum"]))
    sizes = np.asarray([int(row["sample_size_per_stratum"]) for row in cells])
    rates = np.asarray([float(row["acceptance_rate"]) for row in cells])
    ax.plot(sizes, rates, color=GREEN, marker="o", linewidth=1.5, markersize=4.2)
    for index, (dataset, entry) in enumerate(evidence["per_dataset"].items()):
        n = float(entry["bridge_strata"]["minimum"])
        ax.axvline(n if n > 0 else 1.0, color=INK, linewidth=0.65, alpha=0.42)
        label = dataset.replace("-WILDS", "").replace("-Clinical", "")
        ax.text(
            n if n > 0 else 1.0,
            0.055 + 0.16 * (index % 4),
            label,
            rotation=90,
            fontsize=5.25,
            color=INK,
            ha="right" if n <= 3 else "left",
            va="bottom",
        )
    ax.set_xscale("log")
    ax.set_xlim(0.8, 6500)
    ax.set_ylim(-0.04, 1.06)
    ax.set_yticks([0, 0.5, 1.0], ["0", "50", "100"])
    ax.set_xticks([1, 10, 100, 1000, 5000], ["0/1", "10", "100", "1k", "5k"])
    ax.set_xlabel("Bridge samples per source-label stratum (minimum)")
    ax.set_ylabel("Compatible bridge accepted (%)")
    ax.set_title("Real target support explains several abstentions", loc="left")


def utility_panel(ax: plt.Axes, evidence: dict[str, Any]) -> None:
    utility = evidence["diagnostic_anchored_interface_utility"]
    datasets = sorted(utility)
    labels = ["Released\ninterface", "4-bin\ntokenizer", "Full edited\nfeatures", "Unedited\nfeatures"]
    keys = [
        "released_interface_expected_balanced_accuracy_mean",
        "four_bin_tokenizer_before_channel_balanced_accuracy_mean",
        "full_feature_classifier_on_selected_edit_balanced_accuracy_mean",
        "full_feature_classifier_on_unedited_representation_balanced_accuracy_mean",
    ]
    colors = [GREEN, SKY, BLUE, MUTED]
    width = 0.35
    positions = np.arange(len(keys))
    for index, dataset in enumerate(datasets):
        values = [float(utility[dataset][key]) for key in keys]
        ax.bar(
            positions + (index - 0.5) * width,
            values,
            width=width,
            color=colors,
            alpha=0.86 if index == 0 else 0.54,
            edgecolor="white",
            label=f"{dataset} ({utility[dataset]['releases']})",
        )
    ax.set_xticks(positions, labels)
    ax.set_ylim(0.55, 0.96)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9], [".60", ".70", ".80", ".90"])
    ax.set_ylabel("Diagnostic balanced accuracy")
    ax.set_title("Released interfaces retain task signal", loc="left")
    ax.legend(frameon=False, fontsize=5.45, loc="lower left")
    ax.text(
        0.0,
        -0.32,
        "Post-outcome diagnostic audit; each released-token table matches its locked receipt.",
        transform=ax.transAxes,
        fontsize=5.35,
        color=MUTED,
        wrap=True,
    )


def frontier_panel(ax: plt.Axes, evidence: dict[str, Any]) -> None:
    frontier = evidence["real_release_frontier"]
    thresholds = [str(value) for value in frontier["thresholds"]]
    x = np.arange(len(thresholds))
    for name, color, marker, style in (
        ("strict_mosaic", GREEN, "o", "-"),
        ("bridge_plugin", ORANGE, "D", "--"),
    ):
        rows = frontier[name]
        deployment = [float(rows[key]["deployment_rate"]) for key in thresholds]
        violations = [float(rows[key]["false_acceptance_rate_among_estimable"] or 0.0) for key in thresholds]
        label = "Strict MOSAIC" if name == "strict_mosaic" else "Bridge plug-in"
        ax.plot(x, deployment, color=color, marker=marker, linestyle=style, linewidth=1.5, markersize=4.0, label=f"{label}: deployed")
        ax.plot(x, violations, color=color, marker=marker, linestyle=":", linewidth=0.9, markersize=3.2, alpha=0.8, label=f"{label}: diagnostic violations")
    ax.set_xticks(x, thresholds)
    ax.set_ylim(-0.04, 0.88)
    ax.set_yticks([0, 0.25, 0.5, 0.75], ["0", "25", "50", "75"])
    ax.set_xlabel("Worst-stratum error contract")
    ax.set_ylabel("Rate (%)")
    ax.set_title("The real operating frontier", loc="left")
    ax.legend(frameon=False, fontsize=5.15, loc="upper left", ncol=2, columnspacing=0.8)


def main() -> None:
    evidence = load(EVIDENCE)
    misspec = load(MISSPEC)
    stress = load(ADMITTED_SHIFT)
    if evidence["off_event_formal_certificates"]["false_acceptances"] != 0:
        raise ValueError("revision figure requires passing off-event accounting")
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
    figure, axes = plt.subplots(2, 2, figsize=(7.08, 4.8), facecolor="white")
    direct_target_panel(axes[0, 0], stress)
    stratum_panel(axes[0, 1], evidence, misspec)
    utility_panel(axes[1, 0], evidence)
    frontier_panel(axes[1, 1], evidence)
    for axis, label in zip(axes.flat, "abcd"):
        style(axis, label)
    figure.subplots_adjust(left=0.09, right=0.995, top=0.94, bottom=0.18, wspace=0.34, hspace=0.72)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


if __name__ == "__main__":
    main()
