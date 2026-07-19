#!/usr/bin/env python3
"""Render the final MOSAIC evidence figure from passing audited artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[2]
ARTIFACTS = REPOSITORY / "research" / "artifacts"
OUTPUT = ROOT / "figures" / "figure3_mosaic_final_evidence"

BASELINE_REPORT = ARTIFACTS / "mosaic_baseline_extension_v1_schema_repaired.json"
BASELINE_AUDIT = ARTIFACTS / "mosaic_baseline_extension_audit_v1.json"
BASELINE_SUMMARY = ARTIFACTS / "mosaic_baseline_extension_summary_v1.json"
MISSPEC_REPORT = ARTIFACTS / "mosaic_bridge_misspecification_v1.json"
MISSPEC_AUDIT = ARTIFACTS / "mosaic_bridge_misspecification_audit_v1.json"
REAL_SUMMARY = ARTIFACTS / "mosaic_bridge_evidence_summary_v2.json"
CORRECTION_AUDIT = ARTIFACTS / "mosaic_bridge_strict_correction_v2_audit_v1.json"

GREEN = "#009E73"
BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
INK = "#202124"
MUTED = "#667085"
GRID = "#D9DEE7"

RULE_STYLE = {
    "strict_mosaic": ("MOSAIC + bridge", GREEN, "o", "-"),
    "capacity_transfer": ("Capacity transfer", BLUE, "s", "-"),
    "bridge_plugin": ("Bridge plug-in", PURPLE, "D", "--"),
    "validation_plugin": ("Validation plug-in", ORANGE, "^", "--"),
    "always_deploy_validation": ("Always deploy", VERMILION, "X", ":"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def verify() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    baseline = load(BASELINE_SUMMARY)
    baseline_audit = load(BASELINE_AUDIT)
    if baseline_audit.get("pass") is not True:
        raise AssertionError("matched-baseline audit did not pass")
    if baseline.get("audit_status") != "development_replay_not_independent_human_review":
        raise AssertionError("matched-baseline audit scope changed unexpectedly")
    if baseline.get("audit_sha256") != sha256(BASELINE_AUDIT):
        raise AssertionError("matched-baseline summary does not match its audit")
    if baseline.get("report_sha256") != sha256(BASELINE_REPORT):
        raise AssertionError("matched-baseline summary does not match its report")

    misspec = load(MISSPEC_REPORT)
    misspec_audit = load(MISSPEC_AUDIT)
    if misspec_audit.get("pass") is not True:
        raise AssertionError("bridge-misspecification audit did not pass")
    if misspec_audit.get("report_sha256") != sha256(MISSPEC_REPORT):
        raise AssertionError("bridge-misspecification audit does not match its report")

    real = load(REAL_SUMMARY)
    if real.get("status") != "complete":
        raise AssertionError("real bridge evidence summary is incomplete")
    if real.get("strict_receipt_count") != 100:
        raise AssertionError("real summary does not contain 100 strict receipts")
    if real.get("comparator_receipt_count") != 100:
        raise AssertionError("real summary does not contain 100 comparator receipts")
    for receipt in real.get("audit_receipts", {}).values():
        path = REPOSITORY / str(receipt["path"])
        if not path.exists() or sha256(path) != receipt.get("sha256"):
            raise AssertionError(f"real evidence audit hash mismatch: {path}")
        if load(path).get("passed") is not True:
            raise AssertionError(f"real evidence audit did not pass: {path}")
    correction = load(CORRECTION_AUDIT)
    if correction.get("passed") is not True:
        raise AssertionError("strict-v2 correction-scope audit did not pass")
    if correction.get("files_compared") != 100:
        raise AssertionError("strict-v2 correction audit is incomplete")
    return baseline, misspec, real


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.055,
        label,
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
        bbox={"boxstyle": "square,pad=0.20", "facecolor": INK, "edgecolor": INK},
    )


def baseline_panel(ax: plt.Axes, baseline: dict[str, Any]) -> None:
    keys = [
        "mosaic_continuum",
        "holm_ltt_grid",
        "table_region_grid",
        "fare_style_deterministic",
    ]
    labels = ["MOSAIC continuum", "Holm LTT (500)", "Table-region grid", "Deterministic"]
    colors = [GREEN, BLUE, SKY, MUTED]
    cells = baseline["cells"]["retention_and_stochastic_value"]
    rows = [cells[key] for key in keys]
    values = np.asarray([float(row["deployment_rate"]) for row in rows])
    lower = np.asarray([float(row["deployment_exact_95_interval"][0]) for row in rows])
    upper = np.asarray([float(row["deployment_exact_95_interval"][1]) for row in rows])
    positions = np.arange(len(rows))
    bars = ax.barh(positions, values, color=colors, height=0.62, edgecolor="white")
    ax.errorbar(
        values,
        positions,
        xerr=np.vstack((values - lower, upper - values)),
        color=INK,
        fmt="none",
        linewidth=0.75,
        capsize=1.8,
    )
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            max(value + 0.025, 0.025),
            bar.get_y() + bar.get_height() / 2,
            f"{100 * value:.1f}%",
            va="center",
            ha="left",
            fontsize=6.3,
            color=INK,
        )
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 0.68)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6], ["0", "20", "40", "60"])
    ax.set_xlabel("Certified deployments (%)")
    ax.set_title("One table covers a continuum", loc="left")
    ax.text(
        0.98,
        0.06,
        "+29.6 pp vs. Holm LTT\npaired exact p < 10$^{-71}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.1,
        color=MUTED,
    )


def misspecification_panel(ax: plt.Axes, report: dict[str, Any]) -> None:
    scenario_style = {
        "compatible_common_transform": ("Supported common transform", GREEN, "o", "-"),
        "underdeclared_contamination": ("Too much contamination", VERMILION, "s", "--"),
        "source_specific_transform": ("Source-specific transform", PURPLE, "D", ":"),
    }
    sizes = [int(value) for value in report["sample_sizes_per_stratum"]]
    positions = np.arange(len(sizes))
    by_key = {
        (str(cell["scenario"]), int(cell["sample_size_per_stratum"])): cell
        for cell in report["cells"]
    }
    for scenario, (label, color, marker, linestyle) in scenario_style.items():
        rows = [by_key[(scenario, size)] for size in sizes]
        values = np.asarray([float(row["acceptance_rate"]) for row in rows])
        lower = np.asarray([float(row["acceptance_cp95_lower"]) for row in rows])
        upper = np.asarray([float(row["acceptance_cp95_upper"]) for row in rows])
        ax.errorbar(
            positions,
            values,
            yerr=np.vstack((values - lower, upper - values)),
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.35,
            markersize=4.1,
            elinewidth=0.75,
            capsize=1.8,
            label=label,
        )
    ax.set_xticks(positions, [str(size) for size in sizes])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100"])
    ax.set_ylim(-0.04, 1.05)
    ax.set_xlabel("Bridge samples per source-label stratum")
    ax.set_ylabel("Bridge accepted (%)")
    ax.set_title("Invalid bridge models are rejected", loc="left")
    ax.legend(frameon=False, fontsize=5.8, loc="center right", handlelength=2.0)


def real_curve_panel(
    ax: plt.Axes,
    real: dict[str, Any],
    *,
    value_key: str,
    title: str,
    ylabel: str,
    legend: bool,
) -> None:
    thresholds = [str(value) for value in real["utility_thresholds"]]
    positions = np.arange(len(thresholds))
    for rule in real["rules"]:
        label, color, marker, linestyle = RULE_STYLE[rule]
        rows = [real["cells"][rule]["all_datasets"][value] for value in thresholds]
        values = np.asarray(
            [np.nan if row[value_key] is None else float(row[value_key]) for row in rows]
        )
        ax.plot(
            positions,
            values,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.25,
            markersize=3.8,
            label=label,
        )
    ax.set_xticks(positions, thresholds)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100"])
    ax.set_ylim(-0.04, 1.05)
    ax.set_xlabel("Worst-stratum utility-error contract")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    if legend:
        ax.legend(
            frameon=False,
            fontsize=5.45,
            loc="upper left",
            ncol=2,
            handlelength=1.8,
            columnspacing=0.8,
        )


def style_axis(ax: plt.Axes, label: str) -> None:
    panel_label(ax, label)
    ax.grid(color=GRID, linewidth=0.5, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_axisbelow(True)


def main() -> None:
    baseline, misspec, real = verify()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.4,
            "axes.titlesize": 8.0,
            "axes.titleweight": "bold",
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.6,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.08, 4.45), facecolor="white")
    baseline_panel(axes[0, 0], baseline)
    misspecification_panel(axes[0, 1], misspec)
    real_curve_panel(
        axes[1, 0],
        real,
        value_key="deployment_rate",
        title="Real release decisions across contracts",
        ylabel="Dataset-seed jobs deployed (%)",
        legend=True,
    )
    real_curve_panel(
        axes[1, 1],
        real,
        value_key="false_acceptance_rate_among_estimable",
        title="Unsafe real releases are exposed",
        ylabel="Violations among estimable deployments (%)",
        legend=False,
    )
    for ax, label in zip(axes.flat, "abcd", strict=True):
        style_axis(ax, label)
    fig.subplots_adjust(left=0.095, right=0.995, top=0.94, bottom=0.12, wspace=0.34, hspace=0.48)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
