"""Build deterministic VERA main-track figures from checked artifacts.

The figures intentionally avoid generated or hand-entered numbers. Each plotted
metric comes from a receipt, summary CSV, or abstention certificate already
written by the experiment pipeline.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
MAINTRACK = ROOT / "maintrack"
FIGURES = MAINTRACK / "figures"
REPORT = ARTIFACTS / "maintrack_figure_report.json"

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
GREY = "#6E6E6E"
LIGHT_GREY = "#F4F4F4"


@dataclass(frozen=True)
class FigureRecord:
    key: str
    path: Path
    inputs: list[str]
    caption_intent: str


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def f(row: dict[str, str], key: str) -> float:
    candidates = [key]
    if key.endswith("_mean"):
        candidates.append(key.removesuffix("_mean"))
    value = ""
    for candidate in candidates:
        value = row.get(candidate, "")
        if value not in {"", "None", None}:
            break
    if value in {"", "None", None}:
        return float("nan")
    return float(value)


def ensure_output() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)


def savefig(fig: plt.Figure, path: Path) -> None:
    for output_path in (path, path.with_suffix(".pdf")):
        fig.savefig(
            output_path,
            bbox_inches="tight",
            dpi=300,
            metadata={"Creator": "VERA artifact figure builder"},
        )
    plt.close(fig)


def draw_box(ax, xy, width, height, text, facecolor, edgecolor=GREY, fontsize=9):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.025",
        linewidth=1.2,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.2,
    )
    return box


def draw_arrow(ax, start, end, color=GREY, style="-|>"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=14,
        linewidth=1.2,
        color=color,
        shrinkA=6,
        shrinkB=6,
    )
    ax.add_patch(arrow)


def build_method_overview() -> FigureRecord:
    path = FIGURES / "faro_method_overview.png"
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    draw_box(ax, (0.03, 0.58), 0.18, 0.24, "Frozen\nrepresentation $Z$\nlabels $y,s,g$", "#EAF4FB", BLUE)
    draw_box(ax, (0.29, 0.58), 0.20, 0.24, "Candidate edits\nINLP / LEACE\nR-LACE / TaCo\nrank-strength grid", "#FFF6DF", ORANGE)
    draw_box(ax, (0.57, 0.58), 0.18, 0.24, "Audit frontier\n$R_y(M)$\n$Q_s(M)$\nworst group", "#E9F7F1", GREEN)
    draw_box(ax, (0.57, 0.18), 0.18, 0.22, "Certified\nsafe set\nsimultaneous CIs", "#F2ECF7", PURPLE)
    draw_box(ax, (0.82, 0.64), 0.15, 0.16, "EDIT\nsmallest safe\n$M^*$", "#E9F7F1", GREEN)
    draw_box(ax, (0.82, 0.22), 0.15, 0.16, "ABSTAIN\nempty-frontier\ncertificate", "#FCEDE8", RED)

    draw_arrow(ax, (0.21, 0.70), (0.29, 0.70))
    draw_arrow(ax, (0.49, 0.70), (0.57, 0.70))
    draw_arrow(ax, (0.66, 0.58), (0.66, 0.40))
    draw_arrow(ax, (0.75, 0.31), (0.82, 0.30), RED)
    draw_arrow(ax, (0.75, 0.70), (0.82, 0.72), GREEN)

    ax.text(0.66, 0.50, "target budget +\nsource reduction", ha="center", va="center", fontsize=8, color=GREY)
    ax.text(0.03, 0.08, "VERA is a certification layer over candidate erasers, not another unconditional eraser.", fontsize=9)
    savefig(fig, path)
    return FigureRecord(
        key="method_overview",
        path=path,
        inputs=[],
        caption_intent="Schematic of VERA as a frontier-estimation and edit-or-abstain certification layer.",
    )


def build_synthetic_abstention() -> FigureRecord:
    frontier_path = ARTIFACTS / "faro_synthetic_abstention_frontier.csv"
    report_path = ARTIFACTS / "faro_synthetic_abstention_report.json"
    rows = read_csv(frontier_path)
    report = read_json(report_path)
    path = FIGURES / "faro_synthetic_abstention_geometry.png"

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    cases = [
        ("overlap_abstain", "Overlap: abstain", report["overlap_case"]),
        ("nonoverlap_edit", "Non-overlap: edit", report["nonoverlap_case"]),
    ]
    for ax, (case_key, title, cert) in zip(axes, cases):
        case_rows = [row for row in rows if row["case"] == case_key]
        xs = np.array([f(row, "strength") for row in case_rows])
        target_risk = np.array([f(row, "target_risk") for row in case_rows])
        source_reduction = np.array([f(row, "source_reduction") for row in case_rows])
        safe = np.array([row["certified_safe"] == "True" for row in case_rows])
        ax.plot(xs, target_risk, color=RED, marker="o", linewidth=1.5, label="target risk")
        ax.plot(xs, source_reduction, color=BLUE, marker="s", linewidth=1.5, label="source reduction")
        if safe.any():
            ax.scatter(xs[safe], source_reduction[safe], color=GREEN, s=90, marker="*", zorder=5, label="certified safe")
        ax.axvline(cert["lambda_y_star"], color=RED, linestyle="--", linewidth=1.0, label="$\\lambda_y^*$")
        ax.axvline(cert["lambda_s_star"], color=BLUE, linestyle="--", linewidth=1.0, label="$\\lambda_s^*$")
        ax.axhline(cert["delta"], color=BLUE, linestyle=":", linewidth=1.0)
        ax.axhline(cert["epsilon"], color=RED, linestyle=":", linewidth=1.0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("edit strength $\\lambda$")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.22)
        ax.text(
            0.03,
            0.92,
            cert["decision"],
            transform=ax.transAxes,
            fontsize=9,
            weight="bold",
            color=GREEN if cert["decision"] == "EDIT" else RED,
        )
    axes[0].set_ylabel("normalized criterion")
    handles, labels = axes[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.subplots_adjust(bottom=0.25, wspace=0.18)
    savefig(fig, path)
    return FigureRecord(
        key="synthetic_abstention_geometry",
        path=path,
        inputs=[str(frontier_path), str(report_path)],
        caption_intent="Synthetic theorem-aligned geometry showing when target-safe and source-sufficient intervals fail or succeed to intersect.",
    )


def build_real_frontier() -> FigureRecord:
    frontier_path = ARTIFACTS / "faro_real_abstention_stress_frontier.csv"
    report_path = ARTIFACTS / "faro_real_abstention_stress_report.json"
    rows = read_csv(frontier_path)
    report = read_json(report_path)
    path = FIGURES / "faro_civilcomments_frontier.png"
    strict = report["cases"][0]
    relaxed = report["cases"][1]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), sharex=True, sharey=True)
    for ax, case in zip(axes, [strict, relaxed]):
        case_rows = [row for row in rows if row["case"] == case["case"]]
        colors = []
        markers = []
        for row in case_rows:
            if row["certified_safe"] == "True":
                colors.append(GREEN)
                markers.append("*")
            elif row["source_sufficient"] == "True":
                colors.append(ORANGE)
                markers.append("o")
            else:
                colors.append(GREY)
                markers.append("o")
        for row, color, marker in zip(case_rows, colors, markers):
            x = f(row, "source_reduction_lcb95")
            y = f(row, "target_loss_ucb95")
            ax.scatter(x, y, color=color, marker=marker, s=90 if marker == "*" else 38, alpha=0.9)
        ax.axvline(case["delta_source_reduction"], color=BLUE, linestyle="--", linewidth=1.0)
        ax.axhline(case["epsilon_target_loss"], color=RED, linestyle="--", linewidth=1.0)
        ax.fill_betweenx(
            [0, case["epsilon_target_loss"]],
            case["delta_source_reduction"],
            0.27,
            color=GREEN,
            alpha=0.08,
            label="certified-safe region",
        )
        ax.set_title(
            f"{case['decision'].replace('_', ' ')}: {case['safe_candidate_count']} safe",
            fontsize=10,
            color=GREEN if case["decision"] == "EDIT" else RED,
        )
        ax.grid(alpha=0.2)
        ax.set_xlabel("source reduction LCB95")
    axes[0].set_ylabel("target loss UCB95")
    axes[0].set_xlim(-0.01, 0.27)
    axes[0].set_ylim(-0.002, 0.255)
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GREY, markersize=6, label="not source-sufficient"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ORANGE, markersize=6, label="source-sufficient only"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=GREEN, markeredgecolor=GREEN, markersize=10, label="certified safe"),
        Line2D([0], [0], color=BLUE, linestyle="--", label="$\\delta$ threshold"),
        Line2D([0], [0], color=RED, linestyle="--", label="$\\epsilon$ threshold"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, fontsize=7.5)
    fig.subplots_adjust(bottom=0.23)
    fig.subplots_adjust(wspace=0.12)
    savefig(fig, path)
    return FigureRecord(
        key="civilcomments_frontier",
        path=path,
        inputs=[str(frontier_path), str(report_path)],
        caption_intent="Real CivilComments stress frontier showing strict abstention and relaxed editing under declared thresholds.",
    )


def build_camelyon_projection_frontier() -> FigureRecord:
    frontier_path = ARTIFACTS / "camelyon17_faro_projection_frontier.csv"
    certificate_path = ARTIFACTS / "camelyon17_faro_projection_certificate.json"
    rows = read_csv(frontier_path)
    certificate = read_json(certificate_path)
    path = FIGURES / "faro_camelyon_projection_frontier.png"

    xs = np.array([f(row, "source_reduction_lcb95") for row in rows])
    ys = np.array([f(row, "target_loss_ucb95") for row in rows])
    strengths = [f(row, "strength") for row in rows]
    safe = np.array([row["certified_safe"] == "True" for row in rows])
    source_sufficient = np.array([row["source_sufficient"] == "True" for row in rows])

    fig, ax = plt.subplots(figsize=(5.8, 3.35))
    for x, y, strength, is_safe, is_source_ok in zip(xs, ys, strengths, safe, source_sufficient):
        if is_safe:
            color, marker, size = GREEN, "*", 110
        elif is_source_ok:
            color, marker, size = ORANGE, "o", 52
        else:
            color, marker, size = GREY, "o", 44
        ax.scatter(x, y, color=color, marker=marker, s=size, alpha=0.9, zorder=4)
        if strength in (0.0, 1.0):
            ax.annotate(
                f"{strength:.1f}",
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color="#374151",
            )

    delta = float(certificate["delta_source_reduction"])
    epsilon = float(certificate["epsilon_target_loss"])
    ax.axvline(delta, color=BLUE, linestyle="--", linewidth=1.1)
    ax.axhline(epsilon, color=RED, linestyle="--", linewidth=1.1)
    ax.fill_betweenx([0, epsilon], delta, max(delta * 1.3, 0.065), color=GREEN, alpha=0.08)
    ax.set_xlim(-0.004, max(delta * 1.3, 0.065))
    ax.set_ylim(0.0, max(epsilon * 1.25, float(np.nanmax(ys)) * 1.15))
    ax.set_xlabel("source reduction LCB95", fontsize=10)
    ax.set_ylabel("target loss UCB95", fontsize=10)
    ax.set_title(
        f"Camelyon17 {certificate['decision']}: {certificate['safe_candidate_count']} certified edits",
        fontsize=11,
        color=RED if certificate["decision"] == "ABSTAIN" else GREEN,
    )
    ax.tick_params(labelsize=9)
    ax.grid(alpha=0.22)
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GREY, markersize=6, label="not source-sufficient"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=GREEN, markeredgecolor=GREEN, markersize=10, label="certified safe"),
        Line2D([0], [0], color=BLUE, linestyle="--", label="$\\delta$ source target"),
        Line2D([0], [0], color=RED, linestyle="--", label="$\\epsilon$ target budget"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7, frameon=False)
    savefig(fig, path)
    return FigureRecord(
        key="camelyon_projection_frontier",
        path=path,
        inputs=[str(frontier_path), str(certificate_path)],
        caption_intent="Full Camelyon17 validation-only projection frontier: candidates preserve target utility but do not certify source-leakage reduction, so VERA abstains.",
    )


def choose_row(rows: Iterable[dict[str, str]], method_key: str) -> dict[str, str]:
    for row in rows:
        if row.get("method_key") == method_key or row.get("method") == method_key:
            return row
    raise KeyError(method_key)


def build_official_summary() -> FigureRecord:
    waterbirds_path = ARTIFACTS / "waterbirds_official_baseline_summary.csv"
    camelyon_path = ARTIFACTS / "camelyon17_wilds_official_multiseed_results.csv"
    claim_audit_path = ARTIFACTS / "benchmark_claim_audit.json"
    water = read_csv(waterbirds_path)
    camelyon = read_csv(camelyon_path)
    claim = read_json(claim_audit_path)
    path = FIGURES / "faro_official_benchmark_summary.png"

    entries = [
        ("Waterbirds\nVERA\nABSTAIN", choose_row(water, "VERA_selected")),
        ("Waterbirds\nGroup rw.\nbaseline", choose_row(water, "group_reweighted_erm")),
        ("Camelyon17\nVERA\nABSTAIN", choose_row(camelyon, "VERA_selected")),
        ("Camelyon17\nGroupDRO-style", choose_row(camelyon, "group_dro_probe")),
    ]
    metrics = [
        ("external_target_balanced_accuracy_mean", "Ext. BA", BLUE),
        ("external_worst_target_source_accuracy_mean", "Worst group", ORANGE),
        ("validation_source_leakage_balanced_accuracy_mean", "Source leak", PURPLE),
    ]

    fig, ax = plt.subplots(figsize=(6.8, 3.45))
    x = np.arange(len(entries))
    width = 0.23
    offsets = [-width, 0, width]
    for (metric_key, label, color), offset in zip(metrics, offsets):
        values = [f(row, metric_key) for _, row in entries]
        ax.bar(x + offset, values, width=width, label=label, color=color, alpha=0.78)
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in entries], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score")
    ax.grid(axis="y", alpha=0.22)
    ax.axvline(1.5, color="#BDBDBD", linewidth=1.0)
    ax.text(
        0.50,
        0.96,
        "Durable official rows; CivilComments remains prior non-durable stress evidence",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7,
        color="#4A5568",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=3, frameon=False, fontsize=8)
    fig.subplots_adjust(bottom=0.28, top=0.82)
    savefig(fig, path)
    return FigureRecord(
        key="official_benchmark_summary",
        path=path,
        inputs=[str(waterbirds_path), str(camelyon_path), str(claim_audit_path)],
        caption_intent="Evidence-boundary summary: Waterbirds and Camelyon17 are durable claim-ready; CivilComments is retained only as prior non-durable stress evidence in text.",
    )


def main() -> None:
    ensure_output()
    records = [
        build_method_overview(),
        build_synthetic_abstention(),
        build_real_frontier(),
        build_camelyon_projection_frontier(),
        build_official_summary(),
    ]
    payload = {
        "name": "VERA main-track figure report",
        "figure_count": len(records),
        "figures": [
            {
                "key": record.key,
                "path": str(record.path),
                "inputs": record.inputs,
                "caption_intent": record.caption_intent,
                "exists": record.path.exists(),
                "bytes": record.path.stat().st_size if record.path.exists() else 0,
            }
            for record in records
        ],
    }
    REPORT.write_text(json.dumps(payload, indent=2) + "\n")
    print("VERA main-track figures built")
    print(f"figure_count={len(records)}")
    print(f"report={REPORT}")
    for record in records:
        print(f"{record.key}: {record.path}")


if __name__ == "__main__":
    main()
