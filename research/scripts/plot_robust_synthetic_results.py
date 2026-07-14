"""Render the preregistered synthetic theory-match study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "artifacts" / "vera_robust_synthetic_report.json"
DEFAULT_STEM = ROOT / "maintrack" / "figures" / "vera_synthetic_theory_match"
COLORS = {0.01: "#0072B2", 0.05: "#D55E00", 0.10: "#009E73"}
MARKERS = {0.01: "o", 0.05: "s", 0.10: "^"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_STEM)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("claim_grade") is not True or report.get("all_cells_pass") is not True:
        raise RuntimeError("refusing to plot a nonpassing or non-claim-grade report")

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    cells = report["cells"]
    deltas = sorted({float(cell["delta"]) for cell in cells})
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35), constrained_layout=True)

    for delta in deltas:
        subset = sorted(
            (cell for cell in cells if float(cell["delta"]) == delta),
            key=lambda cell: int(cell["n"]),
        )
        x = np.asarray([cell["n"] for cell in subset], dtype=float)
        predicted = np.asarray([cell["predicted_abstention"] for cell in subset])
        observed = np.asarray([cell["empirical_abstention"] for cell in subset])
        lower = np.asarray([cell["predicted_abstention_band_lower"] for cell in subset])
        upper = np.asarray([cell["predicted_abstention_band_upper"] for cell in subset])
        color = COLORS[delta]
        axes[0].fill_between(x, lower, upper, color=color, alpha=0.13, linewidth=0)
        axes[0].plot(x, predicted, color=color, linestyle="--")
        axes[0].scatter(
            x,
            observed,
            color=color,
            marker=MARKERS[delta],
            s=19,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
            label=rf"$\delta={delta:.2f}$",
        )
        axes[1].plot(
            x,
            observed - predicted,
            color=color,
            marker=MARKERS[delta],
            markersize=3.5,
            label=rf"$\delta={delta:.2f}$",
        )

    axes[0].set_xscale("log")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_xlabel("Validation size, n")
    axes[0].set_ylabel("Abstention rate")
    axes[0].set_title("Abstention theory match")
    axes[0].legend(frameon=False, loc="center left")

    axes[1].axhline(0.0, color="#666666", linewidth=0.8, linestyle=":")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Validation size, n")
    axes[1].set_ylabel("Observed minus predicted")
    axes[1].set_title("Prediction error")

    sizes = sorted({int(cell["n"]) for cell in cells})
    offsets = np.linspace(-0.0025, 0.0025, len(sizes))
    for offset, n in zip(offsets, sizes):
        subset = sorted(
            (cell for cell in cells if int(cell["n"]) == n),
            key=lambda cell: float(cell["delta"]),
        )
        axes[2].scatter(
            [float(cell["delta"]) + offset for cell in subset],
            [cell["false_acceptance_cp_upper"] for cell in subset],
            s=15,
            color="#0072B2",
            alpha=0.45 + 0.5 * sizes.index(n) / (len(sizes) - 1),
            edgecolor="none",
        )
    xline = np.linspace(0.005, 0.11, 100)
    axes[2].plot(xline, xline, color="#D55E00", linestyle="--", label=r"Upper bound = $\delta$")
    axes[2].set_xlim(0.0, 0.11)
    axes[2].set_ylim(0.0, 0.105)
    axes[2].set_xlabel(r"Declared $\delta$")
    axes[2].set_ylabel("Simultaneous 95% upper bound")
    axes[2].set_title("False-acceptance control")
    axes[2].legend(frameon=False, loc="upper left")

    for label, axis in zip("ABC", axes):
        axis.text(
            -0.16,
            1.08,
            label,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.7)

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_metadata = {
        "Title": "VERA synthetic shift-certificate theory match",
        "Subject": f"Preregistered receipt {report['prereg_sha256']}",
        "Keywords": "VERA, concept erasure, distribution shift, certification",
    }
    svg_metadata = {
        "Title": "VERA synthetic shift-certificate theory match",
        "Description": f"Preregistered receipt {report['prereg_sha256']}",
        "Keywords": "VERA, concept erasure, distribution shift, certification",
    }
    pdf_path = args.output_stem.with_suffix(".pdf")
    svg_path = args.output_stem.with_suffix(".svg")
    fig.savefig(pdf_path, metadata=pdf_metadata)
    fig.savefig(svg_path, metadata=svg_metadata)
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines())
        + "\n",
        encoding="utf-8",
    )
    fig.savefig(
        args.output_stem.with_suffix(".png"),
        dpi=600,
        metadata={"Description": svg_metadata["Description"]},
    )
    plt.close(fig)
    print(args.output_stem)


if __name__ == "__main__":
    main()
