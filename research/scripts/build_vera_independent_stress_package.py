"""Build an audited figure and TeX block for the independent stress replication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "artifacts" / "vera_independent_stress_report.json"
DEFAULT_ANALYSIS_AUDIT = ROOT / "artifacts" / "vera_independent_stress_analysis_audit.json"
DEFAULT_ROWS = ROOT / "artifacts" / "vera_independent_stress_rule_rows.csv"
DEFAULT_TEX = (
    ROOT
    / "maintrack"
    / "aaai2027_template"
    / "AuthorKit27"
    / "vera_independent_stress_results.tex"
)
DEFAULT_MACROS = (
    ROOT
    / "maintrack"
    / "aaai2027_template"
    / "AuthorKit27"
    / "vera_results_macros.tex"
)
DEFAULT_PDF = ROOT / "maintrack" / "figures" / "vera_independent_stress_replication.pdf"
DEFAULT_PNG = ROOT / "maintrack" / "figures" / "vera_independent_stress_replication.png"
DEFAULT_AUDIT = ROOT / "artifacts" / "vera_independent_stress_package_audit.json"
DATASET_ORDER = ("Bios", "CivilComments-WILDS", "GaitPDB", "Waterbirds")
DISPLAY_NAMES = {
    "Bios": "Bios",
    "CivilComments-WILDS": "CivilComments",
    "GaitPDB": "GaitPDB",
    "Waterbirds": "Waterbirds",
}
POINT_COLOR = "#D55E00"
VERA_COLOR = "#0072B2"
RETENTION_COLOR = "#009E73"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def cp_interval(successes: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 1.0
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    )
    return lower, upper


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "axes.titleweight": "bold",
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.16,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )


def retention_by_dataset(
    rows: list[dict[str, str]],
) -> dict[str, tuple[int, int]]:
    oracle = {
        (row["dataset"], int(row["seed"])): row
        for row in rows
        if row["dataset"] in DATASET_ORDER
        and row["rule"] == "external_balanced_oracle"
    }
    vera = {
        (row["dataset"], int(row["seed"])): row
        for row in rows
        if row["dataset"] in DATASET_ORDER and row["rule"] == "vera_balanced_iut"
    }
    output: dict[str, tuple[int, int]] = {}
    for dataset in DATASET_ORDER:
        keys = [key for key in oracle if key[0] == dataset]
        opportunities = sum(as_bool(oracle[key]["deployed"]) for key in keys)
        retained = sum(
            as_bool(oracle[key]["deployed"])
            and as_bool(vera[key]["deployed"])
            and vera[key]["external_contract_satisfied"] == "True"
            for key in keys
        )
        output[dataset] = retained, opportunities
    return output


def make_figure(
    report: dict[str, Any],
    rows: list[dict[str, str]],
    pdf_path: Path,
    png_path: Path,
) -> None:
    configure_style()
    figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.45))
    x = np.arange(len(DATASET_ORDER), dtype=float)
    width = 0.35

    point_rates: list[float] = []
    vera_rates: list[float] = []
    point_errors: list[tuple[float, float]] = []
    vera_errors: list[tuple[float, float]] = []
    for dataset in DATASET_ORDER:
        point = report["by_dataset"][dataset]["point_selection_balanced"]
        vera = report["by_dataset"][dataset]["vera_balanced_iut"]
        point_rate = float(point["measured_external_violation_rate"])
        vera_rate = float(vera["measured_external_violation_rate"])
        point_interval = tuple(map(float, point["measured_external_violation_cp95"]))
        vera_interval = tuple(map(float, vera["measured_external_violation_cp95"]))
        point_rates.append(point_rate)
        vera_rates.append(vera_rate)
        point_errors.append((point_rate - point_interval[0], point_interval[1] - point_rate))
        vera_errors.append((vera_rate - vera_interval[0], vera_interval[1] - vera_rate))
    axes[0].bar(
        x - width / 2,
        point_rates,
        width,
        color=POINT_COLOR,
        edgecolor="black",
        linewidth=0.5,
        hatch="//",
        label="Point selection",
    )
    axes[0].bar(
        x + width / 2,
        vera_rates,
        width,
        color=VERA_COLOR,
        edgecolor="black",
        linewidth=0.5,
        label="VERA-IUT",
    )
    axes[0].errorbar(
        x - width / 2,
        point_rates,
        yerr=np.asarray(point_errors).T,
        fmt="none",
        ecolor="black",
        elinewidth=0.7,
        capsize=2,
    )
    axes[0].errorbar(
        x + width / 2,
        vera_rates,
        yerr=np.asarray(vera_errors).T,
        fmt="none",
        ecolor="black",
        elinewidth=0.7,
        capsize=2,
    )
    axes[0].axhline(0.20, color="#666666", linestyle="--", linewidth=0.8, label="20% endpoint")
    axes[0].axhline(0.05, color="#000000", linestyle=":", linewidth=0.8, label=r"$\delta=0.05$")
    axes[0].set_ylabel("External violation rate")
    axes[0].set_title("Locked external outcomes")
    axes[0].set_xticks(x, [DISPLAY_NAMES[name] for name in DATASET_ORDER], rotation=28, ha="right")
    axes[0].set_ylim(0.0, max(0.45, max(high for _, high in point_errors) + max(point_rates) + 0.04))
    axes[0].legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
        loc="upper right",
        handlelength=1.6,
    )

    point_only = [
        int(report["one_sided_mcnemar_discordance"][dataset]["point_only_violation"])
        for dataset in DATASET_ORDER
    ]
    vera_only = [
        int(report["one_sided_mcnemar_discordance"][dataset]["vera_only_violation"])
        for dataset in DATASET_ORDER
    ]
    axes[1].bar(
        x - width / 2,
        point_only,
        width,
        color=POINT_COLOR,
        edgecolor="black",
        linewidth=0.5,
        hatch="//",
        label="Point only",
    )
    axes[1].bar(
        x + width / 2,
        vera_only,
        width,
        color=VERA_COLOR,
        edgecolor="black",
        linewidth=0.5,
        label="VERA only",
    )
    axes[1].axhline(7, color="#000000", linestyle=":", linewidth=0.8, label="Power target")
    for index, dataset in enumerate(DATASET_ORDER):
        adjusted = float(report["one_sided_mcnemar_holm_p"][dataset])
        axes[1].text(
            index,
            max(point_only[index], vera_only[index]) + 0.35,
            f"p={adjusted:.3g}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    axes[1].set_ylabel("Discordant seed pairs")
    axes[1].set_title("One-sided paired test")
    axes[1].set_xticks(x, [DISPLAY_NAMES[name] for name in DATASET_ORDER], rotation=28, ha="right")
    axes[1].set_ylim(0, max(8.5, max(point_only + vera_only) + 2.0))
    axes[1].legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
        loc="upper right",
        handlelength=1.6,
    )

    retention = retention_by_dataset(rows)
    retention_labels = [DISPLAY_NAMES[name] for name in DATASET_ORDER] + ["Overall"]
    retention_counts = [retention[name] for name in DATASET_ORDER]
    retention_counts.append(
        (
            sum(value[0] for value in retention.values()),
            sum(value[1] for value in retention.values()),
        )
    )
    retention_rates = [
        np.nan if opportunities == 0 else retained / opportunities
        for retained, opportunities in retention_counts
    ]
    retention_intervals = [
        (np.nan, np.nan)
        if opportunities == 0
        else cp_interval(retained, opportunities)
        for retained, opportunities in retention_counts
    ]
    retention_errors = [
        (0.0, 0.0)
        if np.isnan(rate)
        else (rate - interval[0], interval[1] - rate)
        for rate, interval in zip(retention_rates, retention_intervals)
    ]
    retention_plot_rates = [
        0.0 if np.isnan(rate) else rate for rate in retention_rates
    ]
    rx = np.arange(len(retention_labels), dtype=float)
    axes[2].bar(
        rx,
        retention_plot_rates,
        0.64,
        color=RETENTION_COLOR,
        edgecolor="black",
        linewidth=0.5,
        hatch="..",
    )
    axes[2].errorbar(
        rx,
        retention_plot_rates,
        yerr=np.asarray(retention_errors).T,
        fmt="none",
        ecolor="black",
        elinewidth=0.7,
        capsize=2,
    )
    for index, (retained, opportunities) in enumerate(retention_counts):
        if opportunities == 0:
            label = "NA\n0 opp."
            label_height = 0.04
        else:
            label = f"{retained}/{opportunities}"
            label_height = min(
                1.03,
                retention_plot_rates[index]
                + retention_errors[index][1]
                + 0.04,
            )
        axes[2].text(
            index,
            label_height,
            label,
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    axes[2].set_ylabel("Safe retention")
    axes[2].set_title("Certification tax")
    axes[2].set_xticks(rx, retention_labels, rotation=28, ha="right")
    axes[2].set_ylim(0.0, 1.08)

    for label, axis in zip("ABC", axes):
        add_panel_label(axis, label)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.65)
        axis.set_axisbelow(True)
    figure.tight_layout(w_pad=1.0)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    figure.savefig(png_path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def tex_escape_dataset(name: str) -> str:
    return DISPLAY_NAMES[name].replace("_", r"\_")


def make_tex(report: dict[str, Any]) -> str:
    seed_count = len(report["replication_seeds"])
    supported_count = seed_count * len(DATASET_ORDER)
    point = report["supported_summaries"]["point_selection_balanced"]
    vera = report["supported_summaries"]["vera_balanced_iut"]
    tax = report["certification_tax"]
    point_count = int(point["measured_external_violation_count"])
    vera_count = int(vera["measured_external_violation_count"])
    retention = float(tax["safe_retention"])
    passed_count = int(report["supported_datasets_passing_all_three"])
    if report["passed"]:
        lead = (
            "All four supported datasets passed every preregistered endpoint on "
            f"{seed_count} disjoint replication seeds each. Point selection deployed "
            f"contract-violating edits in {point_count}/{supported_count} decisions "
            f"({100 * point_count / supported_count:.1f}\\%), versus "
            f"{vera_count}/{supported_count} ({100 * vera_count / supported_count:.1f}\\%) "
            "for VERA-IUT."
        )
    else:
        lead = (
            "The independent replication did not pass every preregistered endpoint: "
            f"{passed_count}/4 supported datasets cleared the joint naive-failure, "
            "VERA-control, and Holm-corrected paired-test requirements. The locked "
            "outcome is reported without replacing thresholds, seeds, or tests."
        )
    lines = [
        r"\paragraph{Independent disjoint-seed replication.}",
        lead,
        (
            f"VERA retained {int(tax['vera_safe_deployment_count'])}/"
            f"{int(tax['external_oracle_opportunity_count'])} external-oracle "
            f"opportunities ({100 * retention:.1f}\\%). Camelyon17 forced abstention "
            f"in all {int(report['camelyon_forced_abstention_count'])} registered "
            "VERA decisions because the deployment hospital remained outside "
            "certification support."
        ),
        "Dataset-level stress counts, Holm-adjusted paired tests, and the "
        "three-panel replication figure are generated as audited artifacts and "
        "included in the supplementary archive.",
    ]
    return "\n".join(lines)


def make_macros(report: dict[str, Any]) -> str:
    supported_count = len(report["supported_datasets"]) * len(report["replication_seeds"])
    point = report["supported_summaries"]["point_selection_balanced"]
    vera = report["supported_summaries"]["vera_balanced_iut"]
    tax = report["certification_tax"]
    point_rate = float(point["measured_external_violation_rate"])
    vera_rate = float(vera["measured_external_violation_rate"])
    retention = float(tax["safe_retention"])
    passed_count = int(report["supported_datasets_passing_all_three"])
    if report["passed"]:
        headline = (
            f"Across {supported_count} preregistered, disjoint-seed deployment "
            f"decisions, validation-only selection deployed contract-violating "
            f"edits in {100 * point_rate:.1f}\\% versus {100 * vera_rate:.1f}\\% "
            f"for VERA, while VERA retained {100 * retention:.1f}\\% of "
            "external-oracle opportunities."
        )
        seed_result = (
            "all four supported datasets reached Holm-adjusted $p\\leq0.05$ "
            "under the locked one-sided paired McNemar tests"
        )
        stress = (
            "The independent stress replication passed every preregistered "
            "supported-dataset endpoint and Camelyon17 remained a forced "
            "support-boundary abstention case."
        )
    else:
        headline = (
            "The independent stress replication did not satisfy every "
            f"preregistered empirical endpoint: {passed_count}/4 supported "
            "datasets cleared the joint naive-failure, VERA-control, and "
            "Holm-corrected paired-test requirements."
        )
        seed_result = (
            f"{passed_count}/4 supported datasets cleared all locked "
            "independent stress endpoints"
        )
        stress = (
            "VERA nevertheless forced abstention in all "
            f"{int(report['camelyon_forced_abstention_count'])} registered "
            "Camelyon17 decisions because the deployment hospital was outside "
            "certification support."
        )
    return "\n".join(
        [
            f"\\providecommand{{\\HeadlineResult}}{{{headline}}}",
            f"\\providecommand{{\\PointViolationRate}}{{{100 * point_rate:.1f}\\%}}",
            f"\\providecommand{{\\VERAViolationRate}}{{{100 * vera_rate:.1f}\\%}}",
            f"\\providecommand{{\\SafeRetentionRate}}{{{100 * retention:.1f}\\%}}",
            "\\providecommand{\\OfficialReceiptCount}{1000}",
            f"\\providecommand{{\\SeedBlockedResult}}{{{seed_result}}}",
            f"\\providecommand{{\\StressEffectResult}}{{{stress}}}",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--analysis-audit", type=Path, default=DEFAULT_ANALYSIS_AUDIT)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--macros", type=Path, default=DEFAULT_MACROS)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = load_json(args.report)
    analysis_audit = load_json(args.analysis_audit)
    rows = load_csv(args.rows)
    failures: list[str] = []
    if analysis_audit.get("passed") is not True:
        failures.append("independent raw analysis audit did not pass")
    if analysis_audit.get("report_sha256") != sha256(args.report):
        failures.append("analysis audit does not bind the supplied report")
    if analysis_audit.get("rule_rows_sha256") != sha256(args.rows):
        failures.append("analysis audit does not bind the supplied rule rows")
    expected_rows = len(report["datasets"]) * len(report["replication_seeds"]) * 5
    if len(rows) != expected_rows:
        failures.append(f"rule row count {len(rows)} != {expected_rows}")
    if failures:
        raise RuntimeError("; ".join(failures))

    make_figure(report, rows, args.pdf, args.png)
    args.tex.parent.mkdir(parents=True, exist_ok=True)
    args.tex.write_text(make_tex(report), encoding="utf-8")
    args.macros.write_text(make_macros(report), encoding="utf-8")
    package_audit = {
        "name": "VERA independent stress presentation-package audit",
        "passed": True,
        "confirmatory_passed": report["passed"],
        "report_sha256": sha256(args.report),
        "analysis_audit_sha256": sha256(args.analysis_audit),
        "rule_rows_sha256": sha256(args.rows),
        "tex_sha256": sha256(args.tex),
        "macros_sha256": sha256(args.macros),
        "figure_pdf_sha256": sha256(args.pdf),
        "figure_png_sha256": sha256(args.png),
        "rule_row_count": len(rows),
        "supported_datasets_passing_all_three": report[
            "supported_datasets_passing_all_three"
        ],
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(
        json.dumps(package_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(package_audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
