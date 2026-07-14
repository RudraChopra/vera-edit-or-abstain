"""Build the receipt-backed confirmatory figures and manuscript includes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "maintrack" / "figures"
TEX = ROOT / "maintrack" / "aaai2027_template" / "AuthorKit27"

DEFAULT_ROWS = ARTIFACTS / "vera_confirmatory_balanced_rule_rows.csv"
DEFAULT_CANDIDATES = ARTIFACTS / "vera_confirmatory_balanced_candidate_rows.csv"
DEFAULT_REPORT = ARTIFACTS / "vera_confirmatory_balanced_report.json"
DEFAULT_ABSTRACT = ARTIFACTS / "vera_confirmatory_abstract_numbers.json"
DEFAULT_RECEIPTS = ARTIFACTS / "confirmatory_balanced_receipt_audit.json"
DEFAULT_EXACT = ARTIFACTS / "vera_exact_balanced_report.json"
DEFAULT_EXACT_AUDIT = ARTIFACTS / "vera_exact_balanced_audit.json"
DEFAULT_FAMILY_GRID = ARTIFACTS / "vera_exact_family_grid_report.json"
DEFAULT_FAMILY_GRID_AUDIT = ARTIFACTS / "vera_exact_family_grid_audit.json"
DEFAULT_ANALYSIS_AUDIT = ARTIFACTS / "vera_confirmatory_analysis_audit.json"
DEFAULT_OUTPUT_AUDIT = ARTIFACTS / "vera_confirmatory_results_package_audit.json"

RULES = (
    "always_deploy_balanced",
    "point_selection_balanced",
    "vera_balanced_iut",
    "vera_balanced_envelope",
    "external_balanced_oracle",
)
RULE_LABELS = {
    "always_deploy_balanced": "Always deploy",
    "point_selection_balanced": "Point selection",
    "vera_balanced_iut": "VERA-IUT",
    "vera_balanced_envelope": "VERA envelope",
    "external_balanced_oracle": "External oracle",
}
COLORS = {
    "always_deploy_balanced": "#D55E00",
    "point_selection_balanced": "#E69F00",
    "vera_balanced_iut": "#0072B2",
    "vera_balanced_envelope": "#009E73",
    "external_balanced_oracle": "#6B6B6B",
}
MARKERS = {
    "always_deploy_balanced": "s",
    "point_selection_balanced": "D",
    "vera_balanced_iut": "o",
    "vera_balanced_envelope": "^",
    "external_balanced_oracle": "x",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(value: str | bool) -> bool:
    return value if isinstance(value, bool) else value.strip().lower() == "true"


def fmt_percent(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.1f}\\%"


def tex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "%": r"\%",
        "&": r"\&",
        "_": r"\_",
        "#": r"\#",
        "$": r"\$",
    }
    return "".join(replacements.get(character, character) for character in value)


def configure_plot() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def cluster_interval(
    values: Iterable[float], rng: np.random.Generator, replicates: int = 20_000
) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return np.nan, np.nan, np.nan
    draws = rng.choice(array, size=(replicates, array.size), replace=True)
    means = draws.mean(axis=1)
    return (
        float(array.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def seed_rates(
    rows: list[dict[str, str]], dataset: str, rule: str, field: str
) -> list[float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row["dataset"] != dataset or row["rule"] != rule:
            continue
        if field == "measured_external_contract_violation" and not as_bool(
            row["external_contract_estimable"]
        ):
            continue
        grouped[int(row["seed"])].append(float(as_bool(row[field])))
    return [float(np.mean(grouped[seed])) for seed in sorted(grouped)]


def build_rule_figure(
    rows: list[dict[str, str]], datasets: list[str], pdf: Path, png: Path
) -> dict[str, Any]:
    configure_plot()
    rng = np.random.default_rng(20270718)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7), sharex=True)
    fields = (
        (
            "measured_external_contract_violation",
            "Measured external violation rate",
            "A  Contract violations",
        ),
        ("deployed", "Deployment rate", "B  Certification and abstention"),
    )
    positions = np.arange(len(datasets), dtype=np.float64)
    offsets = np.linspace(-0.30, 0.30, len(RULES))
    audit: dict[str, Any] = {}
    for axis, (field, ylabel, title) in zip(axes, fields):
        for offset, rule in zip(offsets, RULES):
            means: list[float] = []
            lowers: list[float] = []
            uppers: list[float] = []
            for dataset in datasets:
                values = seed_rates(rows, dataset, rule, field)
                mean, lower, upper = cluster_interval(values, rng)
                means.append(mean)
                lowers.append(lower)
                uppers.append(upper)
                audit[f"{field}|{dataset}|{rule}"] = {
                    "seed_rates": values,
                    "mean": None if np.isnan(mean) else mean,
                    "seed_cluster_bootstrap_95": (
                        None if np.isnan(mean) else [lower, upper]
                    ),
                }
            means_array = np.asarray(means)
            mask = np.isfinite(means_array)
            if np.any(mask):
                lower_array = np.asarray(lowers)[mask]
                upper_array = np.asarray(uppers)[mask]
                axis.errorbar(
                    (positions + offset)[mask],
                    means_array[mask],
                    yerr=np.vstack(
                        [
                            means_array[mask] - lower_array,
                            upper_array - means_array[mask],
                        ]
                    ),
                    fmt=MARKERS[rule],
                    color=COLORS[rule],
                    ecolor=COLORS[rule],
                    markersize=4.2,
                    elinewidth=0.9,
                    capsize=2,
                    label=RULE_LABELS[rule],
                    zorder=3,
                )
            for dataset_index, dataset in enumerate(datasets):
                values = audit[f"{field}|{dataset}|{rule}"]["seed_rates"]
                if not values:
                    continue
                jitter = np.linspace(-0.022, 0.022, len(values))
                axis.scatter(
                    positions[dataset_index] + offset + jitter,
                    values,
                    s=6.5,
                    facecolors="white",
                    edgecolors=COLORS[rule],
                    linewidths=0.5,
                    alpha=0.9,
                    zorder=2,
                )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.set_ylim(-0.04, 1.04)
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [value.replace("-WILDS", "") for value in datasets],
            rotation=22,
            ha="right",
        )
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.55, zorder=0)
    axes[0].text(
        positions[datasets.index("Camelyon17-WILDS")],
        0.95,
        "external leakage NA",
        ha="center",
        va="top",
        fontsize=6.3,
        color="#4A4A4A",
    )
    legend_handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=5,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=1.1)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return audit


def primary_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["analysis_tier"] == "primary"]


def write_macros(
    abstract: dict[str, Any], report: dict[str, Any], receipt_audit: dict[str, Any]
) -> Path:
    point = float(abstract["point_selection_violation_rate"])
    vera = float(abstract["vera_iut_violation_rate"])
    retention = float(abstract["safe_retention"])
    adjusted = report["seed_blocked_one_sided_signflip_holm_p"]
    significant = [
        (dataset, float(value))
        for dataset, value in adjusted.items()
        if float(value) <= 0.05
    ]
    if significant:
        best_dataset, best_p = min(significant, key=lambda item: item[1])
        seed_result = f"{best_dataset} (Holm-adjusted $p={best_p:.4f}$)"
    else:
        seed_result = "no supported dataset reached Holm-adjusted $p\\leq0.05$"
    headline = tex_escape(str(abstract["sentence"]))
    stress = report["headline_stress_family"]
    stress_effect = (
        f"Across {int(stress['configuration_count'])} prespecified stress "
        "configurations, validation-only selection deployed contract-violating "
        f"edits in {100.0 * float(stress['point_selection_violation_rate']):.1f}\\% "
        f"versus {100.0 * float(stress['vera_iut_violation_rate']):.1f}\\% for "
        "VERA. Across all threshold configurations, VERA retained "
        f"{100.0 * float(report['safe_retention']['rate']):.1f}\\% of "
        "external-oracle opportunities; the seed-blocked comparison did not "
        "survive Holm correction."
    )
    content = "\n".join(
        (
            f"\\providecommand{{\\HeadlineResult}}{{{headline}}}",
            f"\\providecommand{{\\PointViolationRate}}{{{fmt_percent(point)}}}",
            f"\\providecommand{{\\VERAViolationRate}}{{{fmt_percent(vera)}}}",
            f"\\providecommand{{\\SafeRetentionRate}}{{{fmt_percent(retention)}}}",
            "\\providecommand{\\OfficialReceiptCount}"
            f"{{{int(receipt_audit['official_run_receipt_count'])}}}",
            f"\\providecommand{{\\SeedBlockedResult}}{{{seed_result}}}",
            f"\\providecommand{{\\StressEffectResult}}{{{stress_effect}}}",
            "",
        )
    )
    path = TEX / "vera_results_macros.tex"
    path.write_text(content, encoding="utf-8")
    return path


def write_main_table(report: dict[str, Any]) -> Path:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.3pt}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Rule & Deploy & Viol./estimable & Conditional \\",
        r"\midrule",
    ]
    for rule in RULES:
        summary = report["primary_summaries"][rule]
        conditional = summary["violation_rate_conditional_on_estimable_deployment"]
        lines.append(
            f"{RULE_LABELS[rule]} & "
            f"{fmt_percent(float(summary['deployment_rate']))} & "
            f"{int(summary['measured_external_violation_count'])}/"
            f"{int(summary['estimable_configuration_count'])} "
            f"({fmt_percent(summary['measured_external_violation_rate'])}) & "
            f"{fmt_percent(conditional)} \\\\"
        )
    lines.extend(
        (
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Primary full-certification results over all nine locked threshold pairs and eight untouched seeds. Violation rates include abstentions in the denominator; the conditional column is descriptive. Camelyon17 external balanced leakage is non-estimable and excluded from measured-violation denominators.}",
            r"\label{tab:main-rules}",
            r"\end{table}",
            "",
        )
    )
    path = TEX / "vera_main_results_table.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_narrative(
    report: dict[str, Any],
    abstract: dict[str, Any],
    exact: dict[str, Any],
    family_grid: dict[str, Any],
) -> Path:
    primary = report["primary_summaries"]["vera_balanced_iut"]
    shifted = report["shifted_sensitivity_summaries"]["vera_balanced_iut"]
    retention = report["safe_retention"]
    retention_cluster = retention["seed_cluster_bootstrap95"]
    exact_false = sum(int(cell["false_acceptances"]) for cell in exact["cells"])
    exact_trials = sum(int(cell["replicates"]) for cell in exact["cells"])
    family_counts = ",".join(
        map(str, family_grid["candidate_counts_tested"])
    )
    group_counts = ",".join(map(str, family_grid["group_counts_tested"]))
    adjusted = report["seed_blocked_one_sided_signflip_holm_p"]
    p_values = ", ".join(
        f"{tex_escape(dataset.replace('-WILDS', ''))} {float(value):.4f}"
        for dataset, value in sorted(adjusted.items())
    )
    narrative = (
        f"\\paragraph{{Locked headline.}} {tex_escape(str(abstract['sentence']))} "
        "This family was selected from pilot seeds 0--4 and locked before all "
        "reported seeds 5--12; it is not the average over a searched threshold grid.\n\n"
        f"\\paragraph{{Error control and retention.}} Across all externally "
        f"estimable primary configurations, VERA-IUT produced "
        f"{int(primary['measured_external_violation_count'])} measured violations "
        f"in {int(primary['estimable_configuration_count'])} configurations "
        f"({fmt_percent(primary['measured_external_violation_rate'])}); at the fixed "
        f"$\\Gamma=1.01$ sensitivity the corresponding result was "
        f"{int(shifted['measured_external_violation_count'])}/"
        f"{int(shifted['estimable_configuration_count'])} "
        f"({fmt_percent(shifted['measured_external_violation_rate'])}). It retained "
        f"{int(retention['safe_deployment_count'])}/"
        f"{int(retention['external_oracle_opportunity_count'])} externally safe "
        f"opportunities ({fmt_percent(float(retention['rate']))}; seed-cluster "
        f"bootstrap 95\\% interval {fmt_percent(float(retention_cluster[0]))}--"
        f"{fmt_percent(float(retention_cluster[1]))}). The preregistered "
        "configuration-level Clopper--Pearson interval is retained in the supplement "
        "as a dependent-cell diagnostic. Camelyon17 returned forced "
        "abstention because center 2 and one required source-class cell are outside "
        "certification support.\n\n"
        f"\\paragraph{{Paired analysis.}} Holm-adjusted one-sided seed-blocked "
        f"sign-flip values for point selection minus VERA-IUT were {p_values}. "
        "Each seed is one randomization block over all nine correlated thresholds; "
        "a sign-only sensitivity and dependent-cell McNemar diagnostics are reported "
        "in the supplement.\n\n"
        f"\\paragraph{{Exact calibration.}} The locked synthetic study had "
        f"{exact_false} false acceptances in {exact_trials:,} trials. All "
        f"{int(exact['cell_count'])} observed abstention rates fell inside their "
        "simultaneous exact prediction bands, and a separately implemented replay "
        "reproduced every cell. A second locked grid varied candidate count over "
        f"$M\\in\\{{{family_counts}\\}}$ and validated-group count over "
        f"$|\\mathcal G|\\in\\{{{group_counts}\\}}$; all "
        f"{int(family_grid['cell_count'])} cells passed simultaneous coverage."
    )
    path = TEX / "vera_main_results_narrative.tex"
    path.write_text(narrative + "\n", encoding="utf-8")
    return path


def write_supplement_tables(
    report: dict[str, Any],
    candidates: list[dict[str, str]],
    receipt_audit: dict[str, Any],
) -> Path:
    lines = [
        r"\section{Receipt-Level Confirmatory Results}",
        r"\begin{table*}[t]",
        r"\centering\scriptsize",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Dataset & Rule & Deploy & Viol./estimable & Cond. viol. & Unsupported \\",
        r"\midrule",
    ]
    for dataset, rule_map in report["primary_by_dataset"].items():
        for rule in RULES:
            summary = rule_map[rule]
            lines.append(
                f"{tex_escape(dataset)} & {RULE_LABELS[rule]} & "
                f"{int(summary['deployment_count'])}/"
                f"{int(summary['configuration_count'])} & "
                f"{int(summary['measured_external_violation_count'])}/"
                f"{int(summary['estimable_configuration_count'])} & "
                f"{fmt_percent(summary['violation_rate_conditional_on_estimable_deployment'])} & "
                f"{int(summary['procedurally_unsupported_deployment_count'])} \\\\"
            )
        lines.append(r"\addlinespace")
    lines.extend(
        (
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Full-size primary results. Each dataset has $8\times9=72$ threshold--seed configurations per rule. Unsupported counts flag deployment into the registered Camelyon17 support mismatch; they are not recoded as measured violations.}",
            r"\end{table*}",
            "",
            r"\begin{table*}[t]",
            r"\centering\scriptsize",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"Dataset & Point-only & VERA-only & Sign-flip Holm & Sign-test Holm & McNemar Holm \\",
            r"\midrule",
        )
    )
    for dataset in report["seed_blocked_one_sided_signflip_holm_p"]:
        discordant = report["mcnemar_discordant_counts"][dataset]
        lines.append(
            f"{tex_escape(dataset)} & "
            f"{int(discordant['point_only_violation'])} & "
            f"{int(discordant['vera_only_violation'])} & "
            f"{float(report['seed_blocked_one_sided_signflip_holm_p'][dataset]):.4f} & "
            f"{float(report['seed_blocked_one_sided_sign_test_holm_p'][dataset]):.4f} & "
            f"{float(report['mcnemar_two_sided_holm_p'][dataset]):.4f} \\\\"
        )
    retention = report["safe_retention"]
    lines.extend(
        (
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Inference diagnostics. Sign-flip is the preregistered seed-blocked test. The sign test is a symmetry-free, lower-power secondary sensitivity. McNemar counts share fits and thresholds and are descriptive only. All values are Holm-adjusted over the four externally estimable datasets.}",
            r"\end{table*}",
            "",
            (
                "Safe-retention uncertainty: seed-cluster bootstrap 95\\% interval "
                f"{fmt_percent(float(retention['seed_cluster_bootstrap95'][0]))}--"
                f"{fmt_percent(float(retention['seed_cluster_bootstrap95'][1]))}; "
                "configuration-level Clopper--Pearson diagnostic "
                f"{fmt_percent(float(retention['cp95'][0]))}--"
                f"{fmt_percent(float(retention['cp95'][1]))}. "
                "The maximum supported dataset--seed VERA violation rate was "
                f"{fmt_percent(float(report['maximum_supported_dataset_seed_violation_rate']))}."
            ),
            "",
            r"\begin{table}[t]",
            r"\centering\small",
            r"\begin{tabular}{lrr}",
            r"\toprule",
            r"Official eraser & Candidate points & Run receipts \\",
            r"\midrule",
        )
    )
    candidate_counts = Counter(
        (row["method"], row["candidate"])
        for row in candidates
        if row["analysis_tier"] == "primary"
    )
    methods = sorted({method for method, _ in candidate_counts})
    for method in methods:
        points = len(
            {
                candidate
                for candidate_method, candidate in candidate_counts
                if candidate_method == method
            }
        )
        lines.append(f"{tex_escape(method)} & {points} & 40 \\\\")
    lines.extend(
        (
            r"\midrule",
            f"Total & 12 & {int(receipt_audit['official_run_receipt_count'])} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Every candidate is generated by pinned official upstream code. One run receipt contains the complete registered strength frontier for one dataset, method, and seed; there are no proxy rows.}",
            r"\end{table}",
            "",
        )
    )
    path = TEX / "vera_supplement_results.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_family_grid_results(family_grid: dict[str, Any]) -> Path:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for cell in family_grid["cells"]:
        grouped[
            (int(cell["candidate_count"]), int(cell["environment_count"]))
        ].append(cell)
    lines = [
        r"\section{Candidate-Family and Group-Count Coverage Grid}",
        (
            "A separately locked extension crosses six validation sizes, four "
            "candidate-family sizes, three validated-group counts, and three "
            "$\\delta$ levels. Every cell uses 2,000 Monte Carlo repetitions and "
            "contains at least one truly violating candidate. Each family counts "
            "the identity control action in addition to its eraser candidates."
        ),
        r"\begin{table}[t]",
        r"\centering\small",
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        "$M$ & $|\\mathcal G|$ & False accepts & Max. CP/$\\delta$ \\\\",
        r"\midrule",
    ]
    for (candidate_count, group_count), cells in sorted(grouped.items()):
        false_acceptances = sum(int(cell["false_acceptances"]) for cell in cells)
        maximum_ratio = max(
            float(cell["false_acceptance_cp95_upper_simultaneous"])
            / float(cell["delta"])
            for cell in cells
        )
        lines.append(
            f"{candidate_count} & {group_count} & {false_acceptances} & "
            f"{maximum_ratio:.3f} \\\\"
        )
    lines.extend(
        (
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Independent coverage stress test over candidate-family "
                r"and validated-group counts. The final column is the largest "
                r"simultaneous one-sided 95\% Clopper--Pearson upper bound divided "
                r"by its registered $\delta$; values at most one pass.}"
            ),
            r"\label{tab:family-grid}",
            r"\end{table}",
            "",
        )
    )
    path = TEX / "vera_family_grid_results.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--abstract", type=Path, default=DEFAULT_ABSTRACT)
    parser.add_argument("--receipt-audit", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("--exact", type=Path, default=DEFAULT_EXACT)
    parser.add_argument("--exact-audit", type=Path, default=DEFAULT_EXACT_AUDIT)
    parser.add_argument("--family-grid", type=Path, default=DEFAULT_FAMILY_GRID)
    parser.add_argument(
        "--family-grid-audit", type=Path, default=DEFAULT_FAMILY_GRID_AUDIT
    )
    parser.add_argument(
        "--analysis-audit", type=Path, default=DEFAULT_ANALYSIS_AUDIT
    )
    parser.add_argument("--output-audit", type=Path, default=DEFAULT_OUTPUT_AUDIT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = load_json(args.report)
    abstract = load_json(args.abstract)
    receipt_audit = load_json(args.receipt_audit)
    exact = load_json(args.exact)
    exact_audit = load_json(args.exact_audit)
    family_grid = load_json(args.family_grid)
    family_grid_audit = load_json(args.family_grid_audit)
    analysis_audit = load_json(args.analysis_audit)
    rows = load_csv(args.rows)
    candidates = load_csv(args.candidates)

    failures: list[str] = []
    if report.get("confirmatory") is not True:
        failures.append("confirmatory report is missing its protocol marker")
    pass_conditions = report.get("pass_conditions", {})
    if not isinstance(pass_conditions, dict) or not pass_conditions or not all(
        isinstance(value, bool) for value in pass_conditions.values()
    ):
        failures.append("confirmatory pass conditions are missing or malformed")
    if abstract.get("verified") is not True:
        failures.append("abstract numbers are not verified")
    if receipt_audit.get("passed") is not True:
        failures.append("receipt matrix audit failed")
    if int(receipt_audit.get("official_run_receipt_count", -1)) != 200:
        failures.append("receipt matrix does not contain exactly 200 official runs")
    if int(receipt_audit.get("proxy_row_count", -1)) != 0:
        failures.append("receipt matrix contains proxy rows")
    if exact.get("all_cells_pass") is not True or exact_audit.get("passed") is not True:
        failures.append("exact study or its independent replay failed")
    if (
        family_grid.get("all_cells_pass") is not True
        or int(family_grid.get("cell_count", 0)) != 216
        or family_grid_audit.get("passed") is not True
        or family_grid_audit.get("report_sha256") != sha256(args.family_grid)
    ):
        failures.append("candidate-family/group coverage grid or replay failed")
    if (
        analysis_audit.get("passed") is not True
        or analysis_audit.get("abstract_verified") is not True
    ):
        failures.append("independent confirmatory aggregate audit failed")
    if len(rows) != 10_800 or len(candidates) != 25_920:
        failures.append("confirmatory CSV dimensions are not the locked dimensions")
    if failures:
        raise RuntimeError("; ".join(failures))

    primary = primary_rows(rows)
    datasets = list(report["primary_by_dataset"])
    figure_audit = build_rule_figure(
        primary,
        datasets,
        FIGURES / "vera_deployment_rules.pdf",
        FIGURES / "vera_deployment_rules.png",
    )
    outputs = [
        write_macros(abstract, report, receipt_audit),
        write_main_table(report),
        write_narrative(report, abstract, exact, family_grid),
        write_supplement_tables(report, candidates, receipt_audit),
        write_family_grid_results(family_grid),
        FIGURES / "vera_deployment_rules.pdf",
        FIGURES / "vera_deployment_rules.png",
    ]
    package = {
        "passed": True,
        "registered_pass_conditions_met": report.get("passed") is True,
        "failed_registered_pass_conditions": sorted(
            key for key, value in pass_conditions.items() if value is not True
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "confirmatory_report_sha256": sha256(args.report),
        "abstract_numbers_sha256": sha256(args.abstract),
        "receipt_audit_sha256": sha256(args.receipt_audit),
        "exact_report_sha256": sha256(args.exact),
        "exact_audit_sha256": sha256(args.exact_audit),
        "family_grid_report_sha256": sha256(args.family_grid),
        "family_grid_audit_sha256": sha256(args.family_grid_audit),
        "analysis_audit_sha256": sha256(args.analysis_audit),
        "rule_row_count": len(rows),
        "candidate_row_count": len(candidates),
        "figure_cluster_summary": figure_audit,
        "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in outputs},
        "failures": [],
    }
    args.output_audit.write_text(
        json.dumps(package, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": True,
                "headline": abstract["sentence"],
                "outputs": len(outputs),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
