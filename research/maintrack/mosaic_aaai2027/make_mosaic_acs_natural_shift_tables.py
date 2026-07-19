#!/usr/bin/env python3
"""Render the locked ACS natural-shift summary as main and supplement LaTeX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
DEFAULT_SUMMARY = REPOSITORY / "research/artifacts/mosaic_acs_natural_shift_v1_summary.json"
DEFAULT_AUDIT = REPOSITORY / "research/artifacts/mosaic_acs_natural_shift_v1_audit.json"
DEFAULT_MAIN = HERE / "mosaic_acs_natural_shift_results.tex"
DEFAULT_SUPPLEMENT = HERE / "mosaic_acs_natural_shift_supplement_results.tex"


def count(cell: dict, numerator: str, denominator: str) -> str:
    denominator_value = int(cell[denominator])
    if denominator_value == 0:
        return "--"
    return f"{int(cell[numerator])}/{denominator_value}"


def primary(summary: dict, alphabet: str, rule: str) -> dict:
    return summary["cells"][alphabet][rule]["0.40"]


def main_table(summary: dict) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{clrrr}",
        r"\hline",
        r"$K$ & Rule & Releases & Diagnostic viol. & Batch viol. \\",
        r"\hline",
    ]
    for alphabet in summary["alphabets"]:
        for rule, label in (("mosaic", "MOSAIC"), ("direct", "Direct table")):
            cell = primary(summary, alphabet, rule)
            lines.append(
                f"{alphabet} & {label} & "
                f"{count(cell, 'deployments', 'jobs')} & "
                f"{count(cell, 'false_acceptances', 'diagnostic_estimable_deployments')} & "
                f"{count(cell, 'operational_contract_violations', 'operational_draws')} \\\\"
            )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\caption{Fresh natural geographic shift across 60 registered ACS jobs "
            r"(three tasks, four target states, five seeds). Diagnostic violations are "
            r"among released jobs with complete held-out PUMA support. Batch "
            r"violations count 100 operational batch replays per primary release, each "
            r"sampling one public token per held-out person.}",
            r"\label{tab:acs-natural-shift}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def main_results(summary: dict) -> str:
    k4_m = primary(summary, "4", "mosaic")
    k4_d = primary(summary, "4", "direct")
    k8_d = primary(summary, "8", "direct")
    k4_tasks = summary["primary_breakdowns"]["4"]["task"]
    k4_states = summary["primary_breakdowns"]["4"]["state"]
    state_releases = ", ".join(
        f"{state}={cells['mosaic']['deployments']}" for state, cells in k4_states.items()
    )
    return "\n".join(
        [
            r"\paragraph{Natural multi-state geographic transfer.}",
            "Before downloading the four target states, we locked 60 ACS jobs spanning "
            "income, employment, and public-coverage tasks; Washington, Illinois, New "
            "York, and Florida targets; five seeds; and whole-PUMA holdouts. Every job "
            "runs the same 13-candidate official frontier at $K\\in\\{4,8\\}$. "
            f"At the primary $.40$ contract, $K=4$ MOSAIC releases {k4_m['deployments']}/60 "
            f"with {k4_m['false_acceptances']}/{k4_m['diagnostic_estimable_deployments']} "
            f"held-out violations and {k4_m['operational_contract_violations']}/"
            f"{k4_m['operational_draws']:,} violating operational batches. The releases "
            f"span all four targets ({state_releases}) and two tasks: employment="
            f"{k4_tasks['employment']['mosaic']['deployments']} and income="
            f"{k4_tasks['income']['mosaic']['deployments']}. The direct table releases "
            f"{k4_d['deployments']}/60 with {k4_d['false_acceptances']}/"
            f"{k4_d['diagnostic_estimable_deployments']} held-out violations. At $K=8$, "
            f"MOSAIC abstains on all 60 jobs while the direct table releases {k8_d['deployments']}/60 "
            f"with {k8_d['false_acceptances']}/{k8_d['diagnostic_estimable_deployments']} "
            "held-out violations, exposing the confidence cost of the larger token table. The final "
            "column of Table~\\ref{tab:acs-natural-shift} repeats the held-out diagnostic "
            "100 times, sampling one certified public token per person in each replay.",
            "",
            main_table(summary),
        ]
    )


def breakdown_table(summary: dict, field: str, label: str, table_label: str) -> str:
    values = summary["primary_breakdowns"]["4"][field].keys()
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        f"{label} & $K$ & MOSAIC rel. & MOSAIC viol. & Direct rel. & Direct viol. \\\\",
        r"\hline",
    ]
    display = {"public_coverage": "public coverage"}
    for value in values:
        for alphabet in summary["alphabets"]:
            cells = summary["primary_breakdowns"][alphabet][field][value]
            mosaic = cells["mosaic"]
            direct = cells["direct"]
            lines.append(
                f"{display.get(value, value)} & {alphabet} & "
                f"{count(mosaic, 'deployments', 'jobs')} & "
                f"{count(mosaic, 'false_acceptances', 'diagnostic_estimable_deployments')} & "
                f"{count(direct, 'deployments', 'jobs')} & "
                f"{count(direct, 'false_acceptances', 'diagnostic_estimable_deployments')} \\\\"
            )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            f"\\caption{{Primary-contract results by {label.lower()}. Violations are "
            "measured only on released jobs with complete held-out PUMA support.}",
            f"\\label{{{table_label}}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def threshold_table(summary: dict) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{clrrr}",
        r"\hline",
        r"$K$ & Rule & Error cap & Releases & Diagnostic viol. \\",
        r"\hline",
    ]
    for alphabet in summary["alphabets"]:
        for rule, display in (("mosaic", "MOSAIC"), ("direct", "Direct table")):
            for threshold in summary["thresholds"]:
                cell = summary["cells"][alphabet][rule][threshold]
                lines.append(
                    f"{alphabet} & {display} & {threshold} & "
                    f"{count(cell, 'deployments', 'jobs')} & "
                    f"{count(cell, 'false_acceptances', 'diagnostic_estimable_deployments')} \\\\"
                )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\caption{Complete registered utility-threshold frontier. No cell is omitted.}",
            r"\label{supp:tab:acs-natural-thresholds}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def selected_methods(summary: dict) -> str:
    rows = []
    for alphabet in summary["alphabets"]:
        for rule in ("mosaic", "direct"):
            counts = primary(summary, alphabet, rule)["selected_method_counts"]
            rendered = ", ".join(f"{name}={value}" for name, value in counts.items()) or "none"
            display_rule = "MOSAIC" if rule == "mosaic" else "direct"
            rows.append(f"$K={alphabet}$ {display_rule}: {rendered}")
    return "; ".join(rows)


def supplement_results(summary: dict, audit: dict) -> str:
    failures = sum(int(value) for value in summary["optimization_failures"].values())
    return "\n".join(
        [
            r"\section{Natural Multi-State ACS Confirmation}",
            "The data lock authenticates 12 processed stores before execution. The 60 "
            "receipts contain 1,560 official-candidate rows across two fine alphabets; "
            f"{failures} candidate rows record optimization failures. Bridge and diagnostic "
            "sets use disjoint target-state PUMAs in every receipt. Tables below report "
            "all tasks and states at the registered $.40$ contract. Primary selected-method "
            f"counts are {selected_methods(summary)}.",
            f"Independent exact replay checks {audit['receipts_replayed']} receipts, "
            f"{audit['bridges_replayed']:,} bridge certificates, {audit['releases_replayed']:,} "
            f"outward release bounds, and {audit['selections_replayed']:,} stored selections "
            f"with {len(audit['failures'])} failures.",
            "",
            threshold_table(summary),
            "",
            breakdown_table(summary, "task", "Task", "supp:tab:acs-natural-task"),
            "",
            breakdown_table(summary, "state", "Target", "supp:tab:acs-natural-state"),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--main-output", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--supplement-output", type=Path, default=DEFAULT_SUPPLEMENT)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if not audit.get("passed"):
        raise ValueError("refusing to render a failed natural-shift audit")
    args.main_output.write_text(main_results(summary) + "\n", encoding="utf-8")
    args.supplement_output.write_text(
        supplement_results(summary, audit) + "\n", encoding="utf-8"
    )
    print(args.main_output)
    print(args.supplement_output)


if __name__ == "__main__":
    main()
