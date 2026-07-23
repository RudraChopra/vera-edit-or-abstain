#!/usr/bin/env python3
"""Aggregate the path-to-9 breadth and utility evidence without refitting."""

from __future__ import annotations

import glob
import json
from pathlib import Path
from statistics import median

import numpy as np
from scipy.stats import beta


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "research/artifacts/mosaic_path9_evidence_summary.json"
OUTPUT_TEX = (
    ROOT
    / "research/maintrack/mosaic_aaai2027/mosaic_path9_results.tex"
)


def clopper_pearson(successes: int, total: int) -> tuple[float, float]:
    lower = 0.0 if successes == 0 else float(
        beta.ppf(0.025, successes, total - successes + 1)
    )
    upper = 1.0 if successes == total else float(
        beta.ppf(0.975, successes + 1, total - successes)
    )
    return lower, upper


def expected_balanced_accuracy(
    counts: np.ndarray, channel: np.ndarray, decoder: np.ndarray
) -> float:
    by_label = counts.sum(axis=1)
    return float(
        np.mean(
            [
                np.sum(
                    by_label[label, :, None]
                    * channel[:, decoder == label]
                )
                / by_label[label].sum()
                for label in (0, 1)
            ]
        )
    )


def acs_summary() -> dict[str, object]:
    rows = []
    for path in glob.glob(
        str(
            ROOT
            / "research/artifacts/mosaic_acs_natural_shift_v1_receipts/*.json"
        )
    ):
        receipt = json.loads(Path(path).read_text(encoding="utf-8"))
        if "alphabets" not in receipt:
            continue
        cell = receipt["alphabets"]["4"]
        selected = cell["primary_selection"]["mosaic"]
        if selected["decision"] != "deploy":
            continue
        candidate = next(
            value
            for value in cell["rows"]
            if value["candidate"] == selected["candidate"]
        )
        counts = np.asarray(
            candidate["diagnostic_table"]["token_counts"],
            dtype=np.float64,
        )
        release = candidate["mosaic_release"]
        released = expected_balanced_accuracy(
            counts,
            np.asarray(release["release_channel"], dtype=np.float64),
            np.asarray(release["decoder"], dtype=np.int64),
        )
        pre_channel = expected_balanced_accuracy(
            counts,
            np.eye(4),
            np.asarray([0, 0, 1, 1]),
        )
        rows.append(
            {
                "candidate": selected["candidate"],
                "released_balanced_accuracy": released,
                "pre_channel_balanced_accuracy": pre_channel,
                "channel_utility_gap": pre_channel - released,
                "diagnostic_violation": not selected["diagnostic_safe"],
            }
        )
    return {
        "jobs": 60,
        "releases": len(rows),
        "diagnostic_violations": sum(
            value["diagnostic_violation"] for value in rows
        ),
        "median_released_balanced_accuracy": median(
            value["released_balanced_accuracy"] for value in rows
        ),
        "median_channel_utility_gap": median(
            value["channel_utility_gap"] for value in rows
        ),
        "maximum_channel_utility_gap": max(
            value["channel_utility_gap"] for value in rows
        ),
    }


def cinic_summary() -> dict[str, object]:
    report = json.loads(
        (
            ROOT / "research/artifacts/mosaic_cinic10_natural_v2.json"
        ).read_text(encoding="utf-8")
    )
    selected = [
        candidate
        for row in report["rows"]
        for candidate in row["candidates"]
        if candidate["candidate"] == row["selected_primary_candidate"]
    ]
    gaps = [
        value["unedited_token_balanced_accuracy"]
        - value["released_expected_balanced_accuracy"]
        for value in selected
    ]
    return {
        "jobs": len(report["rows"]),
        "releases": len(selected),
        "diagnostic_violations": sum(
            value["threshold_decisions"]["0.40"]["false_acceptance"]
            for value in selected
        ),
        "selected_method_counts": {
            method: sum(value["candidate"] == method for value in selected)
            for method in sorted(
                {value["candidate"] for value in selected}
            )
        },
        "median_released_balanced_accuracy": median(
            value["released_expected_balanced_accuracy"]
            for value in selected
        ),
        "median_channel_utility_gap": median(gaps),
        "maximum_channel_utility_gap": max(gaps),
    }


def main() -> None:
    biasbios = {
        "jobs": 20,
        "releases": 20,
        "diagnostic_violations": 0,
        "source": (
            "strict-v2 real bridge confirmation; all primary releases "
            "concentrate on BiasBios-Clinical"
        ),
    }
    acs = acs_summary()
    cinic = cinic_summary()
    domains = {
        "BiasBios-Clinical": biasbios,
        "ACS-geographic": acs,
        "CINIC10-natural-origin": cinic,
    }
    total_jobs = sum(value["jobs"] for value in domains.values())
    total_releases = sum(value["releases"] for value in domains.values())
    total_violations = sum(
        value["diagnostic_violations"] for value in domains.values()
    )
    leave_one_out = {}
    for held_out in domains:
        kept = [value for key, value in domains.items() if key != held_out]
        releases = sum(value["releases"] for value in kept)
        jobs = sum(value["jobs"] for value in kept)
        leave_one_out[held_out] = {
            "releases": releases,
            "jobs": jobs,
            "rate": releases / jobs,
            "exact_95_interval": clopper_pearson(releases, jobs),
        }
    proxy = json.loads(
        (
            ROOT / "research/artifacts/mosaic_real_proxy_v1.json"
        ).read_text(encoding="utf-8")
    )
    fare = json.loads(
        (
            ROOT
            / "research/artifacts/mosaic_fare_proxy_comparison_v1.json"
        ).read_text(encoding="utf-8")
    )
    payload = {
        "name": "MOSAIC path-to-9 evidence summary",
        "domains": domains,
        "cross_domain": {
            "jobs": total_jobs,
            "releases": total_releases,
            "release_rate": total_releases / total_jobs,
            "diagnostic_violations": total_violations,
            "leave_one_domain_out": leave_one_out,
        },
        "real_proxy": {
            "proxy_balanced_accuracy": proxy["proxy_balanced_accuracy"],
            "decision": proxy["release"]["decision"],
            "reason": proxy["release"]["reason"],
            "maximum_conditional_l1_radius": max(
                value
                for row in proxy["proxy_certificate"][
                    "conditional_l1_radii"
                ]
                for value in row
            ),
        },
        "official_fare_commensurate": fare["summary"],
        "claim_boundary": (
            "The three domain studies were locked separately and are pooled "
            "descriptively here. Utility gaps compare the stochastic channel "
            "with its fixed four-token task interface, not with every possible "
            "classifier on the full private representation."
        ),
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    OUTPUT_TEX.write_text(
        (
            "\\paragraph{Natural-shift breadth and interface utility.}\n"
            "Across three separately locked natural-shift confirmations, "
            f"MOSAIC releases {total_releases}/{total_jobs} jobs "
            f"({100*total_releases/total_jobs:.1f}\\%) with "
            f"{total_violations}/{total_releases} held-out violations: "
            "20/20 BiasBios-Clinical, "
            f"{acs['releases']}/60 ACS geographic, and "
            f"{cinic['releases']}/{cinic['jobs']} CINIC-10 image-origin jobs. "
            "The minimum leave-one-domain-out pooled release rate is "
            f"{100*min(value['rate'] for value in leave_one_out.values()):.1f}\\%. "
            "Relative to each selected four-token task interface, the median "
            f"channel cost is {100*acs['median_channel_utility_gap']:.2f} "
            "points on ACS and "
            f"{100*cinic['median_channel_utility_gap']:.2f} points on CINIC-10; "
            "the maximum costs are "
            f"{100*acs['maximum_channel_utility_gap']:.2f} and "
            f"{100*cinic['maximum_channel_utility_gap']:.2f} points. "
            "The supplement gives every seed and the full comparison "
            "boundary.\n"
        ),
        encoding="utf-8",
    )
    print(json.dumps(payload["cross_domain"], indent=2))


if __name__ == "__main__":
    main()
