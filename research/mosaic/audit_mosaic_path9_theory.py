#!/usr/bin/env python3
"""Independently verify the locked path-to-9 theory-study receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREREG = (
    ROOT / "research/mosaic/prereg_mosaic_path9_theory_v1.json"
)
DEFAULT_REPORT = ROOT / "research/artifacts/mosaic_path9_theory_v1.json"
DEFAULT_OUTPUT = (
    ROOT / "research/artifacts/mosaic_path9_theory_v1_audit.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def bernoulli_kl(first: float, second: float) -> float:
    return (
        first * math.log(first / second)
        + (1.0 - first)
        * math.log((1.0 - first) / (1.0 - second))
    )


def binary_lower_bound(
    contract: float,
    margin: float,
    soundness: float,
    power: float,
) -> int:
    divergence = bernoulli_kl(
        0.5 + contract - margin,
        0.5 + contract + margin,
    )
    return math.ceil(
        math.log(1.0 / (2.0 * (soundness + power)))
        / divergence
    )


def weissman_upper_bound(
    alphabet_size: int,
    margin: float,
    soundness: float,
) -> int:
    return math.ceil(
        2.0
        * math.log((2**alphabet_size - 2) / soundness)
        / margin**2
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prereg = load(args.prereg)
    report = load(args.report)
    sidecar = args.prereg.with_suffix(".sha256")
    checks: dict[str, Any] = {}
    failures: list[str] = []
    try:
        lock_digest = sha256(args.prereg)
        checks["preregistration_hash_matches_sidecar"] = (
            sidecar.read_text(encoding="utf-8").split()[0] == lock_digest
        )
        checks["report_names_locked_hash"] = (
            report["preregistration_sha256"] == lock_digest
        )

        soundness = float(
            prereg["sample_complexity_study"]["soundness_error"]
        )
        power = float(
            prereg["sample_complexity_study"]["power_error"]
        )
        sample_rows_pass = True
        for row in report["sample_complexity"]["rows"]:
            lower = binary_lower_bound(
                float(row["contract"]),
                float(row["margin"]),
                soundness,
                power,
            )
            upper = weissman_upper_bound(
                int(row["alphabet_size"]),
                float(row["margin"]),
                soundness,
            )
            sample_rows_pass &= (
                int(row["lower_bound_per_stratum"]) == lower
                and int(row["weissman_upper_bound_per_stratum"]) == upper
                and upper >= lower
                and bool(row["pass"])
            )
        checks["sample_complexity_pass"] = bool(
            sample_rows_pass
            and report["sample_complexity"]["pass"]
        )

        transcript_pass = True
        for row in report["transcript"]["rows"]:
            coefficients = [
                float(value) for value in row["per_item_coefficients"]
            ]
            exact = float(row["exact_joint_coefficient"])
            bound = 1.0 - math.prod(
                1.0 - value for value in coefficients
            )
            transcript_pass &= (
                abs(bound - float(row["multiplicative_bound"]))
                <= 1e-12
                and abs(
                    bound
                    - exact
                    - float(row["bound_minus_exact"])
                )
                <= 1e-12
                and exact <= bound + 1e-12
                and bool(row["pass"])
            )
        checks["five_transcript_rows"] = (
            len(report["transcript"]["rows"]) == 5
        )
        checks["transcript_pass"] = bool(
            transcript_pass and report["transcript"]["pass"]
        )

        anytime_pass = True
        for alphabet in prereg["anytime_study"]["alphabets"]:
            cell = report["anytime"][str(alphabet)]
            exclusions = int(cell["anytime_false_exclusions"])
            repetitions = int(cell["repetitions"])
            histogram_total = sum(
                int(value)
                for value in cell["first_exclusion_histogram"].values()
            )
            anytime_pass &= (
                repetitions
                == int(
                    prereg["anytime_study"][
                        "repetitions_per_alphabet"
                    ]
                )
                and exclusions == histogram_total
                and abs(
                    float(cell["anytime_false_exclusion_rate"])
                    - exclusions / repetitions
                )
                <= 1e-15
                and exclusions / repetitions <= 0.07
                and bool(cell["pass"])
            )
        checks["three_anytime_alphabets"] = (
            len(report["anytime"]) == 3
        )
        checks["all_anytime_cells_pass"] = bool(anytime_pass)

        cover = report["continuous_cover"]
        cover_rate = int(cover["covered"]) / int(cover["repetitions"])
        checks["continuous_cover_pass"] = bool(
            abs(float(cover["coverage_rate"]) - cover_rate) <= 1e-15
            and cover_rate >= 0.93
            and float(cover["minimum_certificate_slack"]) >= 0.0
            and bool(cover["pass"])
        )
        checks["report_pass"] = bool(report["pass"])
        if not all(checks.values()):
            failures.append("one or more registered theory gates failed")
    except Exception as error:
        failures.append(f"{type(error).__name__}: {error}")

    output = {
        "name": "MOSAIC path-to-9 theory confirmation audit v1",
        "pass": not failures,
        "report": str(args.report.relative_to(ROOT)),
        "report_sha256": sha256(args.report),
        "checks": checks,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
