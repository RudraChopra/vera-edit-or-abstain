"""Estimate local feasibility of a full no-cap Camelyon17 MANCE++ run."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
DEFAULT_JSON = ARTIFACT_DIR / "camelyon17_mance_scaling_feasibility.json"
DEFAULT_MD = ARTIFACT_DIR / "camelyon17_mance_scaling_feasibility.md"
DEFAULT_RECEIPTS = [
    ARTIFACT_DIR / "camelyon17_mancepp_reference_diagnostic_receipt.json",
    ARTIFACT_DIR / "camelyon17_mancepp_reference_40k_diagnostic_receipt.json",
    ARTIFACT_DIR / "camelyon17_mancepp_reference_80k_diagnostic_receipt.json",
]
FULL_NO_CAP_RECEIPT = ARTIFACT_DIR / "camelyon17_mancepp_reference_full_nocap_receipt.json"


@dataclass(frozen=True)
class RunSummary:
    receipt: str
    train: int
    validation: int
    external: int
    n_steps: int
    runtime_seconds: float
    work_units: int


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def work_units(train: int, validation: int, external: int, n_steps: int) -> int:
    # Exact local-tangent MANCE queries each split against the training reference
    # at every edit step; this is the dominant CPU path on this non-CUDA Mac.
    return int(train * (train + validation + external) * n_steps)


def summarize(path: Path) -> RunSummary:
    receipt = load_json(path)
    counts = receipt["sample"]["counts"]
    method = receipt["reference_method"]
    train = int(counts["train"])
    validation = int(counts["validation"])
    external = int(counts["external"])
    n_steps = int(method["n_steps"])
    return RunSummary(
        receipt=str(path),
        train=train,
        validation=validation,
        external=external,
        n_steps=n_steps,
        runtime_seconds=float(receipt["runtime_seconds"]),
        work_units=work_units(train, validation, external, n_steps),
    )


def format_hours(seconds: float) -> str:
    return f"{seconds / 3600.0:.2f} h"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Camelyon17 MANCE++ Scaling Feasibility",
        "",
        f"Generated at UTC: `{report['created_at_utc']}`",
        f"Recommendation: `{report['recommendation']}`",
        "",
        "| Train | Validation | External | Runtime | Work units |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in report["observed_runs"]:
        lines.append(
            f"| {run['train']} | {run['validation']} | {run['external']} | "
            f"{format_hours(float(run['runtime_seconds']))} | {run['work_units']} |"
        )
    full_run = report.get("full_no_cap_observed_run")
    full = report["full_no_cap_counts"]
    lines.extend(
        [
            "",
            "## Full No-Cap Projection",
            "",
            f"- Full train/validation/external counts: `{full}`",
            f"- Linear lower-bound estimate: `{format_hours(report['linear_lower_bound_seconds'])}`",
            f"- Recent superlinear estimate: `{format_hours(report['recent_superlinear_estimate_seconds'])}`",
            f"- Recent scaling exponent: `{report['recent_scaling_exponent']:.3f}`",
            f"- Full no-cap completed: `{report['full_no_cap_completed']}`",
            (
                f"- Full no-cap observed runtime: `{format_hours(float(full_run['runtime_seconds']))}`"
                if full_run
                else "- Full no-cap observed runtime: `<not completed>`"
            ),
            "",
            "## Boundary",
            "",
            str(report["claim_boundary"]),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MD)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    runs = [summarize(path) for path in DEFAULT_RECEIPTS]
    runs = sorted(runs, key=lambda item: item.train)
    largest_receipt = load_json(Path(runs[-1].receipt))
    full_counts = largest_receipt["sample"]["full_counts"]
    n_steps = runs[-1].n_steps
    full_work = work_units(
        int(full_counts["train"]),
        int(full_counts["validation"]),
        int(full_counts["external"]),
        n_steps,
    )

    largest = runs[-1]
    linear_lower_bound = largest.runtime_seconds * (full_work / largest.work_units)
    previous = runs[-2]
    recent_exponent = math.log(largest.runtime_seconds / previous.runtime_seconds) / math.log(
        largest.work_units / previous.work_units
    )
    recent_superlinear = largest.runtime_seconds * (
        full_work / largest.work_units
    ) ** recent_exponent
    full_completed = FULL_NO_CAP_RECEIPT.exists()
    full_run = summarize(FULL_NO_CAP_RECEIPT) if full_completed else None

    report = {
        "name": "Camelyon17 MANCE++ local scaling feasibility audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "observed_runs": [asdict(run) for run in runs],
        "full_no_cap_completed": full_completed,
        "full_no_cap_observed_run": asdict(full_run) if full_run else None,
        "full_no_cap_counts": {
            "train": int(full_counts["train"]),
            "validation": int(full_counts["validation"]),
            "external": int(full_counts["external"]),
        },
        "full_no_cap_work_units": full_work,
        "linear_lower_bound_seconds": linear_lower_bound,
        "recent_scaling_exponent": recent_exponent,
        "recent_superlinear_estimate_seconds": recent_superlinear,
        "storage_blocker": False,
        "compute_blocker": not full_completed,
        "recommendation": (
            "use the full no-cap Camelyon17 MANCE++ receipt as the current reference row"
            if full_completed
            else "keep Camelyon17 MANCE++ as an 80k diagnostic locally; schedule full no-cap on a dedicated long CPU/CUDA machine if reference parity becomes mandatory"
        ),
        "claim_boundary": (
            "The full no-cap MANCE++ Camelyon17 receipt is materialized and claim-grade "
            "under FARO's frozen-representation protocol."
            if full_completed
            else (
                "The 80k MANCE++ Camelyon17 receipt is a large official-code diagnostic, "
                "not a full no-cap reference row. Storage is no longer the blocker; local "
                "exact-nearest-neighbor tangent computation is the blocker."
            )
        ),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.markdown_out, report)
    print("Camelyon17 MANCE++ scaling feasibility audit complete")
    print(f"linear_lower_bound_hours={linear_lower_bound / 3600.0:.2f}")
    print(f"recent_superlinear_estimate_hours={recent_superlinear / 3600.0:.2f}")
    print(f"full_no_cap_completed={str(full_completed).lower()}")
    print(f"report={args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
