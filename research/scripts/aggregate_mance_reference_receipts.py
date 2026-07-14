"""Aggregate per-seed MANCE reference receipts into statistics artifacts."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--method-name", default="MANCE++ reference")
    parser.add_argument("--min-seeds", type=int, default=5)
    return parser.parse_args()


def summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "sd": float("nan"), "ci95": float("nan")}
    if len(values) == 1:
        return {"mean": float(values[0]), "sd": 0.0, "ci95": 0.0}
    sd = stdev(values)
    return {
        "mean": float(mean(values)),
        "sd": float(sd),
        "ci95": float(1.96 * sd / math.sqrt(len(values))),
    }


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def main() -> None:
    args = parse_args()
    paths = [Path(path) for path in sorted(glob.glob(args.input_glob))]
    receipts = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not receipts:
        raise ValueError(f"no receipts matched {args.input_glob!r}")

    rows: list[dict[str, object]] = []
    metric_names = sorted(receipts[0]["metrics"]["before"])
    for path, receipt in zip(paths, receipts):
        seed = int(receipt["reference_method"].get("seed", receipt["sample"]["seed"]))
        row: dict[str, object] = {
            "path": str(path),
            "seed": seed,
            "claim_grade_reference_row": bool(receipt.get("claim_grade_reference_row")),
            "runtime_seconds": receipt.get("runtime_seconds"),
        }
        for metric in metric_names:
            row[f"before_{metric}"] = receipt["metrics"]["before"].get(metric)
            row[f"after_{metric}"] = receipt["metrics"]["after"].get(metric)
            row[f"delta_{metric}"] = receipt["metrics"]["delta"].get(metric)
        rows.append(row)

    stats: dict[str, dict[str, dict[str, float]]] = {"before": {}, "after": {}, "delta": {}}
    for phase in stats:
        for metric in metric_names:
            values = [
                float(row[f"{phase}_{metric}"])
                for row in rows
                if row.get(f"{phase}_{metric}") is not None
            ]
            stats[phase][metric] = summary(values)

    report = {
        "name": "VERA MANCE reference statistical report",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": args.dataset_name,
        "method_name": args.method_name,
        "input_glob": args.input_glob,
        "receipt_count": len(receipts),
        "min_seeds": args.min_seeds,
        "seeds": [row["seed"] for row in rows],
        "claim_grade_statistics": len(receipts) >= args.min_seeds
        and all(bool(receipt.get("claim_grade_reference_row")) for receipt in receipts),
        "per_seed_csv": str(args.output_csv),
        "statistics": stats,
        "receipts": [str(path) for path in paths],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# MANCE Reference Statistical Report",
        "",
        f"- Dataset: `{args.dataset_name}`",
        f"- Method: `{args.method_name}`",
        f"- Seeds: `{report['seeds']}`",
        f"- Claim-grade statistics: `{report['claim_grade_statistics']}`",
        "",
        "## Mean Metrics",
        "",
        "| Metric | Before mean | After mean | Delta mean | Delta 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in metric_names:
        lines.append(
            "| "
            f"`{metric}` | "
            f"{fmt(stats['before'][metric]['mean'])} | "
            f"{fmt(stats['after'][metric]['mean'])} | "
            f"{fmt(stats['delta'][metric]['mean'])} | "
            f"{fmt(stats['delta'][metric]['ci95'])} |"
        )
    lines.append("")
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(args.output_json), "claim_grade_statistics": report["claim_grade_statistics"]}))


if __name__ == "__main__":
    main()
