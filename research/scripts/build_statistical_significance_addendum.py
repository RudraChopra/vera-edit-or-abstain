"""Build nonparametric paired significance-test addendum for VERA artifacts."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
DEFAULT_JSON = ARTIFACT_DIR / "faro_statistical_significance_addendum.json"
DEFAULT_MD = ARTIFACT_DIR / "faro_statistical_significance_addendum.md"


def exact_sign_test(deltas: list[float]) -> dict[str, object]:
    nonzero = [delta for delta in deltas if abs(delta) > 1e-12]
    n = len(nonzero)
    positives = sum(1 for delta in nonzero if delta > 0)
    negatives = sum(1 for delta in nonzero if delta < 0)
    if n == 0:
        return {
            "n_nonzero": 0,
            "positive_count": positives,
            "negative_count": negatives,
            "two_sided_p": None,
            "note": "all paired deltas are zero",
        }
    k = min(positives, negatives)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return {
        "n_nonzero": n,
        "positive_count": positives,
        "negative_count": negatives,
        "two_sided_p": min(1.0, 2.0 * tail),
        "note": "exact two-sided sign test under median paired delta zero",
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def index_by_method_seed(rows: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, str]]:
    out: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        key = row.get("method_key") or row.get("method")
        out[(str(key), int(row["seed"]))] = row
    return out


def paired_method_tests(
    *,
    path: Path,
    dataset: str,
    method_a: str,
    method_b: str,
    metrics: list[str],
) -> list[dict[str, object]]:
    rows = read_rows(path)
    by_key = index_by_method_seed(rows)
    seeds = sorted({seed for method, seed in by_key if method in {method_a, method_b}})
    tests: list[dict[str, object]] = []
    for metric in metrics:
        deltas = []
        shared = []
        for seed in seeds:
            a = by_key.get((method_a, seed))
            b = by_key.get((method_b, seed))
            if a is None or b is None:
                continue
            deltas.append(float(a[metric]) - float(b[metric]))
            shared.append(seed)
        tests.append(
            {
                "dataset": dataset,
                "comparison": f"{method_a} minus {method_b}",
                "metric": metric,
                "seeds": shared,
                "mean_delta": sum(deltas) / len(deltas) if deltas else None,
                "deltas": deltas,
                "sign_test": exact_sign_test(deltas),
            }
        )
    return tests


def mance_before_after_tests(path: Path) -> list[dict[str, object]]:
    rows = read_rows(path)
    tests: list[dict[str, object]] = []
    delta_columns = [name for name in rows[0] if name.startswith("delta_")]
    for column in delta_columns:
        deltas = [float(row[column]) for row in rows]
        tests.append(
            {
                "dataset": "waterbirds",
                "comparison": "MANCE++ after minus before",
                "metric": column.removeprefix("delta_"),
                "seeds": [int(row["seed"]) for row in rows],
                "mean_delta": sum(deltas) / len(deltas),
                "deltas": deltas,
                "sign_test": exact_sign_test(deltas),
            }
        )
    return tests


def write_markdown(path: Path, report: dict[str, object]) -> None:
    lines = [
        "# VERA Statistical Significance Addendum",
        "",
        f"Generated at UTC: `{report['created_at_utc']}`",
        "",
        "| Dataset | Comparison | Metric | Mean delta | Sign counts | p-value |",
        "| --- | --- | --- | ---: | --- | ---: |",
    ]
    for test in report["tests"]:
        sign = test["sign_test"]
        p_value = sign.get("two_sided_p")
        p_text = "" if p_value is None else f"{float(p_value):.6f}"
        mean_value = test["mean_delta"]
        mean_text = "" if mean_value is None else f"{float(mean_value):.6f}"
        lines.append(
            "| "
            f"{test['dataset']} | {test['comparison']} | `{test['metric']}` | "
            f"{mean_text} | "
            f"+{sign['positive_count']}/-{sign['negative_count']} of n={sign['n_nonzero']} | "
            f"{p_text} |"
        )
    lines.extend(
        [
            "",
            "With five seeds, the smallest possible nonzero two-sided exact sign-test p-value is 0.0625.",
            "These tests are therefore integrity checks on directionality, not claims of conventional p < 0.05 significance.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    tests: list[dict[str, object]] = []
    tests.extend(
        paired_method_tests(
            path=ARTIFACT_DIR / "waterbirds_official_baseline_per_seed.csv",
            dataset="waterbirds",
            method_a="VERA_selected",
            method_b="group_reweighted_erm",
            metrics=[
                "external_target_balanced_accuracy",
                "external_worst_target_source_accuracy",
                "external_source_leakage_balanced_accuracy",
            ],
        )
    )
    tests.extend(
        paired_method_tests(
            path=ARTIFACT_DIR / "camelyon17_wilds_official_multiseed_results.csv",
            dataset="camelyon17",
            method_a="VERA_selected",
            method_b="group_dro_probe",
            metrics=[
                "external_target_balanced_accuracy",
                "external_worst_target_source_accuracy",
                "validation_source_leakage_balanced_accuracy",
            ],
        )
    )
    tests.extend(mance_before_after_tests(ARTIFACT_DIR / "mance_reference_waterbirds_per_seed.csv"))
    grouped: dict[str, int] = defaultdict(int)
    for test in tests:
        grouped[str(test["dataset"])] += 1
    report = {
        "name": "VERA statistical significance addendum",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_family": "exact paired sign tests",
        "claim_boundary": (
            "Five-seed exact sign tests are directionality checks; they do not by themselves "
            "establish p < 0.05 significance because the minimum possible two-sided p-value is 0.0625."
        ),
        "dataset_test_counts": dict(sorted(grouped.items())),
        "tests": tests,
        "significance_addendum_ready": True,
    }
    DEFAULT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(DEFAULT_MD, report)
    print(json.dumps({"report": str(DEFAULT_JSON), "test_count": len(tests)}, indent=2))


if __name__ == "__main__":
    main()
