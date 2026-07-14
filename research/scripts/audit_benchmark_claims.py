"""Audit which FARO benchmark artifacts can support paper claims.

This script is intentionally conservative. A row is claim-ready only when the
receipt and statistics are materialized local files, the receipt explicitly
passes the claim gate, and the statistics report is claim-grade. A dataless
iCloud/File Provider placeholder is treated as missing evidence.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
PAPER_DIR = ROOT / "paper"
JSON_OUT = ARTIFACT_DIR / "benchmark_claim_audit.json"
CSV_OUT = ARTIFACT_DIR / "benchmark_claim_audit.csv"
TABLE_OUT = PAPER_DIR / "benchmark_claim_audit_table.tex"


@dataclass(frozen=True)
class ClaimRow:
    key: str
    benchmark: str
    family: str
    artifact: str
    artifact_status: str
    evidence_level: str
    gate_score: str
    claim_ready: str
    official_dataset: str
    official_splits: str
    real_images_or_samples: str
    frozen_or_deep_embeddings: str
    multi_seed: str
    strong_baselines: str
    worst_group_or_domain_metric: str
    failure_or_abstention_analysis: str
    allowed_claim: str
    next_gate: str


def bool_word(value: bool) -> str:
    return "yes" if value else "no"


def is_materialized_file(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    stat = path.stat()
    return not (stat.st_size > 0 and getattr(stat, "st_blocks", 1) == 0)


def artifact_status(path: Path) -> str:
    if not path.exists():
        return "missing"
    if path.is_file() and not is_materialized_file(path):
        return "dataless_placeholder"
    return "present"


def load_json(path: Path) -> dict[str, Any]:
    if not is_materialized_file(path):
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def receipt_claim_ready(receipt_path: Path, statistics_path: Path) -> bool:
    receipt = load_json(receipt_path)
    statistics = load_json(statistics_path)
    decision = statistics.get("decision", {})
    decision = decision if isinstance(decision, dict) else {}
    stats_ready = (
        bool(statistics.get("claim_grade_statistics")) is True
        or bool(decision.get("conference_claim_ready")) is True
    )
    stats_claim_gate = (
        bool(statistics.get("claim_gate_passed")) is True
        or bool(statistics.get("receipt_claim_gate_passed")) is True
        or bool(decision.get("claim_gate_passed")) is True
    )
    return (
        bool(receipt.get("claim_gate_passed")) is True
        and stats_claim_gate
        and stats_ready
    )


def score(flags: list[bool]) -> str:
    return f"{sum(flags)}/{len(flags)}"


def claim_row_for_official_receipt(
    *,
    key: str,
    benchmark: str,
    family: str,
    receipt_path: Path,
    statistics_path: Path,
    allowed_ready_claim: str,
    next_gate: str,
) -> ClaimRow:
    receipt = load_json(receipt_path)
    statistics = load_json(statistics_path)
    confirmations = receipt.get("claim_confirmations", {})
    confirmations = confirmations if isinstance(confirmations, dict) else {}
    claim_gates = receipt.get("claim_gates", {})
    claim_gates = claim_gates if isinstance(claim_gates, dict) else {}
    provenance = receipt.get("provenance_assessment", {})
    provenance = provenance if isinstance(provenance, dict) else {}
    full_dataset_export = bool(receipt.get("full_dataset_export")) is True
    if not full_dataset_export:
        full_dataset_export = bool(provenance.get("full_dataset_export")) is True
    claim_ready = receipt_claim_ready(receipt_path, statistics_path)
    official_data = (
        bool(confirmations.get("official_data_confirmed")) is True
        or bool(claim_gates.get("official_dataset")) is True
        or bool(receipt.get("official_dataset")) is True
    )
    official_splits = (
        bool(confirmations.get("official_splits_confirmed")) is True
        or bool(claim_gates.get("official_splits")) is True
        or bool(receipt.get("official_splits")) is True
    )
    real_samples = (
        bool(confirmations.get("real_samples_confirmed")) is True
        or bool(claim_gates.get("real_images_or_samples")) is True
        or bool(receipt.get("real_images_or_samples")) is True
        or (official_data and full_dataset_export)
    )
    embeddings = (
        bool(confirmations.get("claim_grade_embeddings_confirmed")) is True
        or bool(claim_gates.get("frozen_or_deep_embeddings")) is True
        or bool(receipt.get("frozen_or_deep_embeddings")) is True
        or int(receipt.get("feature_count", 0)) > 0
    )
    seeds = receipt.get("seeds", [])
    multi_seed = isinstance(seeds, list) and len(seeds) >= 5
    decision = statistics.get("decision", {})
    decision = decision if isinstance(decision, dict) else {}
    stats_ready = (
        bool(statistics.get("claim_grade_statistics")) is True
        or bool(decision.get("conference_claim_ready")) is True
    )
    strong_baselines = stats_ready or bool(claim_gates.get("strong_baselines")) is True
    strong_baselines = strong_baselines or bool(receipt.get("strong_baselines")) is True
    worst_group = (
        stats_ready or bool(claim_gates.get("worst_group_or_domain_metric")) is True
    )
    worst_group = worst_group or bool(receipt.get("worst_group_or_domain_metric")) is True
    failure_analysis = (
        stats_ready or bool(claim_gates.get("failure_or_abstention_analysis")) is True
    )
    failure_analysis = failure_analysis or bool(receipt.get("failure_or_abstention_analysis")) is True
    flags = [
        official_data,
        official_splits,
        real_samples,
        embeddings,
        multi_seed,
        strong_baselines,
        full_dataset_export,
        claim_ready,
    ]
    status = artifact_status(receipt_path)
    if status == "present" and artifact_status(statistics_path) != "present":
        status = artifact_status(statistics_path)
    return ClaimRow(
        key=key,
        benchmark=benchmark,
        family=family,
        artifact=str(receipt_path.relative_to(ROOT)),
        artifact_status=status,
        evidence_level="official receipt with paired statistics" if claim_ready else "incomplete official receipt",
        gate_score=score(flags),
        claim_ready=bool_word(claim_ready),
        official_dataset=bool_word(official_data),
        official_splits=bool_word(official_splits),
        real_images_or_samples=bool_word(real_samples),
        frozen_or_deep_embeddings=bool_word(embeddings),
        multi_seed=bool_word(multi_seed),
        strong_baselines=bool_word(strong_baselines),
        worst_group_or_domain_metric=bool_word(worst_group),
        failure_or_abstention_analysis=bool_word(failure_analysis),
        allowed_claim=allowed_ready_claim if claim_ready else "no claim-grade benchmark result yet",
        next_gate=next_gate if not claim_ready else "maintain claim boundary and replicate in another family",
    )


def claim_row_for_camelyon_store() -> ClaimRow:
    report_path = ARTIFACT_DIR / "camelyon17_resnet18_torch_full_store_report.json"
    report = load_json(report_path)
    sample_count = int(report.get("sample_count", 0))
    expected = int(report.get("expected_full_dataset_examples", 455_954))
    embedding_ready = (
        artifact_status(report_path) == "present"
        and bool(report.get("claim_grade_embedding_store")) is True
        and bool(report.get("binary_size_matches")) is True
        and int(report.get("feature_count", 0)) == 512
    )
    full_store = sample_count == expected and bool(report.get("full_dataset_embedding_export")) is True
    flags = [
        True,
        True,
        sample_count > 0,
        embedding_ready,
        False,
        False,
        full_store,
        False,
    ]
    evidence_level = (
        f"complete frozen embedding store ({sample_count}/{expected} rows)"
        if full_store
        else f"partial frozen embedding store ({sample_count}/{expected} rows)"
    )
    allowed_claim = (
        "complete frozen embedding store only; not a benchmark result"
        if full_store
        else "embedding-store progress only; not a benchmark result"
    )
    next_gate = (
        "convert full store to NumPy store, then run five-seed receipt/statistics"
        if full_store
        else "complete 455,954-row store, convert to NumPy store, then run five-seed receipt/statistics"
    )
    return ClaimRow(
        key="camelyon17_tar_stream_store",
        benchmark="Camelyon17-WILDS tar-stream store",
        family="medical / hospital shift",
        artifact=str(report_path.relative_to(ROOT)),
        artifact_status=artifact_status(report_path),
        evidence_level=evidence_level,
        gate_score=score(flags),
        claim_ready="no",
        official_dataset="yes",
        official_splits="yes",
        real_images_or_samples=bool_word(sample_count > 0),
        frozen_or_deep_embeddings=bool_word(embedding_ready),
        multi_seed="no",
        strong_baselines="no",
        worst_group_or_domain_metric="no",
        failure_or_abstention_analysis="no",
        allowed_claim=allowed_claim,
        next_gate=next_gate,
    )


def static_development_rows() -> list[ClaimRow]:
    specs = [
        (
            "synthetic_abstention",
            "Synthetic FARO abstention certificate",
            "controlled latent",
            ARTIFACT_DIR / "faro_synthetic_abstention_report.json",
            "mechanism and abstention certificate",
            [False, False, False, False, True, False, False, False],
            "supports algorithmic mechanism only",
            "pair with official benchmark evidence and real abstention stress tests",
        ),
        (
            "colored_mnist_frontier",
            "Colored-MNIST frontier",
            "controlled image shift",
            ARTIFACT_DIR / "colored_mnist_frontier_summary.csv",
            "frontier diagnostic",
            [False, False, True, True, True, False, False, True],
            "supports controlled-shift behavior only",
            "repeat frontier diagnostics on official benchmark families",
        ),
    ]
    rows: list[ClaimRow] = []
    for key, benchmark, family, artifact, evidence, flags, allowed, next_gate in specs:
        rows.append(
            ClaimRow(
                key=key,
                benchmark=benchmark,
                family=family,
                artifact=str(artifact.relative_to(ROOT)),
                artifact_status=artifact_status(artifact),
                evidence_level=evidence,
                gate_score=score(flags),
                claim_ready="no",
                official_dataset=bool_word(flags[0]),
                official_splits=bool_word(flags[1]),
                real_images_or_samples=bool_word(flags[2]),
                frozen_or_deep_embeddings=bool_word(flags[3]),
                multi_seed=bool_word(flags[4]),
                strong_baselines=bool_word(flags[5]),
                worst_group_or_domain_metric=bool_word(flags[6]),
                failure_or_abstention_analysis=bool_word(flags[7]),
                allowed_claim=allowed,
                next_gate=next_gate,
            )
        )
    return rows


def build_rows() -> list[ClaimRow]:
    rows = static_development_rows()
    rows.append(
        claim_row_for_official_receipt(
            key="waterbirds_official",
            benchmark="Waterbirds official",
            family="spurious correlation / background shift",
            receipt_path=ARTIFACT_DIR / "waterbirds_official_result_receipt.json",
            statistics_path=ARTIFACT_DIR / "waterbirds_official_statistical_report.json",
            allowed_ready_claim="official frozen-embedding benchmark row; FARO abstains and is not a win",
            next_gate="keep as failure-analysis/abstention evidence, not as a positive result",
        )
    )
    rows.append(
        claim_row_for_official_receipt(
            key="civilcomments_full_store_balanced_trace",
            benchmark="CivilComments-WILDS full official store",
            family="text / subpopulation shift",
            receipt_path=ARTIFACT_DIR
            / "civilcomments_wilds_full_store_balanced_trace_store_result_receipt.json",
            statistics_path=ARTIFACT_DIR
            / "civilcomments_wilds_full_store_balanced_trace_store_statistical_report.json",
            allowed_ready_claim="claim-grade official frozen-embedding benchmark row",
            next_gate="rehydrate/regenerate receipt files and replicate in another family",
        )
    )
    rows.append(
        claim_row_for_official_receipt(
            key="camelyon17_wilds_official",
            benchmark="Camelyon17-WILDS official frozen-embedding",
            family="medical / hospital shift",
            receipt_path=ARTIFACT_DIR / "camelyon17_wilds_official_result_receipt.json",
            statistics_path=ARTIFACT_DIR / "camelyon17_wilds_official_statistical_report.json",
            allowed_ready_claim=(
                "official frozen-embedding high-stakes benchmark row with FARO "
                "abstention boundary; not clinical deployment evidence"
            ),
            next_gate="maintain claim boundary and replicate in another high-stakes family",
        )
    )
    rows.append(claim_row_for_camelyon_store())
    return rows


def write_csv(path: Path, rows: list[ClaimRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ClaimRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def write_table(path: Path, rows: list[ClaimRow]) -> None:
    display = [row for row in rows if row.key not in {"synthetic_abstention"}]
    lines = [
        "% Auto-generated by research/scripts/audit_benchmark_claims.py",
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{FARO benchmark claim audit. Claim-ready requires materialized "
            r"official receipts, paired statistics, and explicit claim gates.}"
        ),
        r"\label{tab:benchmark-claim-audit}",
        r"\begin{tabular}{llclp{5.0cm}}",
        r"\toprule",
        r"Benchmark & Evidence & Gates & Claim-ready & Next gate \\",
        r"\midrule",
    ]
    for row in display:
        lines.append(
            f"{latex_escape(row.benchmark)} & {latex_escape(row.evidence_level)} & "
            f"{latex_escape(row.gate_score)} & {latex_escape(row.claim_ready)} & "
            f"{latex_escape(row.next_gate)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    rows = build_rows()
    claim_ready_rows = [row for row in rows if row.claim_ready == "yes"]
    official_claim_ready_rows = [
        row
        for row in claim_ready_rows
        if row.official_dataset == "yes" and row.official_splits == "yes"
    ]
    report = {
        "name": "FARO benchmark claim audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_ready_count": len(claim_ready_rows),
        "official_claim_ready_count": len(official_claim_ready_rows),
        "submission_ready": len(official_claim_ready_rows) >= 2,
        "claim_boundary": (
            "Dataless placeholder files, partial embedding stores, dry runs, and smoke tests "
            "cannot support benchmark-result claims."
        ),
        "rows": [asdict(row) for row in rows],
    }
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(CSV_OUT, rows)
    write_table(TABLE_OUT, rows)
    print("FARO benchmark claim audit complete")
    print(f"submission_ready={str(report['submission_ready']).lower()}")
    print(f"official_claim_ready_count={report['official_claim_ready_count']}")
    print(f"report={JSON_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
