from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "maintrack"
    / "mosaic_aaai2027"
    / "make_mosaic_final_evidence_figure.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("mosaic_final_figure", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_final_figure_requires_hash_matched_passing_audits(tmp_path, monkeypatch):
    module = load_module()
    baseline_report = tmp_path / "baseline.json"
    baseline_audit = tmp_path / "baseline_audit.json"
    baseline_summary = tmp_path / "baseline_summary.json"
    misspec_report = tmp_path / "misspec.json"
    misspec_audit = tmp_path / "misspec_audit.json"
    real_summary = tmp_path / "real_summary.json"
    correction_audit = tmp_path / "correction_audit.json"
    write_json(baseline_report, {"name": "baseline"})
    write_json(baseline_audit, {"pass": True})
    write_json(
        baseline_summary,
        {
            "audit_status": "development_replay_not_independent_human_review",
            "audit_sha256": module.sha256(baseline_audit),
            "report_sha256": module.sha256(baseline_report),
        },
    )
    write_json(misspec_report, {"name": "misspec"})
    write_json(
        misspec_audit,
        {"pass": True, "report_sha256": module.sha256(misspec_report)},
    )
    audit_receipts = {}
    for name in ("strict", "rational", "comparator"):
        path = tmp_path / f"{name}_audit.json"
        write_json(path, {"passed": True})
        audit_receipts[name] = {"path": str(path), "sha256": module.sha256(path)}
    write_json(
        real_summary,
        {
            "status": "complete",
            "strict_receipt_count": 100,
            "comparator_receipt_count": 100,
            "audit_receipts": audit_receipts,
        },
    )
    write_json(correction_audit, {"passed": True, "files_compared": 100})
    monkeypatch.setattr(module, "BASELINE_REPORT", baseline_report)
    monkeypatch.setattr(module, "BASELINE_AUDIT", baseline_audit)
    monkeypatch.setattr(module, "BASELINE_SUMMARY", baseline_summary)
    monkeypatch.setattr(module, "MISSPEC_REPORT", misspec_report)
    monkeypatch.setattr(module, "MISSPEC_AUDIT", misspec_audit)
    monkeypatch.setattr(module, "REAL_SUMMARY", real_summary)
    monkeypatch.setattr(module, "CORRECTION_AUDIT", correction_audit)
    baseline, misspec, real = module.verify()
    assert baseline["audit_sha256"] == module.sha256(baseline_audit)
    assert misspec["name"] == "misspec"
    assert real["strict_receipt_count"] == 100

    write_json(tmp_path / "strict_audit.json", {"passed": False})
    try:
        module.verify()
    except AssertionError:
        pass
    else:
        raise AssertionError("a failed real-evidence audit was accepted")
