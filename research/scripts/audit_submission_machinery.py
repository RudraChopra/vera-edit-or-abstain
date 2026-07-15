"""Fail-closed audit of technical and human-confirmed submission machinery."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUTHOR_KIT = ROOT / "maintrack" / "aaai2027_template" / "AuthorKit27"
DEFAULT_REGISTRY = ROOT / "private" / "submission_registry.json"
DEFAULT_PRESENTATION = ROOT / "artifacts" / "presentation_readiness_audit.json"
DEFAULT_ARCHIVE = ROOT / "artifacts" / "vera_anonymous_archive_audit.json"
DEFAULT_REPRODUCTION = ROOT / "artifacts" / "vera_one_command_reproduction.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "submission_machinery_audit.json"
CHECKLIST = ROOT / "maintrack" / "REPRODUCIBILITY_CHECKLIST.md"
OFFICIAL_CHECKLIST_TEX = AUTHOR_KIT / "ReproducibilityChecklist.tex"
OFFICIAL_CHECKLIST_PDF = AUTHOR_KIT / "ReproducibilityChecklist.pdf"
ANONYMOUS_MAIN = AUTHOR_KIT / "vera_aaai2027_anonymous.pdf"
ANONYMOUS_SUPPLEMENT = AUTHOR_KIT / "vera_aaai2027_supplement_anonymous.pdf"
SUPPLEMENT_SOURCE = AUTHOR_KIT / "vera_supplement_body.tex"
PREREGISTRATIONS = (
    ROOT / "prereg_confirmatory_balanced.json",
    ROOT / "prereg_confirmatory_secondary_ablations.json",
    ROOT / "prereg_real_learning_curve_diagnostic.json",
    ROOT / "prereg_exact_family_grid.json",
    ROOT / "prereg_independent_stress_replication.json",
)
REGISTERED_AAAI_TOPICS = {
    "ML: Evaluation, Benchmarking, Datasets & Analysis",
    "ML: Machine Unlearning, Data Deletion & Model Editing",
    "ML: Adversarial Learning & Robustness",
    "ML: Representation Learning",
    "PEAI: AI Evaluation, Auditing & Red Teaming",
    "RU: Uncertainty Representations",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--presentation", type=Path, default=DEFAULT_PRESENTATION)
    parser.add_argument("--archive-audit", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--reproduction", type=Path, default=DEFAULT_REPRODUCTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = load_json(args.registry)
    presentation = load_json(args.presentation)
    archive = load_json(args.archive_audit)
    reproduction = load_json(args.reproduction)
    checklist_text = (
        CHECKLIST.read_text(encoding="utf-8") if CHECKLIST.is_file() else ""
    )
    official_checklist_text = (
        OFFICIAL_CHECKLIST_TEX.read_text(encoding="utf-8")
        if OFFICIAL_CHECKLIST_TEX.is_file()
        else ""
    )
    official_responses = (
        official_checklist_text.split("% The questions start here", 1)[-1]
        if "% The questions start here" in official_checklist_text
        else ""
    )
    supplement_source = (
        SUPPLEMENT_SOURCE.read_text(encoding="utf-8")
        if SUPPLEMENT_SOURCE.is_file()
        else ""
    )
    prereg_hash_status: dict[str, dict[str, Any]] = {}
    for prereg in PREREGISTRATIONS:
        sidecar = prereg.with_suffix(".sha256")
        actual = sha256(prereg) if prereg.is_file() else None
        sidecar_hash = (
            sidecar.read_text(encoding="utf-8").split()[0]
            if sidecar.is_file() and sidecar.read_text(encoding="utf-8").split()
            else None
        )
        source_has_hash = bool(
            actual
            and actual[:32] in supplement_source
            and actual[32:] in supplement_source
        )
        prereg_hash_status[prereg.name] = {
            "actual_sha256": actual,
            "sidecar_sha256": sidecar_hash,
            "sidecar_matches": actual is not None and actual == sidecar_hash,
            "supplement_displays_hash": source_has_hash,
        }
    prereg_hashes_match = all(
        item["sidecar_matches"] and item["supplement_displays_hash"]
        for item in prereg_hash_status.values()
    )
    stale_checklist_terms = [
        term
        for term in (
            "faro",
            "proxy rows until",
            "official benchmark rows must use seeds `0, 1, 2, 3, 4`",
            "faro_main.tex",
        )
        if term in checklist_text.lower()
    ]
    checklist_complete = (
        bool(checklist_text)
        and "- [ ]" not in checklist_text
        and "TODO" not in checklist_text
        and "pending" not in checklist_text.lower()
        and not stale_checklist_terms
        and bool(official_responses)
        and "Type your response here" not in official_responses
        and official_responses.count("\\question{") == 31
        and OFFICIAL_CHECKLIST_PDF.is_file()
        and OFFICIAL_CHECKLIST_PDF.stat().st_mtime
        >= OFFICIAL_CHECKLIST_TEX.stat().st_mtime
    )
    keywords = registry.get("keywords", [])
    deadline_urls = registry.get("deadline_source_urls", [])
    technical = {
        "target_style_compiles": presentation.get("passed") is True,
        "exact_page_limit": presentation.get("exact_page_limit") is True,
        "zero_formatting_hacks": presentation.get("formatting_hack_hits") == [],
        "anonymization_complete": presentation.get("anonymous_pdf_clean") is True
        and presentation.get("pdf_metadata_clean") is True,
        "anonymous_archive_reproduces_main_table": archive.get("passed") is True
        and reproduction.get("passed") is True
        and reproduction.get("mode") == "compact_frozen_rows",
        "reproducibility_checklist_complete": checklist_complete,
        "manuscript_prereg_hashes_match": prereg_hashes_match,
        "supplement_ready": ANONYMOUS_SUPPLEMENT.is_file()
        and ANONYMOUS_MAIN.is_file()
        and presentation.get("passed") is True,
        "areas_and_keywords_selected": isinstance(keywords, list)
        and 3 <= len(keywords) <= 5
        and str(registry.get("primary_area", ""))
        in REGISTERED_AAAI_TOPICS
        and set(map(str, keywords)).issubset(REGISTERED_AAAI_TOPICS)
        and str(registry.get("primary_area", "")) not in set(map(str, keywords)),
    }
    human = {
        "openreview_account_human_confirmed": registry.get(
            "openreview_account_human_confirmed"
        )
        is True,
        "single_email_human_confirmed": registry.get(
            "single_email_human_confirmed"
        )
        is True,
        "deadlines_human_confirmed": registry.get("deadlines_human_confirmed")
        is True
        and isinstance(deadline_urls, list)
        and len(deadline_urls) >= 2,
        "scientific_content_human_verified": registry.get(
            "scientific_content_human_verified"
        )
        is True,
        "authorship_criteria_human_confirmed": registry.get(
            "authorship_criteria_human_confirmed"
        )
        is True,
        "ai_assistance_disclosure_human_confirmed": registry.get(
            "ai_assistance_disclosure_human_confirmed"
        )
        is True,
    }
    passed = all(technical.values()) and all(human.values())
    report = {
        "name": "VERA submission machinery audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        **technical,
        **human,
        "registry_present": bool(registry),
        "registry_sha256": sha256(args.registry) if args.registry.is_file() else None,
        "presentation_audit_sha256": (
            sha256(args.presentation) if args.presentation.is_file() else None
        ),
        "archive_audit_sha256": (
            sha256(args.archive_audit) if args.archive_audit.is_file() else None
        ),
        "reproduction_audit_sha256": (
            sha256(args.reproduction) if args.reproduction.is_file() else None
        ),
        "primary_area": registry.get("primary_area"),
        "keywords": keywords,
        "deadline_source_urls": deadline_urls,
        "official_checklist_question_count": official_responses.count(
            "\\question{"
        ),
        "official_checklist_unanswered_count": official_responses.count(
            "Type your response here"
        ),
        "stale_checklist_terms": stale_checklist_terms,
        "preregistration_hashes": prereg_hash_status,
        "technical_failures": [
            key for key, value in technical.items() if not value
        ],
        "human_confirmation_failures": [
            key for key, value in human.items() if not value
        ],
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
