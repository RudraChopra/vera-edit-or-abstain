"""Audit the submission-facing VERA paper layout, naming, and PDF metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from pypdf.generic import DictionaryObject


ROOT = Path(__file__).resolve().parents[1]
AUTHOR_KIT = ROOT / "maintrack" / "aaai2027_template" / "AuthorKit27"
DEFAULT_ANONYMOUS = AUTHOR_KIT / "vera_aaai2027_anonymous.pdf"
DEFAULT_NAMED = AUTHOR_KIT / "vera_aaai2027_named.pdf"
DEFAULT_AUX = AUTHOR_KIT / "vera_aaai2027_anonymous.aux"
DEFAULT_FIGURE = ROOT / "maintrack" / "figures" / "vera_method_overview.pdf"
DEFAULT_REFERENCES = ROOT / "artifacts" / "reference_verification_report.json"
DEFAULT_VISUAL = ROOT / "artifacts" / "vera_figure1_visual_audit.json"
DEFAULT_RESULTS = ROOT / "artifacts" / "vera_confirmatory_results_package_audit.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "presentation_readiness_audit.json"

ACTIVE_SOURCES = (
    AUTHOR_KIT / "vera_aaai2027_anonymous.tex",
    AUTHOR_KIT / "vera_aaai2027_named.tex",
    AUTHOR_KIT / "vera_paper_body.tex",
    AUTHOR_KIT / "vera_supplement_body.tex",
    AUTHOR_KIT / "vera_results_macros.tex",
    AUTHOR_KIT / "vera_main_results_narrative.tex",
    AUTHOR_KIT / "vera_main_results_table.tex",
    AUTHOR_KIT / "vera_family_grid_results.tex",
    ROOT / "maintrack" / "references_verified.bib",
    ROOT / "maintrack" / "figures" / "vera_method_overview_caption.md",
)
NAMING_EXTENSIONS = {
    ".bib",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
NAMING_SCAN_LIMIT_BYTES = 10 * 1024 * 1024


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


def pdf_text(reader: PdfReader) -> str:
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def compact_alphanumeric(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def image_xobject_count(reader: PdfReader) -> int:
    count = 0
    seen: set[int] = set()

    def visit(value: Any) -> None:
        nonlocal count
        try:
            resolved = value.get_object()
        except AttributeError:
            resolved = value
        identity = id(resolved)
        if identity in seen:
            return
        seen.add(identity)
        if not isinstance(resolved, DictionaryObject):
            return
        subtype = str(resolved.get("/Subtype", ""))
        if subtype == "/Image":
            count += 1
        resources = resolved.get("/Resources")
        if resources is not None:
            visit(resources)
        xobjects = resolved.get("/XObject")
        if xobjects is not None:
            try:
                for child in xobjects.get_object().values():
                    visit(child)
            except AttributeError:
                pass

    for page in reader.pages:
        visit(page.get("/Resources", {}))
    return count


def parse_content_page(aux: Path, reader: PdfReader | None) -> int | None:
    if reader is not None:
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if re.search(r"(?:^|\n)References(?:\n|$)", text):
                return page_number - 1
    if not aux.is_file():
        return None
    match = re.search(
        r"\\newlabel\{vera-last-content-page\}\{\{[^}]*\}\{(\d+)\}",
        aux.read_text(encoding="utf-8", errors="replace"),
    )
    return None if match is None else int(match.group(1))


def source_audit() -> tuple[int, list[str], list[str]]:
    forbidden_patterns = ("faro", "vytallink", "rad shield")
    formatting_patterns = (
        r"\vspace{-",
        r"\enlargethispage",
        r"\geometry{",
        r"\addtolength{\text",
        r"\setlength{\textwidth",
        r"\setlength{\textheight",
        r"\setlength{\oddsidemargin",
    )
    forbidden_hits: list[str] = []
    formatting_hits: list[str] = []
    for path in ACTIVE_SOURCES:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        lowered = text.lower()
        for pattern in forbidden_patterns:
            if pattern in lowered:
                forbidden_hits.append(f"{path.relative_to(ROOT)}:{pattern}")
        for pattern in formatting_patterns:
            if pattern in text:
                formatting_hits.append(f"{path.relative_to(ROOT)}:{pattern}")
    return len(forbidden_hits), forbidden_hits, formatting_hits


def repository_naming_audit() -> tuple[int, list[str], list[str]]:
    """Scan the literal requested source/script/figure/artifact naming scope."""
    forbidden_patterns = ("faro", "vytallink", "rad shield")
    hits: list[str] = []
    oversized: list[str] = []
    for base in (ROOT / "maintrack", ROOT / "scripts", ROOT / "artifacts"):
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(ROOT).as_posix()
            lowered_path = relative.lower()
            for pattern in forbidden_patterns:
                if pattern in lowered_path:
                    hits.append(f"path:{relative}:{pattern}")
            if path.suffix.lower() not in NAMING_EXTENSIONS:
                continue
            if path.stat().st_size > NAMING_SCAN_LIMIT_BYTES:
                oversized.append(relative)
                continue
            lowered = path.read_text(encoding="utf-8", errors="replace").lower()
            for pattern in forbidden_patterns:
                count = lowered.count(pattern)
                if count:
                    hits.append(f"content:{relative}:{pattern}:{count}")
    return len(hits), hits[:200], oversized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anonymous", type=Path, default=DEFAULT_ANONYMOUS)
    parser.add_argument("--named", type=Path, default=DEFAULT_NAMED)
    parser.add_argument("--aux", type=Path, default=DEFAULT_AUX)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--visual-audit", type=Path, default=DEFAULT_VISUAL)
    parser.add_argument("--results-audit", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    references = load_json(args.references)
    visual = load_json(args.visual_audit)
    results = load_json(args.results_audit)
    missing = [
        str(path)
        for path in (args.anonymous, args.named, args.figure)
        if not path.is_file()
    ]
    anonymous_reader = PdfReader(str(args.anonymous)) if args.anonymous.is_file() else None
    named_reader = PdfReader(str(args.named)) if args.named.is_file() else None
    figure_reader = PdfReader(str(args.figure)) if args.figure.is_file() else None
    anonymous_text = "" if anonymous_reader is None else pdf_text(anonymous_reader)
    named_text = "" if named_reader is None else pdf_text(named_reader)
    anonymous_metadata = (
        {} if anonymous_reader is None else dict(anonymous_reader.metadata or {})
    )
    named_metadata = {} if named_reader is None else dict(named_reader.metadata or {})
    content_page_count = parse_content_page(args.aux, anonymous_reader)
    forbidden_count, forbidden_hits, formatting_hits = source_audit()
    repository_forbidden_count, repository_forbidden_hits, naming_unscanned = (
        repository_naming_audit()
    )
    forbidden_pdf = ("faro", "vytallink", "rad shield")
    anonymous_lower = anonymous_text.lower()
    named_lower = named_text.lower()
    disclosure_phrase = "openai codex was used extensively"
    source_disclosure_present = (
        AUTHOR_KIT / "vera_paper_body.tex"
    ).read_text(encoding="utf-8").lower().count(disclosure_phrase) == 1
    compact_disclosure = compact_alphanumeric(disclosure_phrase)
    ai_assistance_disclosed = source_disclosure_present and all(
        compact_disclosure in compact_alphanumeric(text)
        for text in (anonymous_text, named_text)
    )
    anonymous_pdf_clean = (
        bool(anonymous_text)
        and "rudra chopra" not in anonymous_lower
        and "contra costa" not in anonymous_lower
        and "github.com/rudrachopra" not in anonymous_lower
        and not any(value in anonymous_lower for value in forbidden_pdf)
    )
    named_pdf_clean = (
        "rudra chopra" in named_lower
        and not any(value in named_lower for value in forbidden_pdf)
    )
    metadata_text = json.dumps(
        {"anonymous": anonymous_metadata, "named": named_metadata},
        sort_keys=True,
        default=str,
    ).lower()
    metadata_clean = (
        not any(value in metadata_text for value in forbidden_pdf)
        and "rudra chopra" not in json.dumps(
            anonymous_metadata, sort_keys=True, default=str
        ).lower()
    )
    figure_image_count = (
        -1 if figure_reader is None else image_xobject_count(figure_reader)
    )
    figure_vector = figure_image_count == 0
    figure_panel_count = int(visual.get("panel_count", 0))
    visual_hash_ok = (
        args.figure.is_file()
        and visual.get("pdf_sha256") == sha256(args.figure)
    )
    exact_page_limit = content_page_count == 7
    total_page_limit = (
        anonymous_reader is not None and len(anonymous_reader.pages) <= 9
    )
    verified_references = int(references.get("verified_reference_count", 0))
    passed = (
        not missing
        and exact_page_limit
        and total_page_limit
        and references.get("passed") is True
        and verified_references >= 40
        and figure_vector
        and figure_panel_count == 3
        and visual_hash_ok
        and visual.get("colorblind_safe_palette") is True
        and visual.get("text_readable_at_half_scale") is True
        and visual.get("abstract_and_figure_teach_the_claim") is True
        and forbidden_count == 0
        and not formatting_hits
        and anonymous_pdf_clean
        and named_pdf_clean
        and metadata_clean
        and ai_assistance_disclosed
        and results.get("passed") is True
    )
    report = {
        "name": "VERA presentation readiness audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "missing_files": missing,
        "content_page_count": content_page_count,
        "anonymous_total_pages": (
            None if anonymous_reader is None else len(anonymous_reader.pages)
        ),
        "named_total_pages": (
            None if named_reader is None else len(named_reader.pages)
        ),
        "exact_page_limit": exact_page_limit,
        "total_page_limit": total_page_limit,
        "verified_reference_count": verified_references,
        "reference_audit_passed": references.get("passed") is True,
        "figure_1_vector": figure_vector,
        "figure_1_panel_count": figure_panel_count,
        "figure_1_image_xobject_count": figure_image_count,
        "figure_1_colorblind_safe": visual.get("colorblind_safe_palette") is True,
        "figure_1_readable_at_half_scale": visual.get(
            "text_readable_at_half_scale"
        )
        is True,
        "abstract_figure1_sufficiency_reviewed": visual.get(
            "abstract_and_figure_teach_the_claim"
        )
        is True,
        "figure_visual_audit_hash_matches": visual_hash_ok,
        "forbidden_name_hit_count": forbidden_count,
        "forbidden_name_hits": forbidden_hits,
        "repository_forbidden_name_hit_count": repository_forbidden_count,
        "repository_forbidden_name_hits_sample": repository_forbidden_hits,
        "repository_naming_oversized_unscanned": naming_unscanned,
        "formatting_hack_hits": formatting_hits,
        "anonymous_pdf_clean": anonymous_pdf_clean,
        "named_pdf_clean": named_pdf_clean,
        "pdf_metadata_clean": metadata_clean,
        "ai_assistance_disclosure_present": ai_assistance_disclosed,
        "anonymous_pdf_sha256": (
            None if not args.anonymous.is_file() else sha256(args.anonymous)
        ),
        "named_pdf_sha256": (
            None if not args.named.is_file() else sha256(args.named)
        ),
        "figure_1_sha256": (
            None if not args.figure.is_file() else sha256(args.figure)
        ),
        "results_package_passed": results.get("passed") is True,
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
