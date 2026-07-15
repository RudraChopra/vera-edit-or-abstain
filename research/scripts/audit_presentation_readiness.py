"""Audit the submission-facing VERA paper layout, naming, and PDF metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from pypdf.generic import DictionaryObject


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
AUTHOR_KIT = ROOT / "maintrack" / "aaai2027_template" / "AuthorKit27"
DEFAULT_ANONYMOUS = AUTHOR_KIT / "vera_aaai2027_anonymous.pdf"
DEFAULT_NAMED = AUTHOR_KIT / "vera_aaai2027_named.pdf"
DEFAULT_AUX = AUTHOR_KIT / "vera_aaai2027_anonymous.aux"
DEFAULT_FIGURE = ROOT / "maintrack" / "figures" / "vera_method_overview.pdf"
DEFAULT_ANONYMOUS_ARCHIVE = REPOSITORY / "dist" / "vera_anonymous_submission.zip"
DEFAULT_REFERENCES = ROOT / "artifacts" / "reference_verification_report.json"
DEFAULT_VISUAL = ROOT / "artifacts" / "vera_figure1_visual_audit.json"
DEFAULT_RESULTS = ROOT / "artifacts" / "vera_confirmatory_results_package_audit.json"
DEFAULT_INDEPENDENT_RESULTS = (
    ROOT / "artifacts" / "vera_independent_stress_package_audit.json"
)
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
NAMING_SCAN_LIMIT_BYTES = 128 * 1024 * 1024


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


def normalized_naming_text(text: str) -> str:
    lowered = text.lower()
    lowered = lowered.replace("/volumes/backups/faro", "<external-storage>")
    lowered = lowered.replace("/tmp/faro-torch-venv", "<runtime-venv>")
    return lowered


def tracked_naming_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "research/maintrack", "research/scripts", "research/artifacts"],
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        paths: list[Path] = []
        for base in (ROOT / "maintrack", ROOT / "scripts", ROOT / "artifacts"):
            if base.is_dir():
                paths.extend(path for path in base.rglob("*") if path.is_file())
        return sorted(paths)
    return [
        REPOSITORY / line
        for line in result.stdout.splitlines()
        if line.startswith("research/")
    ]


def active_archive_paths(archive_path: Path) -> set[str]:
    if not archive_path.is_file():
        return set()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            manifest = json.loads(archive.read("MANIFEST.json"))
    except (OSError, KeyError, json.JSONDecodeError, zipfile.BadZipFile):
        return set()
    entries = manifest.get("entries", [])
    if not isinstance(entries, list):
        return set()
    paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        path = str(entry.get("path", ""))
        if path.startswith("research/"):
            try:
                paths.add(Path(path).relative_to("research").as_posix())
            except ValueError:
                continue
    return paths


def repository_naming_audit(
    archive_path: Path,
) -> tuple[int, list[str], list[str], int, list[str], int, int]:
    """Scan tracked files and separate active archive hits from historical files."""
    forbidden_patterns = ("faro", "vytallink", "rad shield")
    active_paths = active_archive_paths(archive_path)
    active_hits: list[str] = []
    historical_hits: list[str] = []
    oversized: list[str] = []
    scanned = 0
    for path in tracked_naming_files():
        if not path.is_file() or path.is_symlink():
            continue
        try:
            relative = path.relative_to(ROOT).as_posix()
        except ValueError:
            continue
        destination = active_hits if relative in active_paths else historical_hits
        lowered_path = normalized_naming_text(relative)
        for pattern in forbidden_patterns:
            if pattern in lowered_path:
                destination.append(f"path:{relative}:{pattern}")
        if path.suffix.lower() not in NAMING_EXTENSIONS:
            continue
        if path.stat().st_size > NAMING_SCAN_LIMIT_BYTES:
            oversized.append(relative)
            continue
        scanned += 1
        lowered = normalized_naming_text(path.read_text(encoding="utf-8", errors="replace"))
        for pattern in forbidden_patterns:
            count = lowered.count(pattern)
            if count:
                destination.append(f"content:{relative}:{pattern}:{count}")
    return (
        len(active_hits),
        active_hits[:200],
        oversized,
        len(historical_hits),
        historical_hits[:200],
        len(active_paths),
        scanned,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anonymous", type=Path, default=DEFAULT_ANONYMOUS)
    parser.add_argument("--named", type=Path, default=DEFAULT_NAMED)
    parser.add_argument("--aux", type=Path, default=DEFAULT_AUX)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--anonymous-archive", type=Path, default=DEFAULT_ANONYMOUS_ARCHIVE)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--visual-audit", type=Path, default=DEFAULT_VISUAL)
    parser.add_argument("--results-audit", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--independent-results-audit",
        type=Path,
        default=DEFAULT_INDEPENDENT_RESULTS,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    references = load_json(args.references)
    visual = load_json(args.visual_audit)
    results = load_json(args.results_audit)
    independent_results = load_json(args.independent_results_audit)
    active_results = independent_results if independent_results else results
    active_results_kind = "independent_stress" if independent_results else "confirmatory"
    active_macros_sha = (
        active_results.get("macros_sha256")
        or active_results.get("outputs", {}).get(
            "maintrack/aaai2027_template/AuthorKit27/vera_results_macros.tex"
        )
    )
    macros_hash_ok = (
        not active_macros_sha
        or (
            (AUTHOR_KIT / "vera_results_macros.tex").is_file()
            and sha256(AUTHOR_KIT / "vera_results_macros.tex") == active_macros_sha
        )
    )
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
    (
        repository_forbidden_count,
        repository_forbidden_hits,
        naming_unscanned,
        historical_forbidden_count,
        historical_forbidden_hits,
        active_archive_path_count,
        tracked_text_files_scanned,
    ) = repository_naming_audit(args.anonymous_archive)
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
        and active_results.get("passed") is True
        and macros_hash_ok
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
        "repository_historical_forbidden_name_hit_count": historical_forbidden_count,
        "repository_historical_forbidden_name_hits_sample": historical_forbidden_hits,
        "repository_active_archive_path_count": active_archive_path_count,
        "repository_tracked_text_files_scanned": tracked_text_files_scanned,
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
        "results_package_kind": active_results_kind,
        "results_package_passed": active_results.get("passed") is True,
        "results_package_macros_hash_matches": macros_hash_ok,
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
