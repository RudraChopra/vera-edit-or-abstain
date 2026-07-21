#!/usr/bin/env python3
"""Fail closed on the MOSAIC V10 AAAI submission artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

import fitz


HERE = Path(__file__).resolve().parent
PDFS = {
    "main_anonymous": (HERE / "MOSAIC_AAAI2027_V10_ANONYMOUS.pdf", True, True),
    "main_named": (HERE / "MOSAIC_AAAI2027_V10_NAMED.pdf", True, False),
    "supplement_anonymous": (
        HERE / "MOSAIC_AAAI2027_V10_SUPPLEMENT_ANONYMOUS.pdf",
        False,
        True,
    ),
    "supplement_named": (
        HERE / "MOSAIC_AAAI2027_V10_SUPPLEMENT_NAMED.pdf",
        False,
        False,
    ),
    "checklist": (
        HERE / "MOSAIC_AAAI2027_V10_REPRODUCIBILITY_CHECKLIST.pdf",
        False,
        True,
    ),
}
LOGS = {
    "main_anonymous": HERE / "mosaic_aaai2027_anonymous.log",
    "main_named": HERE / "mosaic_aaai2027_named.log",
    "supplement_anonymous": HERE / "mosaic_aaai2027_supplement_anonymous.log",
    "supplement_named": HERE / "mosaic_aaai2027_supplement_named.log",
    "checklist": HERE / "mosaic_aaai2027_reproducibility_checklist.log",
}
ARCHIVE = HERE / "MOSAIC_AAAI2027_V10_CODE_DATA_SUPPLEMENT.zip"
OUTPUT = HERE / "MOSAIC_AAAI2027_V10_SUBMISSION_AUDIT.json"
MAX_ARCHIVE_BYTES = 50 * 1024 * 1024
IDENTITY_PATTERNS = (
    re.compile(r"rudra\s*chopra", re.IGNORECASE),
    re.compile(r"contra\s+costa", re.IGNORECASE),
    re.compile(r"github\.com/rudrachopra", re.IGNORECASE),
    re.compile(r"/users/rudrachopra", re.IGNORECASE),
)
LOG_FAILURE_PATTERNS = (
    "LaTeX Warning: There were undefined references",
    "LaTeX Warning: Citation `",
    "Overfull \\hbox",
    "Overfull \\vbox",
    "Fatal error occurred",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_pdf(path: Path, *, is_main: bool, anonymous: bool) -> dict[str, object]:
    failures: list[str] = []
    document = fitz.open(path)
    pages = [page.get_text() for page in document]
    joined = "\n".join(pages)
    if anonymous:
        for pattern in IDENTITY_PATTERNS:
            if pattern.search(joined):
                failures.append(f"identity pattern in visible text: {pattern.pattern}")
    metadata = document.metadata or {}
    author = str(metadata.get("author", "")).strip()
    if anonymous and author and "anonymous" not in author.lower():
        failures.append("non-anonymous author metadata")
    if is_main and len(document) > 9:
        failures.append(f"main paper has {len(document)} pages; expected at most 9")
    for number, page in enumerate(document, start=1):
        rect = page.rect
        if abs(rect.width - 612.0) > 1.0 or abs(rect.height - 792.0) > 1.0:
            failures.append(f"page {number} is not US Letter")
    font_xrefs = {
        font[0]
        for page_number in range(len(document))
        for font in document.get_page_fonts(page_number, full=True)
    }
    unembedded_fonts = []
    for xref in sorted(font_xrefs):
        name, _, _, payload = document.extract_font(xref)
        if not payload:
            unembedded_fonts.append(name)
    if unembedded_fonts:
        failures.append("unembedded fonts: " + ", ".join(unembedded_fonts))
    references = next(
        (index for index, text in enumerate(pages, start=1) if text.lstrip().startswith("References")),
        None,
    )
    if is_main and references != 8:
        failures.append(f"references begin on page {references}, expected page 8")
    return {
        "path": path.name,
        "sha256": sha256(path),
        "pages": len(document),
        "references_start_page": references,
        "font_count": len(font_xrefs),
        "all_fonts_embedded": not unembedded_fonts,
        "metadata": metadata,
        "passed": not failures,
        "failures": failures,
    }


def audit_log(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    failures = [pattern for pattern in LOG_FAILURE_PATTERNS if pattern in text]
    return {
        "path": path.name,
        "passed": not failures,
        "failures": failures,
    }


def audit_archive(path: Path) -> dict[str, object]:
    failures: list[str] = []
    if path.stat().st_size >= MAX_ARCHIVE_BYTES:
        failures.append("archive exceeds 50 MB")
    with zipfile.ZipFile(path) as archive:
        if bad := archive.testzip():
            failures.append(f"corrupt member: {bad}")
        names = archive.namelist()
        if len(names) != len(set(names)):
            failures.append("duplicate archive member")
        manifest_names = [name for name in names if name.endswith("/MANIFEST.sha256")]
        if len(manifest_names) != 1:
            failures.append("archive must contain exactly one checksum manifest")
        for name in names:
            data = archive.read(name)
            text = data.decode("utf-8", errors="ignore")
            for pattern in IDENTITY_PATTERNS:
                if pattern.search(text):
                    failures.append(f"identity pattern in {name}: {pattern.pattern}")
                    break
        if len(manifest_names) == 1:
            root = manifest_names[0].removesuffix("MANIFEST.sha256")
            for line in archive.read(manifest_names[0]).decode("utf-8").splitlines():
                expected, relative = line.split("  ", 1)
                member = root + relative
                if member not in names:
                    failures.append(f"manifest member missing: {relative}")
                    continue
                actual = hashlib.sha256(archive.read(member)).hexdigest()
                if actual != expected:
                    failures.append(f"manifest mismatch: {relative}")
    return {
        "path": path.name,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "members": len(names),
        "passed": not failures,
        "failures": failures,
    }


def main() -> None:
    pdfs = {
        name: audit_pdf(path, is_main=is_main, anonymous=anonymous)
        for name, (path, is_main, anonymous) in PDFS.items()
    }
    logs = {name: audit_log(path) for name, path in LOGS.items()}
    archive = audit_archive(ARCHIVE)
    report = {
        "name": "MOSAIC AAAI 2027 V10 submission audit",
        "passed": (
            all(item["passed"] for item in pdfs.values())
            and all(item["passed"] for item in logs.values())
            and archive["passed"]
        ),
        "pdfs": pdfs,
        "compilation_logs": logs,
        "archive": archive,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
