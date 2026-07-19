#!/usr/bin/env python3
"""Fail closed on the final MOSAIC v7 AAAI submission artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

import fitz


HERE = Path(__file__).resolve().parent
MAIN = HERE / "MOSAIC_AAAI2027_V7_ANONYMOUS.pdf"
SUPPLEMENT = HERE / "MOSAIC_AAAI2027_V7_SUPPLEMENT_ANONYMOUS.pdf"
CHECKLIST = HERE / "MOSAIC_AAAI2027_V7_REPRODUCIBILITY_CHECKLIST.pdf"
ARCHIVE = HERE / "mosaic_code_data_supplement.zip"
OUTPUT = HERE / "MOSAIC_AAAI2027_V7_SUBMISSION_AUDIT.json"
MAX_ARCHIVE_BYTES = 50 * 1024 * 1024
IDENTITY_PATTERNS = (
    re.compile(r"rudra\s*chopra", re.IGNORECASE),
    re.compile(r"contra\s+costa", re.IGNORECASE),
    re.compile(r"github\.com", re.IGNORECASE),
    re.compile(r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_pdf(path: Path, *, main: bool) -> dict[str, object]:
    failures: list[str] = []
    if not path.is_file():
        return {"path": str(path), "passed": False, "failures": ["missing PDF"]}
    document = fitz.open(path)
    texts = [page.get_text() for page in document]
    joined = "\n".join(texts)
    for pattern in IDENTITY_PATTERNS:
        if pattern.search(joined):
            failures.append(f"identity pattern in visible text: {pattern.pattern}")
    metadata = document.metadata or {}
    author = str(metadata.get("author", "")).strip()
    if author and "anonymous" not in author.lower():
        failures.append("PDF author metadata is not anonymous")
    for page_number, page in enumerate(document, start=1):
        rectangle = page.rect
        if abs(rectangle.width - 612.0) > 1.0 or abs(rectangle.height - 792.0) > 1.0:
            failures.append(f"page {page_number} is not US Letter")
    unembedded = set()
    for page in document:
        for font in page.get_fonts(full=True):
            xref, extension, font_type, base_name = font[:4]
            if xref <= 0:
                unembedded.add(str(base_name))
                continue
            extracted = document.extract_font(xref)
            if not extracted or not extracted[-1]:
                unembedded.add(str(base_name))
    if unembedded:
        failures.append("unembedded fonts: " + ", ".join(sorted(unembedded)))
    reference_start = None
    if main:
        for index, text in enumerate(texts):
            if re.search(r"(?:^|\n)References(?:\n|$)", text):
                reference_start = index + 1
                break
        if reference_start != 8:
            failures.append(f"references start on page {reference_start}, expected page 8")
        elif not texts[7].lstrip().startswith("References"):
            failures.append("technical content spills onto page 8 before references")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "pages": len(document),
        "reference_start_page": reference_start,
        "metadata": metadata,
        "unembedded_fonts": sorted(unembedded),
        "passed": not failures,
        "failures": failures,
    }


def audit_archive(path: Path) -> dict[str, object]:
    failures: list[str] = []
    if not path.is_file():
        return {"path": str(path), "passed": False, "failures": ["missing archive"]}
    if path.stat().st_size >= MAX_ARCHIVE_BYTES:
        failures.append("archive exceeds 50 MB")
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            failures.append(f"corrupt member: {bad}")
        names = archive.namelist()
        if len(names) != len(set(names)):
            failures.append("duplicate archive members")
        for name in names:
            data = archive.read(name)
            for pattern in IDENTITY_PATTERNS:
                if pattern.search(data.decode("utf-8", errors="ignore")):
                    failures.append(f"identity pattern in {name}: {pattern.pattern}")
                    break
    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "members": len(names),
        "passed": not failures,
        "failures": failures,
    }


def main() -> None:
    pdfs = {
        "main": audit_pdf(MAIN, main=True),
        "supplement": audit_pdf(SUPPLEMENT, main=False),
        "checklist": audit_pdf(CHECKLIST, main=False),
    }
    archive = audit_archive(ARCHIVE)
    logs = sorted(HERE.glob("mosaic_aaai2027*.log"))
    log_failures = []
    for path in logs:
        text = path.read_text(encoding="utf-8", errors="replace")
        for marker in ("Overfull \\hbox", "Undefined control sequence", "Citation `", "Reference `"):
            if marker in text:
                log_failures.append(f"{path.name}: {marker}")
    passed = all(report["passed"] for report in pdfs.values()) and archive["passed"] and not log_failures
    report = {
        "name": "MOSAIC AAAI 2027 v7 submission audit",
        "passed": passed,
        "pdfs": pdfs,
        "archive": archive,
        "latex_log_failures": log_failures,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
