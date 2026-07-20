#!/usr/bin/env python3
"""Fail closed on the MOSAIC v8 anonymous AAAI submission artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

import fitz


HERE = Path(__file__).resolve().parent
PDFS = {
    "main": HERE / "MOSAIC_AAAI2027_V8_ANONYMOUS.pdf",
    "supplement": HERE / "MOSAIC_AAAI2027_V8_SUPPLEMENT_ANONYMOUS.pdf",
    "checklist": HERE / "MOSAIC_AAAI2027_V8_REPRODUCIBILITY_CHECKLIST.pdf",
}
ARCHIVE = HERE / "MOSAIC_AAAI2027_V8_CODE_DATA_SUPPLEMENT.zip"
OUTPUT = HERE / "MOSAIC_AAAI2027_V8_SUBMISSION_AUDIT.json"
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


def audit_pdf(path: Path, *, is_main: bool) -> dict[str, object]:
    failures: list[str] = []
    document = fitz.open(path)
    pages = [page.get_text() for page in document]
    joined = "\n".join(pages)
    for pattern in IDENTITY_PATTERNS:
        if pattern.search(joined):
            failures.append(f"identity pattern in visible text: {pattern.pattern}")
    metadata = document.metadata or {}
    author = str(metadata.get("author", "")).strip()
    if author and "anonymous" not in author.lower():
        failures.append("non-anonymous author metadata")
    for number, page in enumerate(document, start=1):
        rect = page.rect
        if abs(rect.width - 612.0) > 1.0 or abs(rect.height - 792.0) > 1.0:
            failures.append(f"page {number} is not US Letter")
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
        "metadata": metadata,
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
        for name in names:
            text = archive.read(name).decode("utf-8", errors="ignore")
            for pattern in IDENTITY_PATTERNS:
                if pattern.search(text):
                    failures.append(f"identity pattern in {name}: {pattern.pattern}")
                    break
    return {
        "path": path.name,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "members": len(names),
        "passed": not failures,
        "failures": failures,
    }


def main() -> None:
    reports = {name: audit_pdf(path, is_main=name == "main") for name, path in PDFS.items()}
    archive = audit_archive(ARCHIVE)
    report = {
        "name": "MOSAIC AAAI 2027 v8 submission audit",
        "passed": all(item["passed"] for item in reports.values()) and archive["passed"],
        "pdfs": reports,
        "archive": archive,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
