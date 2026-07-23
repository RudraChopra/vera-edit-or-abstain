#!/usr/bin/env python3
"""Fail closed on the current MOSAIC paper, supplement, and code archive."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "research/maintrack/mosaic_aaai2027"
OUTPUT = ROOT / "research/artifacts/mosaic_submission_package_audit.json"
PDFS = {
    "main_anonymous": (
        PAPER / "mosaic_aaai2027_anonymous.pdf",
        True,
        True,
    ),
    "main_named": (PAPER / "mosaic_aaai2027_named.pdf", True, False),
    "supplement_anonymous": (
        PAPER / "mosaic_aaai2027_supplement_anonymous.pdf",
        False,
        True,
    ),
    "supplement_named": (
        PAPER / "mosaic_aaai2027_supplement_named.pdf",
        False,
        False,
    ),
    "checklist": (
        PAPER / "mosaic_aaai2027_reproducibility_checklist.pdf",
        False,
        True,
    ),
}
ARCHIVE = PAPER / "mosaic_aaai2027_code_data_anonymous.zip"
EVIDENCE_AUDITS = (
    ROOT / "research/artifacts/mosaic_path9_theory_v1_audit.json",
    ROOT / "research/artifacts/mosaic_cinic10_natural_v1_audit.json",
    ROOT / "research/artifacts/mosaic_cinic10_natural_v2_audit.json",
    ROOT / "research/artifacts/mosaic_real_proxy_v1_audit.json",
    ROOT / "research/artifacts/mosaic_fare_proxy_comparison_v1_audit.json",
)
IDENTITY_PATTERNS = (
    re.compile(r"rudra\s*chopra", re.IGNORECASE),
    re.compile(r"github\.com/rudrachopra", re.IGNORECASE),
    re.compile(r"/users/rudrachopra", re.IGNORECASE),
)
LOG_FAILURES = (
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


def audit_pdf(
    path: Path, *, is_main: bool, anonymous: bool
) -> dict[str, object]:
    failures: list[str] = []
    if not path.exists():
        return {"path": str(path), "passed": False, "failures": ["missing"]}
    reader = PdfReader(path)
    page_text = [page.extract_text() or "" for page in reader.pages]
    pages = len(page_text)
    text = "\n".join(page_text)
    if anonymous:
        failures.extend(
            f"identity pattern: {pattern.pattern}"
            for pattern in IDENTITY_PATTERNS
            if pattern.search(text)
        )
        metadata_text = " ".join(
            str(value) for value in (reader.metadata or {}).values()
        )
        failures.extend(
            f"identity metadata pattern: {pattern.pattern}"
            for pattern in IDENTITY_PATTERNS
            if pattern.search(metadata_text)
        )
    references_page = None
    if is_main:
        for page_number, page in enumerate(page_text, start=1):
            if re.search(r"(?m)^References\s*$", page):
                references_page = page_number
                break
        if references_page != 8:
            failures.append(
                f"references start on page {references_page}, expected 8"
            )
    return {
        "path": path.name,
        "pages": pages,
        "references_start_page": references_page,
        "sha256": sha256(path),
        "passed": not failures,
        "failures": failures,
    }


def audit_log(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": path.name, "passed": False, "failures": ["missing"]}
    text = path.read_text(encoding="utf-8", errors="replace")
    failures = [pattern for pattern in LOG_FAILURES if pattern in text]
    return {"path": path.name, "passed": not failures, "failures": failures}


def audit_archive(path: Path) -> dict[str, object]:
    failures: list[str] = []
    if not path.exists():
        return {"path": path.name, "passed": False, "failures": ["missing"]}
    if path.stat().st_size >= 50 * 1024 * 1024:
        failures.append("archive exceeds 50 MB")
    with zipfile.ZipFile(path) as archive:
        if bad := archive.testzip():
            failures.append(f"corrupt member: {bad}")
        names = archive.namelist()
        if len(names) != len(set(names)):
            failures.append("duplicate archive member")
        manifests = [
            name for name in names if name.endswith("/MANIFEST.sha256")
        ]
        if len(manifests) != 1:
            failures.append("expected exactly one checksum manifest")
        for name in names:
            text = archive.read(name).decode("utf-8", errors="ignore")
            if any(pattern.search(text) for pattern in IDENTITY_PATTERNS):
                failures.append(f"identity pattern in {name}")
        if len(manifests) == 1:
            root = manifests[0].removesuffix("MANIFEST.sha256")
            for line in archive.read(manifests[0]).decode().splitlines():
                expected, relative = line.split("  ", 1)
                member = root + relative
                if member not in names:
                    failures.append(f"manifest member missing: {relative}")
                elif hashlib.sha256(archive.read(member)).hexdigest() != expected:
                    failures.append(f"manifest mismatch: {relative}")
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "passed": not failures,
        "failures": failures,
    }


def main() -> None:
    pdfs = {
        name: audit_pdf(path, is_main=is_main, anonymous=anonymous)
        for name, (path, is_main, anonymous) in PDFS.items()
    }
    logs = {
        name: audit_log(PAPER / f"{path.stem}.log")
        for name, (path, _, _) in PDFS.items()
    }
    evidence = {}
    for path in EVIDENCE_AUDITS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        evidence[path.name] = {
            "passed": payload.get("pass") is True,
            "sha256": sha256(path),
        }
    lean = json.loads(
        (
            ROOT / "research/formal/MosaicFormal/BUILD_RECEIPT.json"
        ).read_text(encoding="utf-8")
    )
    archive = audit_archive(ARCHIVE)
    passed = (
        all(item["passed"] for item in pdfs.values())
        and all(item["passed"] for item in logs.values())
        and all(item["passed"] for item in evidence.values())
        and lean.get("status") == "pass"
        and archive["passed"]
    )
    report = {
        "name": "MOSAIC submission package audit",
        "passed": passed,
        "pdfs": pdfs,
        "compilation_logs": logs,
        "evidence_audits": evidence,
        "lean_build_receipt": lean,
        "archive": archive,
    }
    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
