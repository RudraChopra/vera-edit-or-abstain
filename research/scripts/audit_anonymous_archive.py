"""Independently audit and compact-reproduce the anonymous VERA archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from pypdf import PdfReader


REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = REPOSITORY / "dist" / "vera_anonymous_submission.zip"
DEFAULT_OUTPUT = (
    REPOSITORY / "research" / "artifacts" / "vera_anonymous_archive_audit.json"
)
TEXT_SUFFIXES = {
    ".bib",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sha256",
    ".sty",
    ".tex",
    ".txt",
}
FORBIDDEN = (
    "rudrachopra",
    "rudra chopra",
    "/users/rudra",
    "github.com/rudrachopra",
)
LEGACY_NAME = "faro"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        bool(name)
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in name
        and not name.startswith("/")
    )


def scan_text(name: str, text: str) -> tuple[list[str], list[str]]:
    lowered = text.lower()
    identities = [token for token in FORBIDDEN if token in lowered]
    branding_text = lowered.replace("/volumes/backups/faro", "<external-storage>")
    legacy = [LEGACY_NAME] if LEGACY_NAME in branding_text else []
    return identities, legacy


def pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pieces = [json.dumps(dict(reader.metadata or {}), sort_keys=True, default=str)]
    pieces.extend(page.extract_text() or "" for page in reader.pages)
    for command in (["pdfinfo", str(path)], ["pdftotext", str(path), "-"]):
        if shutil.which(command[0]) is None:
            continue
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if result.returncode == 0:
            pieces.append(result.stdout)
    return "\n".join(pieces)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures: list[str] = []
    identity_hits: dict[str, list[str]] = {}
    legacy_name_hits: list[str] = []
    reproduction: dict[str, Any] = {
        "attempted": False,
        "passed": False,
        "returncode": None,
        "stdout_tail": "",
        "stderr_tail": "",
    }
    payload_count = 0
    receipt_count = 0
    independent_stress_receipt_count = 0
    manifest: dict[str, Any] = {}

    if not args.archive.is_file():
        failures.append(f"archive does not exist: {args.archive}")
    else:
        try:
            with zipfile.ZipFile(args.archive) as archive, tempfile.TemporaryDirectory(
                prefix="vera-anonymous-audit-"
            ) as temporary:
                names = archive.namelist()
                unsafe = [name for name in names if not safe_member(name)]
                duplicates = sorted({name for name in names if names.count(name) > 1})
                if unsafe:
                    failures.append(f"unsafe archive paths: {unsafe}")
                if duplicates:
                    failures.append(f"duplicate archive paths: {duplicates}")
                if names.count("MANIFEST.json") != 1:
                    failures.append("archive must contain exactly one MANIFEST.json")
                else:
                    manifest = json.loads(archive.read("MANIFEST.json"))
                    entries = manifest.get("entries", [])
                    if not isinstance(entries, list):
                        failures.append("manifest entries is not a list")
                        entries = []
                    declared = {
                        str(entry.get("path")): entry
                        for entry in entries
                        if isinstance(entry, dict)
                    }
                    actual = set(names) - {"MANIFEST.json"}
                    if set(declared) != actual:
                        failures.append("manifest payload paths do not match ZIP payload")
                    payload_count = len(actual)
                    receipt_count = sum(
                        name.startswith(
                            "research/artifacts/confirmatory_balanced_receipts/"
                        )
                        for name in actual
                    )
                    independent_stress_receipt_count = sum(
                        name.startswith(
                            "research/artifacts/independent_stress_replication_receipts/"
                        )
                        for name in actual
                    )
                    if receipt_count != 200:
                        failures.append(
                            f"expected 200 confirmatory receipts, found {receipt_count}"
                        )
                    if independent_stress_receipt_count != 800:
                        failures.append(
                            "expected 800 independent stress receipts, found "
                            f"{independent_stress_receipt_count}"
                        )
                    if manifest.get("payload_file_count") != payload_count:
                        failures.append("manifest payload count is inconsistent")
                    if manifest.get("confirmatory_receipt_count") != receipt_count:
                        failures.append("manifest receipt count is inconsistent")
                    if (
                        manifest.get("independent_stress_receipt_count")
                        != independent_stress_receipt_count
                    ):
                        failures.append(
                            "manifest independent stress receipt count is inconsistent"
                        )
                    for name in sorted(actual):
                        data = archive.read(name)
                        entry = declared.get(name, {})
                        if entry.get("sha256") != sha256(data):
                            failures.append(f"SHA-256 mismatch: {name}")
                        if entry.get("bytes") != len(data):
                            failures.append(f"byte-count mismatch: {name}")
                        path_identities, path_legacy = scan_text(name, name)
                        if path_identities:
                            identity_hits[name] = path_identities
                        if path_legacy:
                            legacy_name_hits.append(name)
                        if Path(name).suffix.lower() in TEXT_SUFFIXES:
                            try:
                                text = data.decode("utf-8")
                            except UnicodeDecodeError:
                                failures.append(f"non-UTF-8 text payload: {name}")
                                continue
                            found_identities, found_legacy = scan_text(name, text)
                            if found_identities:
                                identity_hits[name] = sorted(set(found_identities))
                            if found_legacy:
                                legacy_name_hits.append(name)
                    if not failures:
                        destination = Path(temporary)
                        archive.extractall(destination)
                        for pdf in destination.rglob("*.pdf"):
                            found_identities, found_legacy = scan_text(
                                pdf.relative_to(destination).as_posix(), pdf_text(pdf)
                            )
                            if found_identities:
                                identity_hits[
                                    pdf.relative_to(destination).as_posix()
                                ] = sorted(set(found_identities))
                            if found_legacy:
                                legacy_name_hits.append(
                                    pdf.relative_to(destination).as_posix()
                                )
                        if not identity_hits and not legacy_name_hits:
                            reproduction["attempted"] = True
                            result = subprocess.run(
                                [
                                    sys.executable,
                                    "research/scripts/reproduce_vera_submission.py",
                                ],
                                cwd=destination,
                                text=True,
                                capture_output=True,
                                check=False,
                            )
                            reproduction.update(
                                {
                                    "passed": result.returncode == 0,
                                    "returncode": result.returncode,
                                    "stdout_tail": result.stdout[-4000:],
                                    "stderr_tail": result.stderr[-4000:],
                                }
                            )
                            if result.returncode != 0:
                                failures.append("compact reproduction failed")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            failures.append(f"archive audit error: {error}")

    if identity_hits:
        failures.append("identity-bearing content remains in anonymous archive")
    if legacy_name_hits:
        failures.append("legacy project name remains in anonymous archive")
    passed = not failures and reproduction["passed"] is True
    report = {
        "passed": passed,
        "archive": str(args.archive),
        "archive_sha256": sha256(args.archive.read_bytes()) if args.archive.is_file() else None,
        "payload_file_count": payload_count,
        "confirmatory_receipt_count": receipt_count,
        "independent_stress_receipt_count": independent_stress_receipt_count,
        "identity_hits": identity_hits,
        "legacy_name_hits": sorted(set(legacy_name_hits)),
        "source_commit": manifest.get("source_commit"),
        "reproduction": reproduction,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
