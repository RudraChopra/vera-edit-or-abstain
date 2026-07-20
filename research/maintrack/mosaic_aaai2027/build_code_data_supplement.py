#!/usr/bin/env python3
"""Build and verify the self-contained anonymous AAAI code/data supplement."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
TEMPLATE = HERE / "code_supplement"
PACKAGE_NAME = "mosaic_code_data_supplement"
DEFAULT_OUTPUT = HERE / "mosaic_code_data_supplement.zip"
FIXED_ZIP_TIME = (2026, 7, 19, 0, 0, 0)
MAX_BYTES = 50 * 1024 * 1024
SYNTHETIC_LOCK_COMMIT = "2c9eb880bec838ef0794b235d62c188a27d8267e"

CORE_SOURCES = (
    "research/mosaic/mosaic_channel.py",
    "research/mosaic/mosaic_envelope.py",
    "research/mosaic/mosaic_invariant.py",
    "research/mosaic/mosaic_exact.py",
    "research/mosaic/mosaic_optimizer.py",
    "research/mosaic/mosaic_bridge.py",
    "research/mosaic/mosaic_transform_exact.py",
    "research/mosaic/mosaic_transform_exact_optimizer.py",
    "research/mosaic/mosaic_strict_certification.py",
    "research/mosaic/mosaic_strict_certification_v2.py",
    "research/mosaic/mosaic_rational_certificate.py",
    "research/mosaic/mosaic_real.py",
    "research/mosaic/audit_mosaic_acs_natural_shift.py",
)

REAL_PREP_SOURCES = (
    "research/scripts/prepare_acs_natural_shift_stores.py",
)

FROZEN_ARTIFACTS = {
    "admitted_shift_stress.json": "research/artifacts/mosaic_admitted_shift_stress_v2.json",
    "release_utility_table.json": "research/artifacts/mosaic_release_utility_table_v1.json",
    "scaling_study.json": "research/artifacts/mosaic_scaling_study_v1.json",
    "acs_ca_tx.json": "research/artifacts/mosaic_acs_bridge_strict_v3_summary.json",
    "acs_primary_infeasibility.json": "research/artifacts/mosaic_acs_primary_infeasibility_v1.json",
    "rational_audit_reference.json": "research/artifacts/mosaic_bridge_rational_v2_audit_v1.json",
}

ORIGINAL_CERTIFICATES = REPOSITORY / "research/artifacts/mosaic_bridge_confirmation_receipts_v1"
STRICT_CERTIFICATES = REPOSITORY / "research/artifacts/mosaic_bridge_strict_v2_receipts_v1"
DIRECT_RECEIPTS = REPOSITORY / "research/artifacts/mosaic_direct_target_receipts_v1"
COMPARATOR_RECEIPTS = REPOSITORY / "research/artifacts/mosaic_bridge_comparator_receipts_v1"
NATURAL_SHIFT_RECEIPTS = REPOSITORY / "research/artifacts/mosaic_acs_natural_shift_v1_receipts"

DROP_KEYS = {
    "acknowledgement",
    "acknowledgements",
    "affiliation",
    "affiliations",
    "author",
    "authors",
    "email",
    "emails",
    "official_commit",
    "original_receipt",
    "path",
    "remote",
    "repository",
    "repository_head_at_lock",
    "repository_path",
    "url",
    "urls",
}

FORBIDDEN_PATTERNS = (
    (re.compile(rb"rudra\s*chopra", re.IGNORECASE), "author name"),
    (re.compile(rb"contra\s+costa", re.IGNORECASE), "location identifier"),
    (re.compile(rb"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE), "email address"),
    (re.compile(rb"https?://", re.IGNORECASE), "web address"),
    (re.compile(rb"\bwww\."), "web address"),
    (re.compile(rb"github\.com|anonymousgithub|zenodo|dropbox|drive\.google", re.IGNORECASE), "external host"),
    (re.compile(rb"/(Users|Volumes|home)/", re.IGNORECASE), "absolute user path"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: sanitize_value(item)
            for key, item in value.items()
            if str(key).lower() not in DROP_KEYS
        }
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, str):
        cleaned = re.sub(r"https?://\S+", "<external-reference-removed>", value)
        cleaned = re.sub(r"/(Users|Volumes|home)/[^\s\"']+", "<local-path-removed>", cleaned)
        cleaned = re.sub(r"rudra\s*chopra", "Anonymous Authors", cleaned, flags=re.IGNORECASE)
        return cleaned
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_value(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sanitize_text(text: str) -> str:
    text = re.sub(r"https?://\S+", "<external-reference-removed>", text)
    text = re.sub(r"/(Users|Volumes|home)/[^\s\"']+", "<local-path-removed>", text)
    text = re.sub(r"rudra\s*chopra", "Anonymous Authors", text, flags=re.IGNORECASE)
    return text


def copy_core_sources(root: Path) -> None:
    destination = root / "src"
    destination.mkdir(parents=True, exist_ok=True)
    for relative in CORE_SOURCES:
        source = REPOSITORY / relative
        (destination / source.name).write_text(
            sanitize_text(source.read_text(encoding="utf-8")),
            encoding="utf-8",
        )


def copy_real_preparation_sources(root: Path) -> None:
    destination = root / "data/real"
    destination.mkdir(parents=True, exist_ok=True)
    for relative in REAL_PREP_SOURCES:
        source = REPOSITORY / relative
        text = source.read_text(encoding="utf-8")
        text = text.replace(
            'Path("/Volumes/Backups/FARO/artifacts/acs_folktables_raw")',
            'Path("data/real/raw/acs")',
        ).replace(
            'Path("/Volumes/Backups/FARO/artifacts/acs_natural_shift_stores")',
            'Path("data/real/processed/acs_natural_shift")',
        )
        (destination / source.name).write_text(sanitize_text(text), encoding="utf-8")


def git_blob(commit: str, relative: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def write_full_synthetic_generator(root: Path) -> None:
    prereg_path = REPOSITORY / "research/mosaic/prereg_mosaic_synthetic_v1.json"
    original_prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    generator = root / "data/synthetic/full_generator"
    code_hashes: dict[str, str] = {}
    for relative, expected in original_prereg["code_sha256"].items():
        data = git_blob(SYNTHETIC_LOCK_COMMIT, relative)
        if sha256_bytes(data) != expected:
            raise RuntimeError(f"locked synthetic source mismatch: {relative}")
        destination = generator / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative.endswith("requirements-confirmation.txt"):
            data = (root / "environment/requirements.txt").read_bytes()
        else:
            try:
                data = sanitize_text(data.decode("utf-8")).encode("utf-8")
            except UnicodeDecodeError:
                pass
        destination.write_bytes(data)
        code_hashes[relative] = sha256(destination)
    pilot_hashes: dict[str, str] = {}
    for relative, expected in original_prereg["pilot_artifact_sha256"].items():
        source = REPOSITORY / relative
        if sha256(source) != expected:
            raise RuntimeError(f"locked synthetic pilot mismatch: {relative}")
        destination = generator / relative
        write_json(destination, json.loads(source.read_text(encoding="utf-8")))
        pilot_hashes[relative] = sha256(destination)
    anonymous_prereg = sanitize_value(original_prereg)
    anonymous_prereg["code_sha256"] = code_hashes
    anonymous_prereg["pilot_artifact_sha256"] = pilot_hashes
    anonymous_prereg["runtime_environment"] = {
        "python": "3.12.13",
        "numpy": "2.5.1",
        "scipy": "1.18.0",
        "scikit_learn": "1.9.0",
        "torch": "2.13.0",
        "solver": "HiGHS via SciPy",
        "device": "CPU",
        "confirmation_workers": 8,
    }
    anonymous_prereg.pop("repository_head_at_lock", None)
    output_prereg = generator / "research/mosaic/prereg_mosaic_synthetic_v1.json"
    write_json(output_prereg, anonymous_prereg)
    output_prereg.with_suffix(".sha256").write_text(sha256(output_prereg) + "\n", encoding="utf-8")


def compact_synthetic_safety(root: Path) -> None:
    source = REPOSITORY / "research/artifacts/mosaic_synthetic_confirmation_v1.json"
    report = json.loads(source.read_text(encoding="utf-8"))
    paired: dict[int, dict[str, dict[str, Any]]] = {}
    for row in report["replicate_results"]:
        if row["scenario"] != "hard_safety_boundary" or row["sample_size_per_stratum"] != 125:
            continue
        if row["method"] not in {"plugin_continuum", "mosaic"}:
            continue
        paired.setdefault(int(row["seed"]), {})[row["method"]] = row
    trials = []
    for seed, methods in sorted(paired.items()):
        if set(methods) != {"plugin_continuum", "mosaic"}:
            raise RuntimeError(f"incomplete synthetic safety pair for seed {seed}")
        trials.append(
            {
                "seed": seed,
                "plugin_continuum_false_acceptance": bool(methods["plugin_continuum"]["false_acceptance"]),
                "mosaic_false_acceptance": bool(methods["mosaic"]["false_acceptance"]),
            }
        )
    if len(trials) != 1000:
        raise RuntimeError(f"expected 1000 synthetic safety pairs, found {len(trials)}")
    write_json(
        root / "artifacts/frozen/synthetic_safety_trials.json",
        {
            "source_sha256": sha256(source),
            "scenario": "hard_safety_boundary",
            "sample_size_per_stratum": 125,
            "trials": trials,
        },
    )


def compact_matched_baselines(root: Path) -> None:
    source = REPOSITORY / "research/artifacts/mosaic_baseline_extension_v1_schema_repaired.json"
    report = json.loads(source.read_text(encoding="utf-8"))
    paired: dict[int, dict[str, dict[str, Any]]] = {}
    for row in report["replicate_results"]:
        if row["scenario"] != "retention_and_stochastic_value" or row["sample_size_per_stratum"] != 250:
            continue
        if row["method"] not in {"mosaic_continuum", "holm_ltt_grid"}:
            continue
        paired.setdefault(int(row["seed"]), {})[row["method"]] = row
    trials = []
    for seed, methods in sorted(paired.items()):
        if set(methods) != {"mosaic_continuum", "holm_ltt_grid"}:
            raise RuntimeError(f"incomplete matched baseline pair for seed {seed}")
        trials.append(
            {
                "seed": seed,
                "mosaic_deployed": bool(methods["mosaic_continuum"]["deployed"]),
                "holm_ltt_deployed": bool(methods["holm_ltt_grid"]["deployed"]),
                "mosaic_false_acceptance": bool(methods["mosaic_continuum"]["false_acceptance"]),
                "holm_ltt_false_acceptance": bool(methods["holm_ltt_grid"]["false_acceptance"]),
            }
        )
    if len(trials) != 1000:
        raise RuntimeError(f"expected 1000 matched baseline pairs, found {len(trials)}")
    write_json(
        root / "artifacts/frozen/matched_baseline_trials.json",
        {
            "source_sha256": sha256(source),
            "scenario": "retention_and_stochastic_value",
            "sample_size_per_stratum": 250,
            "trials": trials,
        },
    )


def compact_bridge_power(root: Path) -> None:
    source = REPOSITORY / "research/artifacts/mosaic_bridge_misspecification_v1.json"
    report = json.loads(source.read_text(encoding="utf-8"))
    write_json(
        root / "artifacts/frozen/bridge_power.json",
        {
            "source_sha256": sha256(source),
            "cells": report["cells"],
            "scenarios": report["scenarios"],
            "replicates_per_cell": report["replicates_per_cell"],
        },
    )


def selection_row(path: Path, *, kind: str, rule: str | None = None) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if kind in {"strict", "direct"}:
        selected = report["selection_by_utility_threshold"]["0.40"]
    else:
        if rule is None:
            raise ValueError("comparator rule is required")
        selected = report["selection_by_rule_and_utility_threshold"][rule]["0.40"]
    return {
        "dataset": report["dataset"],
        "seed": int(report["seed"]),
        "decision": selected["decision"],
        "false_acceptance": bool(selected.get("false_acceptance", False)),
        "diagnostic_estimable": bool(selected.get("diagnostic_estimable", False)),
    }


def compact_real_jobs(root: Path) -> None:
    rules: dict[str, list[dict[str, Any]]] = {
        "strict_mosaic": [selection_row(path, kind="strict") for path in sorted(STRICT_CERTIFICATES.glob("*.json"))],
        "direct_target": [selection_row(path, kind="direct") for path in sorted(DIRECT_RECEIPTS.glob("*.json"))],
    }
    comparator_map = {
        "bridge_plugin": "bridge_plugin",
        "validation_plugin": "validation_plugin",
        "unconditional": "always_deploy_validation",
    }
    comparator_paths = sorted(COMPARATOR_RECEIPTS.glob("*.json"))
    for output_name, receipt_name in comparator_map.items():
        rules[output_name] = [
            selection_row(path, kind="comparator", rule=receipt_name)
            for path in comparator_paths
        ]
    if any(len(rows) != 100 for rows in rules.values()):
        raise RuntimeError(f"real 100-job replay has incomplete rule rows: { {key: len(value) for key, value in rules.items()} }")
    write_json(
        root / "artifacts/frozen/real_100jobs.json",
        {"utility_threshold": 0.40, "jobs_by_rule": rules},
    )


def copy_frozen_artifacts(root: Path) -> None:
    frozen = root / "artifacts/frozen"
    frozen.mkdir(parents=True, exist_ok=True)
    for output_name, relative in FROZEN_ARTIFACTS.items():
        source = REPOSITORY / relative
        write_json(frozen / output_name, json.loads(source.read_text(encoding="utf-8")))
    compact_synthetic_safety(root)
    compact_matched_baselines(root)
    compact_bridge_power(root)
    compact_real_jobs(root)


def copy_certificate_pairs(root: Path) -> None:
    original_output = root / "artifacts/certificates/original"
    strict_output = root / "artifacts/certificates/strict"
    originals = sorted(ORIGINAL_CERTIFICATES.glob("*.json"))
    strict = {path.name: path for path in STRICT_CERTIFICATES.glob("*.json")}
    if set(strict) != {path.name for path in originals}:
        raise RuntimeError("original and strict certificate sets differ")
    for original_path in originals:
        original_payload = json.loads(original_path.read_text(encoding="utf-8"))
        output_original = original_output / original_path.name
        write_json(output_original, original_payload)
        strict_payload = sanitize_value(json.loads(strict[original_path.name].read_text(encoding="utf-8")))
        strict_payload["original_receipt_sha256"] = sha256(output_original)
        write_json(strict_output / original_path.name, strict_payload)


def copy_natural_shift_evidence(root: Path) -> None:
    prereg_source = REPOSITORY / "research/mosaic/prereg_mosaic_acs_natural_shift_v1.json"
    data_lock_source = REPOSITORY / "research/mosaic/prereg_mosaic_acs_natural_shift_data_v1.json"
    prereg = sanitize_value(json.loads(prereg_source.read_text(encoding="utf-8")))
    destination = root / "artifacts/natural_shift"
    prereg_output = destination / "preregistration.json"
    write_json(prereg_output, prereg)
    prereg_sha = sha256(prereg_output)

    data_lock = sanitize_value(json.loads(data_lock_source.read_text(encoding="utf-8")))
    data_lock["preregistration_sha256"] = prereg_sha
    data_lock_output = root / "data/real/acs_natural_shift_data_lock.json"
    write_json(data_lock_output, data_lock)

    receipt_output = destination / "receipts"
    receipts = sorted(NATURAL_SHIFT_RECEIPTS.glob("ACS-*.json"))
    if len(receipts) != 60:
        raise RuntimeError(f"expected 60 natural-shift receipts, found {len(receipts)}")
    for source in receipts:
        payload = sanitize_value(json.loads(source.read_text(encoding="utf-8")))
        payload["preregistration_sha256"] = prereg_sha
        write_json(receipt_output / source.name, payload)

    for name, relative in (
        ("summary.json", "research/artifacts/mosaic_acs_natural_shift_v1_summary.json"),
        ("audit.json", "research/artifacts/mosaic_acs_natural_shift_v1_audit.json"),
    ):
        source = REPOSITORY / relative
        write_json(destination / name, json.loads(source.read_text(encoding="utf-8")))


def write_manifest(root: Path) -> None:
    manifest = root / "MANIFEST_SHA256.txt"
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != manifest:
            rows.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}")
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")


def scan_anonymity(root: Path) -> None:
    failures = []
    for path in sorted(root.rglob("*")):
        if path.name == ".git" or ".git" in path.parts:
            failures.append(f"version-control metadata: {path.relative_to(root)}")
            continue
        if not path.is_file():
            continue
        data = path.read_bytes()
        for pattern, label in FORBIDDEN_PATTERNS:
            if pattern.search(data):
                failures.append(f"{path.relative_to(root)} contains {label}")
    if failures:
        raise RuntimeError("anonymity scan failed:\n" + "\n".join(failures[:100]))


def verify_python(root: Path) -> None:
    scripts = [str(path) for path in root.rglob("*.py")]
    subprocess.run([sys.executable, "-m", "py_compile", *scripts], check=True)
    smoke = (
        "import sys; "
        f"sys.path.insert(0, {str(root / 'src')!r}); "
        "import mosaic_bridge, mosaic_strict_certification_v2, "
        "mosaic_rational_certificate, mosaic_real; "
        "print('core implementation imports: pass')"
    )
    subprocess.run([sys.executable, "-c", smoke], check=True)


def remove_python_caches(root: Path) -> None:
    for cache in sorted(root.rglob("__pycache__"), reverse=True):
        shutil.rmtree(cache)


def verify_manifest(root: Path) -> None:
    subprocess.run([sys.executable, str(root / "verify/check_hashes.py")], cwd=root, check=True)


def write_zip(root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            name = (Path(PACKAGE_NAME) / path.relative_to(root)).as_posix()
            info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = 0o100755 if path.suffix == ".py" else 0o100644
            info.external_attr = mode << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    temporary.replace(output)


def verify_zip(output: Path) -> tuple[int, int]:
    if output.stat().st_size >= MAX_BYTES:
        raise RuntimeError(f"archive is {output.stat().st_size} bytes; limit is {MAX_BYTES}")
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"corrupt archive member: {bad}")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("archive contains duplicate members")
        if not all(name.startswith(f"{PACKAGE_NAME}/") for name in names):
            raise RuntimeError("archive has an unexpected top-level member")
        for name in names:
            data = archive.read(name)
            for pattern, label in FORBIDDEN_PATTERNS:
                if pattern.search(data):
                    raise RuntimeError(f"archive member {name} contains {label}")
    return len(names), output.stat().st_size


def tree_summary(root: Path) -> list[str]:
    rows = []
    for child in sorted(root.iterdir()):
        if child.is_dir():
            files = sum(path.is_file() for path in child.rglob("*"))
            rows.append(f"{child.name}/ ({files} files)")
        else:
            rows.append(child.name)
    return rows


def build(output: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="mosaic-code-data-") as temporary:
        root = Path(temporary) / PACKAGE_NAME
        shutil.copytree(TEMPLATE, root)
        copy_core_sources(root)
        copy_real_preparation_sources(root)
        write_full_synthetic_generator(root)
        copy_frozen_artifacts(root)
        copy_certificate_pairs(root)
        copy_natural_shift_evidence(root)
        verify_python(root)
        remove_python_caches(root)
        write_manifest(root)
        scan_anonymity(root)
        verify_manifest(root)
        write_zip(root, output)
        tree = tree_summary(root)
    members, size = verify_zip(output)
    return {
        "output": str(output),
        "sha256": sha256(output),
        "bytes": size,
        "megabytes": size / (1024 * 1024),
        "members": members,
        "size_limit_megabytes": 50,
        "anonymity_scan": "pass",
        "manifest_verification": "pass",
        "python_syntax": "pass",
        "zip_integrity": "pass",
        "top_level_tree": tree,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(build(args.output.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
