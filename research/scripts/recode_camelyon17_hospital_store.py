"""Add the official hospital center as Camelyon17's environment-group array."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DEFAULT_INPUT = Path("/Volumes/Backups/FARO/artifacts/camelyon17_resnet18_torch_full_numpy_store")
DEFAULT_METADATA = Path("/Volumes/Backups/FARO/data/wilds/camelyon17_v1.0/metadata.csv")
DEFAULT_OUTPUT = Path("/Volumes/Backups/FARO/artifacts/camelyon17_resnet18_torch_center_numpy_store")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"existing {destination} differs in size from {source}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def counts(values: np.ndarray) -> dict[str, int]:
    return {str(int(key)): int(value) for key, value in Counter(map(int, values)).items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_manifest_path = args.input_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    n = int(source_manifest["n_examples"])
    arrays = source_manifest["arrays"]

    centers_by_metadata_row: list[int] = []
    with args.metadata.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            centers_by_metadata_row.append(int(row["center"]))
    if len(centers_by_metadata_row) != n:
        raise RuntimeError(
            f"metadata has {len(centers_by_metadata_row)} rows, expected {n}"
        )

    ids_path = args.input_dir / str(source_manifest.get("ids", "ids.csv"))
    centers = np.empty(n, dtype=np.int8)
    with ids_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for output_row, row in enumerate(reader):
            identifier = row["id"]
            prefix, metadata_row = identifier.rsplit("_", 1)
            if prefix != "camelyon17":
                raise RuntimeError(f"unexpected identifier: {identifier}")
            centers[output_row] = centers_by_metadata_row[int(metadata_row)]
    if output_row + 1 != n:
        raise RuntimeError(f"ID file has {output_row + 1} rows, expected {n}")

    split = np.load(args.input_dir / arrays["split"], mmap_mode="r")
    split_center_counts = {
        f"split={split_code},center={center}": int(np.sum((split == split_code) & (centers == center)))
        for split_code in sorted(map(int, np.unique(split)))
        for center in sorted(map(int, np.unique(centers[split == split_code])))
    }
    certification_centers = set(map(int, np.unique(centers[split == 1])))
    external_centers = set(map(int, np.unique(centers[split == 2])))
    unsupported_external = sorted(external_centers - certification_centers)
    if not unsupported_external:
        raise RuntimeError("expected at least one external hospital absent from certification")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key in ("z", "y", "split"):
        link_or_copy(args.input_dir / arrays[key], args.output_dir / arrays[key])
    link_or_copy(ids_path, args.output_dir / ids_path.name)
    original_source = np.asarray(
        np.load(args.input_dir / arrays["s"], mmap_mode="r"), dtype=np.int8
    )
    np.save(args.output_dir / "s.npy", original_source)
    np.save(args.output_dir / "g.npy", centers)

    manifest = {
        **source_manifest,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "name": "VERA Camelyon17 frozen ResNet18 store with official hospital environment groups",
        "source_store": str(args.input_dir),
        "source_store_manifest_sha256": sha256(source_manifest_path),
        "raw_metadata": str(args.metadata),
        "raw_metadata_sha256": sha256(args.metadata),
        "source_concept": "binary source concept from the frozen embedding export (center 0 versus centers 1-4)",
        "source_counts": counts(original_source),
        "environment_concept": "official Camelyon17 hospital center ID (0 through 4)",
        "environment_counts": counts(centers),
        "split_center_counts": split_center_counts,
        "certification_source_classes": sorted(certification_centers),
        "external_source_classes": sorted(external_centers),
        "unsupported_external_source_classes": unsupported_external,
        "impossibility_instance": True,
        "claim_boundary": (
            "Center 2 occurs externally but not in certification. VERA must abstain from a "
            "uniform leakage certificate unless an additional structural assumption is supplied."
        ),
        "arrays": {**arrays, "s": "s.npy", "g": "g.npy"},
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output_dir),
        "manifest_sha256": sha256(manifest_path),
        "unsupported_external_source_classes": unsupported_external,
        "split_center_counts": split_center_counts,
    }, indent=2))


if __name__ == "__main__":
    main()
