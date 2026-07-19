#!/usr/bin/env python3
"""Validate a processed public feature store and write its anonymous manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DATASETS = (
    "biasbios_clinical",
    "camelyon17_wilds",
    "civilcomments_wilds",
    "gaitpdb",
    "waterbirds",
    "acs_income_ca_tx",
)
REQUIRED = ("x.npy", "y.npy", "s.npy", "split.npy")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    missing = [name for name in REQUIRED if not (args.source / name).is_file()]
    if missing:
        raise SystemExit(
            "The dataset-specific feature extraction must create x.npy, y.npy, s.npy, and split.npy. "
            f"Missing: {', '.join(missing)}"
        )
    arrays = {name[:-4]: np.load(args.source / name, mmap_mode="r") for name in REQUIRED}
    rows = arrays["x"].shape[0]
    if any(array.shape[0] != rows for array in arrays.values()):
        raise SystemExit("processed arrays have inconsistent row counts")
    args.output.mkdir(parents=True, exist_ok=True)
    files = {name: sha256(args.source / name) for name in REQUIRED}
    manifest = {
        "dataset": args.dataset,
        "rows": rows,
        "feature_dimension": int(arrays["x"].shape[1]),
        "files": files,
        "schema": {name: {"shape": list(array.shape), "dtype": str(array.dtype)} for name, array in arrays.items()},
    }
    destination = args.output / "manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"manifest SHA-256: {sha256(destination)}")


if __name__ == "__main__":
    main()
