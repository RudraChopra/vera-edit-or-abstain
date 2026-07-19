#!/usr/bin/env python3
"""Freeze a California-to-Texas ACSIncome store for MOSAIC evaluation.

California supplies disjoint eraser-training and reference splits. Texas is a
geographic deployment population, later split into bridge and diagnostic folds
by the MOSAIC runner. The audited binary source is sex, which is excluded from
the released feature matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DEFAULT_RAW = Path("/Volumes/Backups/FARO/artifacts/acs_folktables_raw")
DEFAULT_OUTPUT = Path("/Volumes/Backups/FARO/artifacts/acs_income_ca_tx_numpy_store")
FEATURES = ("AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "RAC1P")
SOURCE_COLUMN = "SEX"
REFERENCE_FRACTION = 0.20
SPLIT_TRAIN = 0
SPLIT_REFERENCE = 1
SPLIT_EXTERNAL = 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_california_splits(size: int, *, seed: int) -> np.ndarray:
    if size < 2:
        raise ValueError("California requires at least two filtered records")
    reference_count = int(np.floor(REFERENCE_FRACTION * size))
    reference_count = min(max(1, reference_count), size - 1)
    generator = np.random.default_rng(seed)
    order = generator.permutation(size)
    split = np.full(size, SPLIT_TRAIN, dtype=np.int8)
    split[order[:reference_count]] = SPLIT_REFERENCE
    return split


def extract(data: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the canonical Folktables task filter then remove the source feature."""

    from folktables import ACSIncome

    filtered = ACSIncome._preprocess(data)  # Official task preprocessing.
    features = np.asarray(filtered.loc[:, FEATURES].to_numpy(), dtype=np.float32)
    labels = np.asarray(
        ACSIncome.target_transform(filtered[ACSIncome.target]).to_numpy(), dtype=np.int8
    )
    source_values = np.asarray(filtered[SOURCE_COLUMN].to_numpy(), dtype=np.int16)
    unique = set(int(value) for value in np.unique(source_values))
    if unique != {1, 2}:
        raise ValueError(f"unexpected ACS sex coding after filtering: {unique}")
    sources = (source_values == 2).astype(np.int8)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("ACS features must be a finite matrix")
    if not (len(features) == len(labels) == len(sources)):
        raise ValueError("ACS feature, target, and source arrays disagree")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("ACSIncome target must be binary after preprocessing")
    return features, labels, sources


def value_counts(values: np.ndarray) -> dict[str, int]:
    keys, counts = np.unique(values, return_counts=True)
    return {str(int(key)): int(count) for key, count in zip(keys, counts, strict=True)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20_270_719)
    parser.add_argument("--survey-year", default="2018")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite populated store: {args.output_dir}")

    from folktables import ACSDataSource

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    source = ACSDataSource(
        survey_year=args.survey_year,
        horizon="1-Year",
        survey="person",
        root_dir=str(args.raw_dir),
    )
    california = source.get_data(states=["CA"], download=True)
    texas = source.get_data(states=["TX"], download=True)
    x_ca, y_ca, s_ca = extract(california)
    x_tx, y_tx, s_tx = extract(texas)
    split_ca = make_california_splits(len(x_ca), seed=args.seed)
    split_tx = np.full(len(x_tx), SPLIT_EXTERNAL, dtype=np.int8)

    z = np.concatenate((x_ca, x_tx), axis=0)
    y = np.concatenate((y_ca, y_tx), axis=0)
    s = np.concatenate((s_ca, s_tx), axis=0)
    split = np.concatenate((split_ca, split_tx), axis=0)
    g = np.concatenate(
        (np.zeros(len(x_ca), dtype=np.int8), np.ones(len(x_tx), dtype=np.int8))
    )
    if not np.isfinite(z).all() or not all(len(values) == len(z) for values in (y, s, g, split)):
        raise RuntimeError("ACS store arrays are incomplete")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "z.npy", z)
    np.save(args.output_dir / "y.npy", y)
    np.save(args.output_dir / "s.npy", s)
    np.save(args.output_dir / "g.npy", g)
    np.save(args.output_dir / "split.npy", split)
    raw_files = sorted(path for path in args.raw_dir.rglob("*") if path.is_file())
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "Folktables ACSIncome California-to-Texas",
        "source": "US Census ACS PUMS via Folktables 0.0.12",
        "survey_year": str(args.survey_year),
        "horizon": "1-Year",
        "states": {"reference": "CA", "external": "TX"},
        "n_examples": int(len(z)),
        "dimension": int(z.shape[1]),
        "arrays": {"z": "z.npy", "y": "y.npy", "s": "s.npy", "g": "g.npy", "split": "split.npy"},
        "split_codes": {"eraser_train": 0, "reference": 1, "external_target": 2},
        "target": "ACSIncome: personal income above $50,000",
        "source_concept": "binary sex from ACS PUMS, used only for source auditing",
        "excluded_feature": SOURCE_COLUMN,
        "feature_columns": list(FEATURES),
        "california_reference_fraction": REFERENCE_FRACTION,
        "split_seed": int(args.seed),
        "target_counts": value_counts(y),
        "source_counts": value_counts(s),
        "split_counts": value_counts(split),
        "state_target_counts": {"CA": value_counts(y_ca), "TX": value_counts(y_tx)},
        "state_source_counts": {"CA": value_counts(s_ca), "TX": value_counts(s_tx)},
        "raw_assets": {
            str(path.relative_to(args.raw_dir)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in raw_files
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output_dir), "n": int(len(z)), "d": int(z.shape[1])}, indent=2))


if __name__ == "__main__":
    main()
