"""Convert a TRACE/VERA embedding CSV into the NumPy store format."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SPLIT_CODES = {
    "train": 0,
    "validation": 1,
    "val": 1,
    "external": 2,
    "test": 2,
    "counterfactual": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("input CSV has no header")
        embedding_cols = [name for name in reader.fieldnames if name.startswith("embedding_")]
        required = {"id", "split", "y", "s"}
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"input CSV missing columns: {missing}")
        ids: list[str] = []
        splits: list[int] = []
        targets: list[int] = []
        sources: list[int] = []
        embeddings: list[list[float]] = []
        for row in reader:
            split_name = row["split"].strip().lower()
            if split_name not in SPLIT_CODES:
                raise ValueError(f"unknown split value: {row['split']!r}")
            ids.append(row["id"])
            splits.append(SPLIT_CODES[split_name])
            targets.append(int(row["y"]))
            sources.append(int(row["s"]))
            embeddings.append([float(row[col]) for col in embedding_cols])

    z = np.asarray(embeddings, dtype=np.float32)
    y = np.asarray(targets, dtype=np.int64)
    s = np.asarray(sources, dtype=np.int64)
    split = np.asarray(splits, dtype=np.int64)
    np.save(args.output_dir / "z.npy", z)
    np.save(args.output_dir / "y.npy", y)
    np.save(args.output_dir / "s.npy", s)
    np.save(args.output_dir / "split.npy", split)
    with (args.output_dir / "ids.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "id"])
        writer.writerows((i, value) for i, value in enumerate(ids))

    split_counts = {
        name: int(np.sum(split == code))
        for name, code in {"train": 0, "validation": 1, "external": 2, "counterfactual": 3}.items()
        if int(np.sum(split == code)) > 0
    }
    manifest = {
        "format": "trace_embedding_store_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "name": args.name,
        "source_csv": str(args.input_csv),
        "n_examples": int(z.shape[0]),
        "feature_count": int(z.shape[1]),
        "dtype": "float32",
        "split_codes": {
            "train": 0,
            "validation": 1,
            "external": 2,
            "counterfactual": 3,
        },
        "split_counts": split_counts,
        "arrays": {
            "z": "z.npy",
            "y": "y.npy",
            "s": "s.npy",
            "split": "split.npy",
        },
        "ids": "ids.csv",
        "claim_grade_embedding_store": True,
        "claim_grade_benchmark_row": False,
        "claim_boundary": (
            "Converted from an existing claim-grade TRACE embedding CSV. "
            "Benchmark-row claims require a downstream receipt."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "n_examples": int(z.shape[0])}, indent=2))


if __name__ == "__main__":
    main()
