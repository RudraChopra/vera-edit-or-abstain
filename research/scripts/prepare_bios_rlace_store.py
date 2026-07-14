"""Convert official R-LACE BiasBios embeddings into the shared NumPy store."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pickle
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DEFAULT_RAW = Path("/Volumes/Backups/FARO/artifacts/bios_rlace_upstream/raw")
DEFAULT_OUTPUT = Path("/Volumes/Backups/FARO/artifacts/bios_rlace_numpy_store")
SPLITS = (("train", 0), ("dev", 1), ("test", 2))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_counts(values: np.ndarray) -> dict[str, int]:
    unique, frequency = np.unique(values, return_counts=True)
    return {str(int(key)): int(value) for key, value in zip(unique, frequency)}


def extract_labels(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("rb") as handle:
        records = pickle.load(handle)
    professions: list[str] = []
    genders = np.empty(len(records), dtype=np.int8)
    for index, record in enumerate(records):
        profession = record.get("p")
        gender = record.get("g")
        if not isinstance(profession, str) or gender not in {"f", "m"}:
            raise ValueError(f"unexpected BiasBios record at {path}:{index}")
        professions.append(profession)
        genders[index] = 1 if gender == "f" else 0
    del records
    gc.collect()
    return professions, genders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=16_384)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels: dict[str, tuple[list[str], np.ndarray]] = {}
    embeddings: dict[str, np.ndarray] = {}
    dimensions: set[int] = set()
    for name, _ in SPLITS:
        embedding_path = args.raw_dir / f"{name}_cls.npy"
        record_path = args.raw_dir / f"{name}.pickle"
        if not embedding_path.exists() or not record_path.exists():
            raise FileNotFoundError(f"missing official BiasBios assets for {name}")
        embeddings[name] = np.load(embedding_path, mmap_mode="r")
        labels[name] = extract_labels(record_path)
        if embeddings[name].ndim != 2 or len(embeddings[name]) != len(labels[name][0]):
            raise ValueError(f"embedding/record mismatch in {name}")
        dimensions.add(int(embeddings[name].shape[1]))
    if len(dimensions) != 1:
        raise ValueError(f"inconsistent embedding dimensions: {dimensions}")

    professions = sorted({value for names, _ in labels.values() for value in names})
    profession_to_id = {name: index for index, name in enumerate(professions)}
    n = sum(len(embeddings[name]) for name, _ in SPLITS)
    dimension = dimensions.pop()
    z = np.lib.format.open_memmap(
        args.output_dir / "z.npy", mode="w+", dtype=np.float32, shape=(n, dimension)
    )
    y = np.empty(n, dtype=np.int16)
    s = np.empty(n, dtype=np.int8)
    split = np.empty(n, dtype=np.int8)
    offset = 0
    for name, split_code in SPLITS:
        source = embeddings[name]
        professions_for_split, genders = labels[name]
        stop = offset + len(source)
        for start in range(0, len(source), args.chunk_size):
            chunk_stop = min(len(source), start + args.chunk_size)
            z[offset + start : offset + chunk_stop] = np.asarray(
                source[start:chunk_stop], dtype=np.float32
            )
        y[offset:stop] = np.asarray(
            [profession_to_id[value] for value in professions_for_split], dtype=np.int16
        )
        s[offset:stop] = genders
        split[offset:stop] = split_code
        offset = stop
        z.flush()
    if offset != n or not np.isfinite(z).all():
        raise RuntimeError("BiasBios store is incomplete or non-finite")
    del z
    np.save(args.output_dir / "y.npy", y)
    np.save(args.output_dir / "s.npy", s)
    np.save(args.output_dir / "split.npy", split)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "BiasBios",
        "source": "official R-LACE author artifact server",
        "source_base_url": "https://nlp.biu.ac.il/~ravfogs/rlace-cr/bios/bios_data/",
        "n_examples": n,
        "dimension": dimension,
        "arrays": {"z": "z.npy", "y": "y.npy", "s": "s.npy", "split": "split.npy"},
        "split_codes": {"train": 0, "validation": 1, "external_test": 2},
        "target": "profession (28-way)",
        "source_concept": "binary gender label released with BiasBios",
        "profession_to_id": profession_to_id,
        "target_counts": value_counts(y),
        "source_counts": value_counts(s),
        "split_counts": value_counts(split),
        "assets": {
            f"{name}{suffix}": {
                "bytes": (args.raw_dir / f"{name}{suffix}").stat().st_size,
                "sha256": sha256(args.raw_dir / f"{name}{suffix}"),
            }
            for name, _ in SPLITS
            for suffix in ("_cls.npy", ".pickle")
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output_dir), "n": n, "d": dimension}, indent=2))


if __name__ == "__main__":
    main()
