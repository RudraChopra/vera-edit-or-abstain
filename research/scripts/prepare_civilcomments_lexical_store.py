"""Build an out-of-core CivilComments frozen lexical representation store."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer


DEFAULT_CSV = Path(
    "/Volumes/Backups/FARO/data/wilds/civilcomments_v1.0/all_data_with_identities.csv"
)
DEFAULT_OUTPUT = Path("/Volumes/Backups/FARO/artifacts/civilcomments_lexical_numpy_store")
SPLIT_MAP = {"train": 0, "val": 1, "test": 2}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def counts(values: np.ndarray) -> dict[str, int]:
    unique, frequency = np.unique(values, return_counts=True)
    return {str(int(key)): int(value) for key, value in zip(unique, frequency)}


def group_counts(y: np.ndarray, s: np.ndarray, split: np.ndarray) -> dict[str, int]:
    result: Counter[str] = Counter()
    for yy, ss, split_code in zip(y, s, split):
        result[f"split={int(split_code)},y={int(yy)},s={int(ss)}"] += 1
    return dict(sorted(result.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--hash-features", type=int, default=65_536)
    parser.add_argument("--fit-sample", type=int, default=100_000)
    parser.add_argument("--chunksize", type=int, default=16_384)
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(
        args.csv,
        usecols=["id", "split", "toxicity", "identity_any"],
        low_memory=False,
    )
    split_names = metadata["split"].astype(str).to_numpy()
    unknown_splits = sorted(set(split_names) - set(SPLIT_MAP))
    if unknown_splits:
        raise ValueError(f"unknown CivilComments split values: {unknown_splits}")
    split = np.asarray([SPLIT_MAP[value] for value in split_names], dtype=np.int8)
    y = (metadata["toxicity"].to_numpy(dtype=np.float64) >= 0.5).astype(np.int8)
    s = (metadata["identity_any"].to_numpy(dtype=np.float64) >= 0.5).astype(np.int8)
    ids = metadata["id"].to_numpy(dtype=np.int64)
    n = len(metadata)

    train_indices = np.flatnonzero(split == SPLIT_MAP["train"])
    rng = np.random.default_rng(args.seed)
    fit_count = min(args.fit_sample, len(train_indices))
    fit_indices = np.sort(rng.choice(train_indices, size=fit_count, replace=False))
    fit_mask = np.zeros(n, dtype=bool)
    fit_mask[fit_indices] = True
    fit_texts: list[str] = []
    offset = 0
    for chunk in pd.read_csv(
        args.csv,
        usecols=["comment_text"],
        chunksize=args.chunksize,
        low_memory=False,
    ):
        texts = chunk["comment_text"].fillna("").astype(str).tolist()
        local_mask = fit_mask[offset : offset + len(texts)]
        fit_texts.extend(text for text, keep in zip(texts, local_mask) if keep)
        offset += len(texts)
    if offset != n or len(fit_texts) != fit_count:
        raise RuntimeError("text scan did not match metadata rows")

    vectorizer = HashingVectorizer(
        n_features=args.hash_features,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        dtype=np.float32,
    )
    fit_matrix = vectorizer.transform(fit_texts)
    svd = TruncatedSVD(
        n_components=args.dimension,
        algorithm="randomized",
        n_iter=7,
        random_state=args.seed,
    )
    svd.fit(fit_matrix)
    del fit_matrix, fit_texts

    z_path = args.output_dir / "z.npy"
    z = np.lib.format.open_memmap(
        z_path, mode="w+", dtype=np.float32, shape=(n, args.dimension)
    )
    offset = 0
    for chunk in pd.read_csv(
        args.csv,
        usecols=["comment_text"],
        chunksize=args.chunksize,
        low_memory=False,
    ):
        texts = chunk["comment_text"].fillna("").astype(str).tolist()
        transformed = svd.transform(vectorizer.transform(texts)).astype(np.float32)
        z[offset : offset + len(texts)] = transformed
        offset += len(texts)
        z.flush()
    if offset != n or not np.isfinite(z).all():
        raise RuntimeError("embedding export is incomplete or non-finite")
    del z

    np.save(args.output_dir / "y.npy", y)
    np.save(args.output_dir / "s.npy", s)
    np.save(args.output_dir / "split.npy", split)
    np.save(args.output_dir / "ids.npy", ids)
    np.save(args.output_dir / "svd_components.npy", svd.components_.astype(np.float32))
    np.save(
        args.output_dir / "svd_explained_variance_ratio.npy",
        svd.explained_variance_ratio_.astype(np.float32),
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "CivilComments-WILDS",
        "source_csv": str(args.csv),
        "source_csv_sha256": sha256(args.csv),
        "n_examples": n,
        "dimension": args.dimension,
        "arrays": {
            "z": "z.npy",
            "y": "y.npy",
            "s": "s.npy",
            "split": "split.npy",
            "ids": "ids.npy",
        },
        "split_codes": {"train": 0, "validation": 1, "external_test": 2},
        "target": "toxicity >= 0.5",
        "source_concept": "identity_any >= 0.5",
        "encoder": {
            "family": "HashingVectorizer plus train-only TruncatedSVD",
            "hash_features": args.hash_features,
            "ngram_range": [1, 2],
            "stop_words": "english",
            "fit_sample": fit_count,
            "fit_sample_seed": args.seed,
            "dimension": args.dimension,
            "fit_uses_external_split": False,
        },
        "label_counts": counts(y),
        "source_counts": counts(s),
        "split_counts": counts(split),
        "group_counts": group_counts(y, s, split),
        "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output_dir), "n": n, "d": args.dimension}, indent=2))


if __name__ == "__main__":
    main()
