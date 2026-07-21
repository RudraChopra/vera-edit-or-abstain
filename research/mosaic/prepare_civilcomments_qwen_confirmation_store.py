#!/usr/bin/env python3
"""Extract the locked temporal-confirmation Qwen representation store."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_CSV = Path(
    "/Volumes/Backups/FARO/data/wilds/civilcomments_v1.0/all_data_with_identities.csv"
)
DEFAULT_OUTPUT = Path(
    "/Volumes/Backups/FARO/artifacts/civilcomments_qwen25_temporal_confirmation"
)
DEFAULT_PREREG = (
    REPOSITORY / "research/mosaic/prereg_mosaic_qwen_temporal_confirmation_v1.json"
)
ROLE_CONSTRUCTION = 0
ROLE_REFERENCE = 1
ROLE_TARGET = 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def balanced_sample(
    indices: np.ndarray,
    target: np.ndarray,
    source: np.ndarray,
    *,
    maximum_total: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = tuple(sorted({(int(target[i]), int(source[i])) for i in indices}))
    if groups != ((0, 0), (0, 1), (1, 0), (1, 1)):
        raise ValueError(f"temporal role has incomplete strata: {groups}")
    members = {
        group: indices[
            (target[indices] == group[0]) & (source[indices] == group[1])
        ]
        for group in groups
    }
    per_group = min(min(len(values) for values in members.values()), maximum_total // 4)
    if per_group * 4 != maximum_total:
        raise ValueError(
            f"role cannot fill registered cap {maximum_total}; per-stratum count is {per_group}"
        )
    return np.sort(
        np.concatenate(
            [rng.choice(members[group], size=per_group, replace=False) for group in groups]
        ).astype(np.int64)
    )


def select_temporal_rows(
    metadata: pd.DataFrame,
    *,
    reference_start: str,
    target_start: str,
    role_caps: dict[str, int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    ids = metadata["id"].to_numpy(dtype=np.int64)
    dates = pd.to_datetime(metadata["created_date"], errors="coerce", utc=True)
    target = (metadata["toxicity"].to_numpy(dtype=np.float64) >= 0.5).astype(np.int8)
    source = (metadata["identity_any"].to_numpy(dtype=np.float64) >= 0.5).astype(np.int8)
    confirmation = ids % 4 != 0
    valid = dates.notna().to_numpy()
    reference_cutoff = pd.Timestamp(reference_start, tz="UTC")
    target_cutoff = pd.Timestamp(target_start, tz="UTC")
    roles = (
        confirmation & valid & (dates < reference_cutoff).to_numpy(),
        confirmation
        & valid
        & (dates >= reference_cutoff).to_numpy()
        & (dates < target_cutoff).to_numpy(),
        confirmation & valid & (dates >= target_cutoff).to_numpy(),
    )
    caps = (
        int(role_caps["construction"]),
        int(role_caps["reference"]),
        int(role_caps["target"]),
    )
    sampled = [
        balanced_sample(
            np.flatnonzero(mask).astype(np.int64),
            target,
            source,
            maximum_total=cap,
            seed=seed + offset,
        )
        for offset, (mask, cap) in enumerate(zip(roles, caps, strict=True))
    ]
    rows = np.concatenate(sampled)
    role = np.concatenate(
        [np.full(len(values), code, dtype=np.int8) for code, values in enumerate(sampled)]
    )
    order = np.argsort(rows)
    return rows[order], role[order], int((confirmation & ~valid).sum())


def load_selected_text(csv_path: Path, rows: np.ndarray, chunksize: int) -> list[str]:
    texts: list[str | None] = [None] * len(rows)
    offset = 0
    cursor = 0
    for chunk in pd.read_csv(
        csv_path,
        usecols=["comment_text"],
        chunksize=chunksize,
        low_memory=False,
    ):
        stop = offset + len(chunk)
        while cursor < len(rows) and rows[cursor] < stop:
            value = chunk.iloc[int(rows[cursor] - offset)]["comment_text"]
            texts[cursor] = "" if pd.isna(value) else str(value)
            cursor += 1
        offset = stop
        if cursor == len(rows):
            break
    if cursor != len(rows) or any(value is None for value in texts):
        raise RuntimeError("failed to recover every selected comment")
    return [str(value) for value in texts]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--chunksize", type=int, default=16384)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    prereg = json.loads(args.prereg.read_text(encoding="utf-8"))
    if prereg.get("status") != "locked_confirmation_authorized":
        raise RuntimeError("confirmation preregistration is not authorized")
    expected_script = prereg["code_sha256"][
        "research/mosaic/prepare_civilcomments_qwen_confirmation_store.py"
    ]
    if sha256(Path(__file__).resolve()) != expected_script:
        raise RuntimeError("confirmation extraction source differs from the lock")
    if sha256(args.csv) != prereg["source_csv_sha256"]:
        raise RuntimeError("CivilComments source differs from the lock")

    metadata = pd.read_csv(
        args.csv,
        usecols=["id", "created_date", "toxicity", "identity_any"],
        low_memory=False,
    )
    temporal = prereg["temporal_roles"]
    rows, role, invalid_timestamp_count = select_temporal_rows(
        metadata,
        reference_start=temporal["reference_start_utc"],
        target_start=temporal["target_start_utc"],
        role_caps=temporal["balanced_role_caps"],
        seed=int(temporal["selection_seed"]),
    )
    selected = metadata.iloc[rows].reset_index(drop=True)
    ids = selected["id"].to_numpy(dtype=np.int64)
    target = (selected["toxicity"].to_numpy(dtype=np.float64) >= 0.5).astype(np.int8)
    source = (selected["identity_any"].to_numpy(dtype=np.float64) >= 0.5).astype(np.int8)
    if np.any(ids % 4 == 0):
        raise RuntimeError("pilot ID crossed into temporal confirmation")
    texts = load_selected_text(args.csv, rows, args.chunksize)

    model_spec = prereg["model"]
    representation = prereg["selected_candidate"]["representation"]
    layer = int(prereg["selected_candidate"]["hidden_layer"])
    pooling = str(prereg["selected_candidate"]["pooling"])
    batch_size = int(model_spec["batch_size"])
    max_length = int(model_spec["max_length"])
    if pooling not in {"mean", "last"}:
        raise ValueError("locked pooling rule is invalid")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device.type == "mps" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["model_id"], revision=model_spec["revision"]
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_spec["model_id"],
        revision=model_spec["revision"],
        dtype=dtype,
    ).to(device)
    model.eval()
    if int(model.config.num_hidden_layers) != int(model_spec["layer_count"]):
        raise RuntimeError("model layer count differs from the lock")
    hidden_size = int(model.config.hidden_size)

    args.output.mkdir(parents=True)
    features = np.lib.format.open_memmap(
        args.output / "z.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), hidden_size),
    )
    prompt_prefix = str(model_spec["prompt_prefix"])
    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), batch_size), desc="Qwen temporal hidden states"):
            batch = [prompt_prefix + text for text in texts[start : start + batch_size]]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden = outputs.hidden_states[layer]
            mask = encoded["attention_mask"].unsqueeze(-1)
            if pooling == "mean":
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            else:
                positions = encoded["attention_mask"].sum(dim=1) - 1
                pooled = hidden[torch.arange(hidden.shape[0], device=device), positions]
            stop = min(start + batch_size, len(texts))
            features[start:stop] = pooled.float().cpu().numpy()
    features.flush()
    del features

    np.save(args.output / "y.npy", target)
    np.save(args.output / "s.npy", source)
    np.save(args.output / "split.npy", role)
    np.save(args.output / "ids.npy", ids)
    np.save(args.output / "source_rows.npy", rows)
    group_counts = {
        f"role={role_code},y={yy},s={ss}": int(
            np.sum((role == role_code) & (target == yy) & (source == ss))
        )
        for role_code in (ROLE_CONSTRUCTION, ROLE_REFERENCE, ROLE_TARGET)
        for yy in (0, 1)
        for ss in (0, 1)
    }
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "CivilComments-Qwen2.5-temporal-confirmation",
        "n_examples": len(rows),
        "dimension": hidden_size,
        "arrays": {"z": "z.npy", "y": "y.npy", "s": "s.npy", "split": "split.npy"},
        "auxiliary_arrays": {"ids": "ids.npy", "source_rows": "source_rows.npy"},
        "preregistration_sha256": sha256(args.prereg),
        "source_csv_sha256": prereg["source_csv_sha256"],
        "pilot_partition": "integer dataset id modulo 4 equals 0",
        "confirmation_partition": "integer dataset id modulo 4 is nonzero",
        "temporal_roles": temporal,
        "invalid_confirmation_timestamp_count": invalid_timestamp_count,
        "model": model_spec["model_id"],
        "model_revision": model_spec["revision"],
        "model_layers": int(model.config.num_hidden_layers),
        "representation": representation,
        "hidden_layer": layer,
        "pooling": pooling,
        "prompt_prefix": prompt_prefix,
        "max_length": max_length,
        "padding_side": "right",
        "dtype": "float32",
        "device": device.type,
        "group_counts": group_counts,
        "source_concept": "identity_any >= 0.5 (identity mention, not author identity)",
        "target": "toxicity >= 0.5",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), **manifest}, indent=2, default=str))


if __name__ == "__main__":
    main()

