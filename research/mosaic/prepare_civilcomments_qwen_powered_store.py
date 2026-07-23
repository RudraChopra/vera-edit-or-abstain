#!/usr/bin/env python3
"""Extract the locked powered Qwen temporal hidden-state store."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from prepare_civilcomments_qwen_confirmation_store import (
    ROLE_CONSTRUCTION,
    ROLE_REFERENCE,
    ROLE_TARGET,
    load_selected_text,
    select_temporal_rows,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = (
    ROOT
    / "research/data/wilds/civilcomments_v1.0/"
    "all_data_with_identities.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "research/data/civilcomments_qwen25_powered_confirmation"
)
DEFAULT_PREREG = (
    ROOT / "research/mosaic/prereg_mosaic_qwen_powered_confirmation_v1.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def validate_lock(prereg_path: Path, csv_path: Path) -> dict[str, Any]:
    sidecar = prereg_path.with_suffix(prereg_path.suffix + ".sha256")
    if sidecar.read_text(encoding="utf-8").split()[0] != sha256(prereg_path):
        raise ValueError("Qwen powered lock sidecar mismatch")
    prereg = load(prereg_path)
    if prereg.get("status") != "locked_before_model_and_outcomes":
        raise RuntimeError("Qwen powered preregistration is not locked")
    if csv_path.stat().st_size != prereg["source_csv"]["bytes"]:
        raise RuntimeError("CivilComments source size differs from the lock")
    if sha256(csv_path) != prereg["source_csv"]["sha256"]:
        raise RuntimeError("CivilComments source hash differs from the lock")
    for relative, expected in prereg["code_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"locked source mismatch: {relative}")
    for local in (prereg_path, sidecar):
        relative = local.relative_to(ROOT)
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative.as_posix()}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if committed != local.read_bytes():
            raise RuntimeError(f"{relative} is not the committed lock")
    return prereg


def save_progress(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--chunksize", type=int, default=16384)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prereg = validate_lock(args.prereg, args.csv)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"completed store already exists: {args.output}")

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
    target = (
        selected["toxicity"].to_numpy(dtype=np.float64) >= 0.5
    ).astype(np.int8)
    source = (
        selected["identity_any"].to_numpy(dtype=np.float64) >= 0.5
    ).astype(np.int8)
    if np.any(ids % 4 == 0):
        raise RuntimeError("pilot ID crossed into the powered confirmation")
    texts = load_selected_text(args.csv, rows, args.chunksize)

    model_spec = prereg["model"]
    batch_size = int(model_spec["batch_size"])
    max_length = int(model_spec["max_length"])
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device.type == "mps" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["model_id"],
        revision=model_spec["revision"],
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

    args.output.mkdir(parents=True, exist_ok=True)
    rows_hash = sha256_array(rows)
    progress_path = args.output / "progress.json"
    feature_path = args.output / "z.npy"
    start_row = 0
    if progress_path.exists():
        progress = load(progress_path)
        if progress["preregistration_sha256"] != sha256(args.prereg):
            raise RuntimeError("partial store belongs to another lock")
        if progress["source_rows_sha256"] != rows_hash:
            raise RuntimeError("partial store row selection differs")
        if progress["shape"] != [len(rows), hidden_size]:
            raise RuntimeError("partial store shape differs")
        start_row = int(progress["completed_rows"])
        features = np.lib.format.open_memmap(feature_path, mode="r+")
    else:
        features = np.lib.format.open_memmap(
            feature_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(rows), hidden_size),
        )
        np.save(args.output / "y.npy", target)
        np.save(args.output / "s.npy", source)
        np.save(args.output / "split.npy", role)
        np.save(args.output / "ids.npy", ids)
        np.save(args.output / "source_rows.npy", rows)
        save_progress(
            progress_path,
            {
                "preregistration_sha256": sha256(args.prereg),
                "source_rows_sha256": rows_hash,
                "shape": [len(rows), hidden_size],
                "completed_rows": 0,
            },
        )

    prompt_prefix = str(model_spec["prompt_prefix"])
    with torch.inference_mode():
        for start in tqdm(
            range(start_row, len(texts), batch_size),
            desc="Qwen powered hidden states",
            initial=start_row // batch_size,
            total=(len(texts) + batch_size - 1) // batch_size,
        ):
            batch = [
                prompt_prefix + text
                for text in texts[start : start + batch_size]
            ]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(device) for key, value in encoded.items()
            }
            outputs = model.model(
                **encoded,
                use_cache=False,
                return_dict=True,
            )
            hidden = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            stop = min(start + batch_size, len(texts))
            features[start:stop] = pooled.float().cpu().numpy()
            if stop == len(texts) or stop % (batch_size * 32) == 0:
                features.flush()
                save_progress(
                    progress_path,
                    {
                        "preregistration_sha256": sha256(args.prereg),
                        "source_rows_sha256": rows_hash,
                        "shape": [len(rows), hidden_size],
                        "completed_rows": stop,
                    },
                )
    features.flush()
    del features

    group_counts = {
        f"role={role_code},y={label},s={group}": int(
            np.sum(
                (role == role_code)
                & (target == label)
                & (source == group)
            )
        )
        for role_code in (
            ROLE_CONSTRUCTION,
            ROLE_REFERENCE,
            ROLE_TARGET,
        )
        for label in (0, 1)
        for group in (0, 1)
    }
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "CivilComments-Qwen2.5-powered-temporal-confirmation",
        "n_examples": len(rows),
        "dimension": hidden_size,
        "arrays": {
            "z": "z.npy",
            "y": "y.npy",
            "s": "s.npy",
            "split": "split.npy",
        },
        "auxiliary_arrays": {
            "ids": "ids.npy",
            "source_rows": "source_rows.npy",
        },
        "preregistration_sha256": sha256(args.prereg),
        "source_csv_sha256": prereg["source_csv"]["sha256"],
        "source_rows_sha256": rows_hash,
        "pilot_partition": "integer dataset id modulo 4 equals 0",
        "confirmation_partition": "integer dataset id modulo 4 is nonzero",
        "temporal_roles": temporal,
        "invalid_confirmation_timestamp_count": invalid_timestamp_count,
        "model": model_spec["model_id"],
        "model_revision": model_spec["revision"],
        "model_layers": int(model.config.num_hidden_layers),
        "representation": "layer28_mean",
        "hidden_layer": 28,
        "pooling": "mean",
        "prompt_prefix": prompt_prefix,
        "max_length": max_length,
        "padding_side": "right",
        "dtype": "float32",
        "device": device.type,
        "group_counts": group_counts,
        "source_concept": (
            "identity_any >= 0.5 (identity mention, not author identity)"
        ),
        "target": "toxicity >= 0.5",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    progress_path.unlink()
    print(json.dumps({"output": str(args.output), **manifest}, indent=2))


if __name__ == "__main__":
    main()
