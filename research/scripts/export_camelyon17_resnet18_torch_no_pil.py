"""Export Camelyon17-WILDS frozen ResNet-18 embeddings without PIL/torchvision.

This script exists because the local official runtime can stage Camelyon17 but
its PIL/torchvision import path is brittle. It keeps the benchmark route
auditable by using a small standard-library PNG decoder plus a manual ResNet-18
encoder loaded from the official torchvision ResNet-18 checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
import time
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
DEFAULT_METADATA = ARTIFACT_DIR / "camelyon17_wilds_metadata.csv"
DEFAULT_OUT = ARTIFACT_DIR / "camelyon17_resnet18_torch_smoke_embeddings.csv"
DEFAULT_REPORT = ARTIFACT_DIR / "camelyon17_resnet18_torch_smoke_report.json"
DEFAULT_WEIGHTS = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "resnet18-f37072fd.pth"

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
EXPECTED_CAMELYON17_EXAMPLES = 455_954
FEATURE_COUNT = 512


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paeth_predictor(left: int, up: int, up_left: int) -> int:
    predictor = left + up - up_left
    pa = abs(predictor - left)
    pb = abs(predictor - up)
    pc = abs(predictor - up_left)
    if pa <= pb and pa <= pc:
        return left
    if pb <= pc:
        return up
    return up_left


def png_chunks(data: bytes) -> Iterable[tuple[bytes, bytes]]:
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("not a PNG file")
    offset = len(PNG_SIGNATURE)
    while offset + 8 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        chunk = data[offset + 8 : offset + 8 + length]
        yield kind, chunk
        offset += length + 12
        if kind == b"IEND":
            break


def unfilter_png(raw: bytes, width: int, height: int, channels: int) -> bytes:
    row_bytes = width * channels
    rows: list[bytearray] = []
    offset = 0
    for row_index in range(height):
        filter_type = raw[offset]
        offset += 1
        row = bytearray(raw[offset : offset + row_bytes])
        offset += row_bytes
        prior = rows[row_index - 1] if row_index else bytearray(row_bytes)

        for idx in range(row_bytes):
            left = row[idx - channels] if idx >= channels else 0
            up = prior[idx]
            up_left = prior[idx - channels] if idx >= channels else 0
            if filter_type == 0:
                correction = 0
            elif filter_type == 1:
                correction = left
            elif filter_type == 2:
                correction = up
            elif filter_type == 3:
                correction = (left + up) // 2
            elif filter_type == 4:
                correction = paeth_predictor(left, up, up_left)
            else:
                raise ValueError(f"unsupported PNG filter type: {filter_type}")
            row[idx] = (row[idx] + correction) & 0xFF
        rows.append(row)
    return b"".join(rows)


def read_png_rgb(path: Path) -> tuple[bytes, int, int]:
    ihdr: tuple[int, int, int, int, int, int, int] | None = None
    idat_parts: list[bytes] = []
    for kind, chunk in png_chunks(path.read_bytes()):
        if kind == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", chunk)
        elif kind == b"IDAT":
            idat_parts.append(chunk)
    if ihdr is None:
        raise ValueError(f"{path} has no IHDR chunk")

    width, height, bit_depth, color_type, compression, filter_method, interlace = ihdr
    if bit_depth != 8 or compression != 0 or filter_method != 0 or interlace != 0:
        raise ValueError(
            f"{path} must be non-interlaced 8-bit PNG; got bit_depth={bit_depth}, "
            f"compression={compression}, filter={filter_method}, interlace={interlace}"
        )

    channels_by_color = {0: 1, 2: 3, 4: 2, 6: 4}
    if color_type not in channels_by_color:
        raise ValueError(f"{path} has unsupported PNG color type: {color_type}")
    channels = channels_by_color[color_type]
    decoded = unfilter_png(zlib.decompress(b"".join(idat_parts)), width, height, channels)

    if color_type == 2:
        rgb = decoded
    elif color_type == 6:
        rgb = bytes(value for idx, value in enumerate(decoded) if idx % 4 != 3)
    elif color_type == 0:
        rgb = bytes(channel for value in decoded for channel in (value, value, value))
    else:
        rgb = bytes(
            channel
            for idx in range(0, len(decoded), 2)
            for channel in (decoded[idx], decoded[idx], decoded[idx])
        )
    return rgb, height, width


def import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:
        raise SystemExit(
            "This exporter needs torch. Use the clean runtime created for FARO, "
            "for example `/tmp/faro-torch-venv/bin/python`."
        ) from exc
    return torch, nn, functional


torch, nn, F = import_torch()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample: nn.Module | None = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):  # type: ignore[no-untyped-def]
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet18Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, blocks=2)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = [BasicBlock(self.inplanes, planes, stride)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):  # type: ignore[no-untyped-def]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


def load_encoder(weights_path: Path, device: "torch.device") -> tuple[ResNet18Encoder, list[str], list[str]]:
    model = ResNet18Encoder()
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"{weights_path} did not contain a ResNet state dict")
    filtered = {key.removeprefix("module."): value for key, value in state.items() if not key.endswith("fc.weight") and not key.endswith("fc.bias")}
    load_result = model.load_state_dict(filtered, strict=False)
    model.to(device)
    model.eval()
    return model, list(load_result.missing_keys), list(load_result.unexpected_keys)


def resolve_device(requested: str) -> "torch.device":
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def preprocess_image(rgb: bytes, height: int, width: int, short_size: int, crop_size: int):
    tensor = torch.frombuffer(bytearray(rgb), dtype=torch.uint8).to(dtype=torch.float32)
    tensor = tensor.view(height, width, 3).permute(2, 0, 1).unsqueeze(0) / 255.0
    scale = short_size / float(min(height, width))
    new_height = max(crop_size, int(math.floor(height * scale + 0.5)))
    new_width = max(crop_size, int(math.floor(width * scale + 0.5)))
    tensor = F.interpolate(tensor, size=(new_height, new_width), mode="bilinear", align_corners=False)
    top = max(0, (new_height - crop_size) // 2)
    left = max(0, (new_width - crop_size) // 2)
    tensor = tensor[:, :, top : top + crop_size, left : left + crop_size]
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    return (tensor - mean) / std


def encode_batch(
    model: ResNet18Encoder,
    rows: list[dict[str, str]],
    device: "torch.device",
    short_size: int,
    crop_size: int,
) -> list[list[float]]:
    tensors = []
    for row in rows:
        rgb, height, width = read_png_rgb(Path(row["image_path"]))
        tensors.append(preprocess_image(rgb, height, width, short_size, crop_size))
    batch = torch.cat(tensors, dim=0).to(device)
    with torch.no_grad():
        embeddings = model(batch).detach().cpu()
    return [[float(value) for value in embedding] for embedding in embeddings]


def required_metadata_columns(fieldnames: list[str] | None) -> None:
    required = {"id", "split", "y", "s", "image_path"}
    available = set(fieldnames or [])
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"metadata file missing required columns: {missing}")


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames and "id" in reader.fieldnames:
            for row in reader:
                ids.add(row["id"])
    return ids


def selected_rows(metadata_path: Path, max_examples_per_split: int) -> Iterable[dict[str, str]]:
    counts: Counter[str] = Counter()
    with metadata_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_metadata_columns(reader.fieldnames)
        for row in reader:
            split = row["split"]
            if max_examples_per_split > 0 and counts[split] >= max_examples_per_split:
                continue
            counts[split] += 1
            yield row


def summarize_trace_table(path: Path) -> tuple[int, dict[str, int], int]:
    if not path.exists():
        return 0, {}, 0
    splits: Counter[str] = Counter()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        feature_count = len([name for name in (reader.fieldnames or []) if name.startswith("embedding_")])
        total = 0
        for row in reader:
            total += 1
            splits[row.get("split", "")] += 1
    return total, dict(sorted(splits.items())), feature_count


def write_embeddings(
    metadata_path: Path,
    out_path: Path,
    model: ResNet18Encoder,
    device: "torch.device",
    batch_size: int,
    max_examples_per_split: int,
    resume: bool,
    progress_every: int,
    short_size: int,
    crop_size: int,
) -> dict[str, object]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = completed_ids(out_path) if resume else set()
    mode = "a" if resume and out_path.exists() else "w"
    fieldnames = ["id", "split", "y", "s"] + [f"embedding_{idx:04d}" for idx in range(FEATURE_COUNT)]
    selected_count = 0
    skipped_completed = 0
    rows_written = 0
    written_splits: Counter[str] = Counter()
    started = time.monotonic()

    with out_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()

        batch: list[dict[str, str]] = []
        for row in selected_rows(metadata_path, max_examples_per_split):
            selected_count += 1
            if row["id"] in existing_ids:
                skipped_completed += 1
                continue
            batch.append(row)
            if len(batch) >= batch_size:
                embeddings = encode_batch(model, batch, device, short_size, crop_size)
                for source_row, embedding in zip(batch, embeddings):
                    writer.writerow(
                        {
                            "id": source_row["id"],
                            "split": source_row["split"],
                            "y": source_row["y"],
                            "s": source_row["s"],
                            **{
                                f"embedding_{idx:04d}": f"{value:.9g}"
                                for idx, value in enumerate(embedding)
                            },
                        }
                    )
                    rows_written += 1
                    written_splits[source_row["split"]] += 1
                batch.clear()
                if progress_every > 0 and rows_written and rows_written % progress_every == 0:
                    elapsed = time.monotonic() - started
                    print(f"wrote {rows_written} rows in {elapsed:.1f}s")

        if batch:
            embeddings = encode_batch(model, batch, device, short_size, crop_size)
            for source_row, embedding in zip(batch, embeddings):
                writer.writerow(
                    {
                        "id": source_row["id"],
                        "split": source_row["split"],
                        "y": source_row["y"],
                        "s": source_row["s"],
                        **{
                            f"embedding_{idx:04d}": f"{value:.9g}"
                            for idx, value in enumerate(embedding)
                        },
                    }
                )
                rows_written += 1
                written_splits[source_row["split"]] += 1

    return {
        "selected_count": selected_count,
        "skipped_completed": skipped_completed,
        "rows_written_this_run": rows_written,
        "written_splits_this_run": dict(sorted(written_splits.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-examples-per-split",
        type=int,
        default=5,
        help="Safety cap for smoke exports; use 0 for all rows.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=4096)
    parser.add_argument("--short-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_examples_per_split < 0:
        raise ValueError("--max-examples-per-split must be non-negative")
    if not args.metadata.exists():
        raise FileNotFoundError(args.metadata)
    if not args.weights.exists():
        raise FileNotFoundError(args.weights)

    started = time.monotonic()
    device = resolve_device(args.device)
    weights_sha = sha256_file(args.weights)
    model, missing_keys, unexpected_keys = load_encoder(args.weights, device)
    export_report = write_embeddings(
        args.metadata,
        args.out,
        model,
        device,
        args.batch_size,
        args.max_examples_per_split,
        args.resume,
        args.progress_every,
        args.short_size,
        args.crop_size,
    )
    total_rows, split_summary, feature_count = summarize_trace_table(args.out)
    embedding_sha = sha256_file(args.out)
    full_export = total_rows == EXPECTED_CAMELYON17_EXAMPLES and args.max_examples_per_split == 0
    claim_grade_embedding = (
        feature_count == FEATURE_COUNT
        and total_rows > 0
        and not missing_keys
        and not unexpected_keys
        and weights_sha == "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
    )

    report = {
        "name": "FARO Camelyon17 no-PIL torch ResNet18 export",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_mode": "direct_camelyon17_wilds_metadata_csv",
        "encoder_family": "manual_resnet18_imagenet_frozen_no_pil",
        "metadata_path": str(args.metadata),
        "embedding_table": str(args.out),
        "embedding_sha256": embedding_sha,
        "weights_path": str(args.weights),
        "weights_sha256": weights_sha,
        "device": str(device),
        "batch_size": args.batch_size,
        "max_examples_per_split": args.max_examples_per_split,
        "short_size": args.short_size,
        "crop_size": args.crop_size,
        "sample_count": total_rows,
        "splits": split_summary,
        "feature_count": feature_count,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "full_dataset_embedding_export": full_export,
        "expected_full_dataset_examples": EXPECTED_CAMELYON17_EXAMPLES,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "claim_grade_embedding": claim_grade_embedding,
        "claim_grade_benchmark_row": False,
        "claim_boundary": (
            "Claim-grade frozen encoder export only. A benchmark row becomes "
            "claim-ready only after all Camelyon17 rows are embedded, official "
            "receipts are written, and paired statistics pass."
        ),
        "export_report": export_report,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("FARO Camelyon17 no-PIL ResNet-18 export complete")
    print(f"embedding_table={args.out}")
    print(f"report={args.report}")
    print(f"sample_count={total_rows}")
    print(f"splits={split_summary}")
    print(f"claim_grade_embedding={str(claim_grade_embedding).lower()}")
    print("claim_grade_benchmark_row=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
