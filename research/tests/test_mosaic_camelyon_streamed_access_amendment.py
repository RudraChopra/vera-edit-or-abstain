from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_access_amendment_preserves_scientific_protocol() -> None:
    original = (
        ROOT
        / "research/mosaic/"
        "prepare_camelyon_streamed_confirmation_store.py"
    ).read_text(encoding="utf-8")
    recovery = (
        ROOT
        / "research/mosaic/"
        "prepare_camelyon_streamed_confirmation_store_v2.py"
    ).read_text(encoding="utf-8")
    assert "validate_lock(args.prereg, args.metadata)" in recovery
    assert "selected_image_ids_sha256" in recovery
    assert "resnet18(weights=weights)" in original
    assert "resnet18(weights=weights)" in recovery
    assert "model.fc = torch.nn.Identity()" in original
    assert "model.fc = torch.nn.Identity()" in recovery
    assert 'config={"threads": "1"}' in recovery
    assert "completed_shards" in recovery
