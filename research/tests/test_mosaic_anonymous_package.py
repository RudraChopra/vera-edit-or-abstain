from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "maintrack"
    / "mosaic_aaai2027"
    / "build_anonymous_code_package.py"
)


def load_builder():
    spec = importlib.util.spec_from_file_location("mosaic_anonymous_builder", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_files_includes_nested_bridge_receipts(tmp_path, monkeypatch):
    builder = load_builder()
    expected = {
        "research/artifacts/mosaic_bridge_confirmation_receipts_v1/raw.json",
        "research/artifacts/mosaic_bridge_strict_receipts_v1/strict.json",
        "research/artifacts/mosaic_bridge_comparator_receipts_v1/comparator.json",
    }
    for relative in expected:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(builder, "REPOSITORY", tmp_path)
    observed = {
        path.relative_to(tmp_path).as_posix() for path in builder.collect_files()
    }
    assert observed == expected


def test_sanitizer_removes_machine_local_volume_paths():
    builder = load_builder()
    text = (
        "store=/Volumes/Backups/FARO/artifacts/store "
        "user=/Users/example/project"
    )
    sanitized = builder.sanitize_text_copy(text)
    assert "/Volumes/" not in sanitized
    assert "/Users/" not in sanitized
    assert "/data/external/FARO/artifacts/store" in sanitized
