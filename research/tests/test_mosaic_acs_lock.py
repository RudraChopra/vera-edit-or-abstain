from __future__ import annotations

from run_mosaic_real_pilot import DATASETS


def test_acs_dataset_is_registered_as_native_binary_tabular_shift() -> None:
    config = DATASETS["ACSIncome-CA-TX"]
    assert config["target_mode"] == "native_binary"
    assert config["modality"] == "tabular geographic shift"
