from __future__ import annotations

import io

from run_mosaic_acs_scalar_confirmation_transport import (
    read_state_specific_csv,
)


def test_state_specific_reader_restores_locked_fips() -> None:
    source = io.BytesIO(b"PUMA,SEX,RELP\n101,1,0\n102,2,1\n")
    frame = read_state_specific_csv(
        source,
        {"PUMA", "SEX", "RELP", "ST"},
        state="FL",
    )
    assert set(frame.columns) == {"PUMA", "RELP", "SEX", "ST"}
    assert frame["ST"].tolist() == [12, 12]


def test_state_specific_reader_preserves_observed_state() -> None:
    source = io.BytesIO(b"PUMA,SEX,RELP,ST\n101,1,0,12\n")
    frame = read_state_specific_csv(
        source,
        {"PUMA", "SEX", "RELP", "ST"},
        state="FL",
    )
    assert frame["ST"].tolist() == [12]
