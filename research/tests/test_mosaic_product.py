import json

import numpy as np

from mosaic_certified_release import AnytimeAuditMonitor, Mosaic
from mosaic_certified_release.report import verify_report


def test_anytime_monitor_updates_without_alpha_spending() -> None:
    monitor = AnytimeAuditMonitor(2, 3, failure_probability=0.05)
    first = monitor.update(0, [0, 1, 1, 2])
    second = monitor.update(1, [0, 0, 1, 2])
    assert first.update_count == 4
    assert second.update_count == 8
    assert len(second.regions) == 2


def test_report_generator_handles_uncertified_model(tmp_path) -> None:
    model = Mosaic()
    json_path, html_path = model.write_report(tmp_path)
    payload = json.loads(json_path.read_text())
    assert payload["schema"] == "mosaic_certification_report_v1"
    assert payload["certification"] is None
    assert html_path.exists()
    assert verify_report(json_path)["valid_schema"]
