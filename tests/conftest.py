from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from meteor_quant.datasets import EXPECTED_FILES

HEADERS = [
    "Open Time",
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Close Time",
    "Quote Asset Volume",
    "Number of Trades",
    "Taker Buy Base Asset Volume",
    "Taker Buy Quote Asset",
]


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    start_ms = 1_600_000_000_000
    for file_index, name in enumerate(EXPECTED_FILES):
        with (data / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADERS)
            for index in range(600):
                offset = file_index * 600 + index
                price = 100.0 + math.sin(offset / 20.0) * 5.0 + offset * 0.01
                timestamp = start_ms + offset * 1_000
                writer.writerow(
                    [
                        timestamp,
                        "",
                        price,
                        price + 1.0,
                        price - 1.0,
                        price + 0.2,
                        1.0,
                        timestamp + 999,
                        price,
                        10,
                        0.4,
                        price * 0.4,
                    ]
                )
    return data
