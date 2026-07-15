from __future__ import annotations

import csv
import json
from pathlib import Path

from meteor_quant.datasets import EXPECTED_FILES, DatasetCatalog


def test_discovers_and_prepares_exact_binance_files(data_dir: Path) -> None:
    catalog = DatasetCatalog(data_dir)
    before = catalog.describe("btcusdt_1s")
    assert len(before.raw_files) == 2
    assert not before.prepared

    prepared = catalog.prepare()

    assert prepared.prepared
    assert prepared.row_count == 1_200
    assert prepared.first_timestamp == 1_600_000_000
    assert prepared.last_timestamp == 1_600_001_199
    assert Path(prepared.canonical_path).exists()


def test_resamples_without_loading_python_rows(data_dir: Path) -> None:
    catalog = DatasetCatalog(data_dir)
    catalog.prepare()
    frame = catalog.scan("btcusdt_1s", timeframe_seconds=60).collect()
    assert frame.height in {20, 21}
    assert frame["number_of_trades"].sum() == 12_000
    assert frame["volume"].sum() == 1_200


def test_prepares_text_timestamps_and_normalized_headers(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    headers = [
        "\ufeff Open Time ",
        " Date ",
        " Open ",
        " High ",
        " Low ",
        " Close ",
        " Volume ",
        " Close Time ",
        " Quote Asset Volume ",
        " Number of Trades ",
        " Taker Buy Base Asset Volume ",
        " Taker Buy Quote Asset Volume ",
    ]
    row_offset = 0
    for filename in EXPECTED_FILES:
        with (data / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow(headers)
            for index in range(3):
                second = row_offset + index
                writer.writerow(
                    [
                        f"2021-02-23 00:00:0{second}",
                        "",
                        100 + second,
                        101 + second,
                        99 + second,
                        100.5 + second,
                        1.0,
                        "",
                        100.5,
                        10,
                        0.4,
                        40.2,
                    ]
                )
        row_offset += 3

    prepared = DatasetCatalog(data).prepare()

    assert prepared.prepared
    assert prepared.row_count == 6
    assert prepared.first_timestamp == 1_614_038_400
    assert prepared.last_timestamp == 1_614_038_405


def test_zero_row_metadata_is_not_considered_prepared(data_dir: Path) -> None:
    catalog = DatasetCatalog(data_dir)
    canonical = catalog.canonical_path("btcusdt_1s")
    canonical.write_bytes(b"invalid empty cache")
    metadata_path = data_dir / "cache" / "btcusdt_1s.metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "row_count": 0,
                "first_timestamp": None,
                "last_timestamp": None,
                "source_signature": "stale",
            }
        ),
        encoding="utf-8",
    )

    assert not catalog.describe("btcusdt_1s").prepared

    rebuilt = catalog.prepare()
    assert rebuilt.prepared
    assert rebuilt.row_count == 1_200


def test_uses_date_when_open_time_is_not_numeric_or_datetime(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    headers = [
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
    row_offset = 0
    for filename in EXPECTED_FILES:
        with (data / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            for index in range(3):
                second = row_offset + index
                writer.writerow(
                    [
                        "not-an-epoch",
                        f"2021-02-23 09:20:{20 + second:02d}",
                        100 + second,
                        101 + second,
                        99 + second,
                        100.5 + second,
                        1.0,
                        "",
                        100.5,
                        10,
                        0.4,
                        40.2,
                    ]
                )
        row_offset += 3

    prepared = DatasetCatalog(data).prepare()

    assert prepared.prepared
    assert prepared.row_count == 6
    assert prepared.first_timestamp == 1_614_072_020
    assert prepared.last_timestamp == 1_614_072_025
