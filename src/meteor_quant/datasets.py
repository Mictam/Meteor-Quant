from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import polars as pl

EXPECTED_FILES = (
    "2021-02-23_2024-08-28_BTCUSDT_1s.csv",
    "2024-08-29_2026-07-12_BTCUSDT_1s.csv",
)

COLUMN_ALIASES = {
    "Open Time": "open_time",
    "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "Close Time": "close_time",
    "Quote Asset Volume": "quote_asset_volume",
    "Number of Trades": "number_of_trades",
    "Taker Buy Base Asset Volume": "taker_buy_base_volume",
    "Taker Buy Quote Asset": "taker_buy_quote_volume",
    "Taker Buy Quote Asset Volume": "taker_buy_quote_volume",
}

NORMALIZED_COLUMN_ALIASES = {
    " ".join(name.lstrip("\ufeff").strip().split()).casefold(): alias
    for name, alias in COLUMN_ALIASES.items()
}

CANONICAL_COLUMNS = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
)


@dataclass(slots=True, frozen=True)
class DatasetDescriptor:
    key: str
    symbol: str
    base_timeframe_seconds: int
    raw_files: tuple[str, ...]
    canonical_path: str
    prepared: bool
    row_count: int | None = None
    first_timestamp: int | None = None
    last_timestamp: int | None = None
    source_bytes: int = 0
    prepared_bytes: int = 0
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetCatalog:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir.resolve()
        self.raw_dir = self.data_dir / "raw"
        self.cache_dir = self.data_dir / "cache"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def list_datasets(self) -> list[DatasetDescriptor]:
        return [self.describe("btcusdt_1s")]

    def describe(self, key: str) -> DatasetDescriptor:
        if key != "btcusdt_1s":
            raise KeyError(f"unknown dataset: {key}")
        raw = self._resolve_expected_files()
        canonical = self.canonical_path(key)
        metadata = self._read_metadata(key)
        return DatasetDescriptor(
            key=key,
            symbol="BTCUSDT",
            base_timeframe_seconds=1,
            raw_files=tuple(str(path) for path in raw),
            canonical_path=str(canonical),
            prepared=canonical.exists() and self._metadata_is_valid(metadata),
            row_count=metadata.get("row_count") if metadata else None,
            first_timestamp=metadata.get("first_timestamp") if metadata else None,
            last_timestamp=metadata.get("last_timestamp") if metadata else None,
            source_bytes=sum(path.stat().st_size for path in raw if path.exists()),
            prepared_bytes=canonical.stat().st_size if canonical.exists() else 0,
            updated_at=metadata.get("updated_at") if metadata else None,
        )

    def canonical_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.parquet"

    def prepare(self, key: str = "btcusdt_1s", force: bool = False) -> DatasetDescriptor:
        descriptor = self.describe(key)
        paths = [Path(item) for item in descriptor.raw_files]
        if len(paths) != len(EXPECTED_FILES):
            missing = [
                name for name in EXPECTED_FILES if not any(path.name == name for path in paths)
            ]
            raise FileNotFoundError(
                "missing Binance CSV files under data/ or data/raw/: " + ", ".join(missing)
            )
        canonical = self.canonical_path(key)
        metadata_path = self._metadata_path(key)
        signature = self._source_signature(paths)
        existing = self._read_metadata(key)
        if (
            not force
            and canonical.exists()
            and existing is not None
            and self._metadata_is_valid(existing)
            and existing.get("source_signature") == signature
        ):
            return self.describe(key)

        lazy = self.scan_raw(paths)
        temp = canonical.with_suffix(".parquet.tmp")
        temp.unlink(missing_ok=True)
        lazy.sink_parquet(
            temp,
            compression="zstd",
            compression_level=3,
            statistics=True,
            maintain_order=True,
        )
        stats = (
            pl.scan_parquet(temp)
            .select(
                pl.len().alias("row_count"),
                pl.col("timestamp").min().alias("first_timestamp"),
                pl.col("timestamp").max().alias("last_timestamp"),
            )
            .collect()
            .row(0, named=True)
        )
        if not self._stats_are_valid(stats):
            temp.unlink(missing_ok=True)
            if not self._metadata_is_valid(existing):
                canonical.unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)
            raise ValueError(
                "dataset preparation produced fewer than two valid bars. "
                "The CSV files were found, but their timestamp or OHLCV values could not be "
                "normalized. This build accepts numeric epoch timestamps and common UTC date-time "
                "strings in either 'Open Time' or 'Date'."
            )

        temp.replace(canonical)
        metadata = {
            "row_count": int(stats["row_count"]),
            "first_timestamp": int(stats["first_timestamp"]),
            "last_timestamp": int(stats["last_timestamp"]),
            "source_signature": signature,
            "source_files": [str(path) for path in paths],
            "updated_at": datetime.now(UTC).isoformat(),
            "schema_version": 2,
        }
        metadata_temp = metadata_path.with_suffix(".json.tmp")
        metadata_temp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        metadata_temp.replace(metadata_path)
        return self.describe(key)

    def scan(
        self,
        key: str,
        *,
        timeframe_seconds: int = 1,
        start_timestamp: int | None = None,
        end_timestamp: int | None = None,
    ) -> pl.LazyFrame:
        if timeframe_seconds <= 0:
            raise ValueError("timeframe_seconds must be positive")
        descriptor = self.describe(key)
        if descriptor.prepared:
            frame = pl.scan_parquet(descriptor.canonical_path)
        else:
            raw = [Path(path) for path in descriptor.raw_files]
            if len(raw) != len(EXPECTED_FILES):
                raise FileNotFoundError(
                    "dataset is not prepared and expected CSV files are missing"
                )
            frame = self.scan_raw(raw)
        if start_timestamp is not None:
            frame = frame.filter(pl.col("timestamp") >= start_timestamp)
        if end_timestamp is not None:
            frame = frame.filter(pl.col("timestamp") <= end_timestamp)
        if timeframe_seconds > 1:
            frame = self._resample(frame, timeframe_seconds)
        return frame

    @staticmethod
    def scan_raw(paths: list[Path]) -> pl.LazyFrame:
        scans: list[pl.LazyFrame] = []
        for path in sorted(paths):
            separator = DatasetCatalog._detect_separator(path)
            scan = pl.scan_csv(
                path,
                separator=separator,
                infer_schema_length=10_000,
                ignore_errors=False,
                null_values=["", "null", "NULL", "NaN"],
                encoding="utf8-lossy",
            )
            rename = DatasetCatalog._column_rename_map(scan.collect_schema().names())
            scan = scan.rename(rename)
            missing = {"open_time", "open", "high", "low", "close", "volume"} - set(
                scan.collect_schema().names()
            )
            if missing:
                raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
            optional = {
                "quote_asset_volume": 0.0,
                "number_of_trades": 0,
                "taker_buy_base_volume": 0.0,
                "taker_buy_quote_volume": 0.0,
            }
            expressions: list[pl.Expr] = []
            names = set(scan.collect_schema().names())
            for name, default in optional.items():
                if name not in names:
                    expressions.append(pl.lit(default).alias(name))
            scan = scan.with_columns(expressions)
            timestamp_candidates = [DatasetCatalog._timestamp_expression("open_time")]
            if "date" in names:
                timestamp_candidates.append(DatasetCatalog._timestamp_expression("date"))
            timestamp_seconds = pl.coalesce(timestamp_candidates).alias("timestamp")
            scan = scan.with_columns(
                timestamp_seconds,
                pl.col("open").cast(pl.Float64, strict=False),
                pl.col("high").cast(pl.Float64, strict=False),
                pl.col("low").cast(pl.Float64, strict=False),
                pl.col("close").cast(pl.Float64, strict=False),
                pl.col("volume").cast(pl.Float64, strict=False),
                pl.col("quote_asset_volume").cast(pl.Float64, strict=False).fill_null(0.0),
                pl.col("number_of_trades").cast(pl.Int64, strict=False).fill_null(0),
                pl.col("taker_buy_base_volume").cast(pl.Float64, strict=False).fill_null(0.0),
                pl.col("taker_buy_quote_volume").cast(pl.Float64, strict=False).fill_null(0.0),
            ).select(CANONICAL_COLUMNS)
            scans.append(scan)
        return pl.concat(scans, how="vertical_relaxed").filter(
            pl.all_horizontal(
                pl.col("timestamp").is_not_null(),
                pl.col("open").is_not_null(),
                pl.col("high").is_not_null(),
                pl.col("low").is_not_null(),
                pl.col("close").is_not_null(),
                pl.col("volume").is_not_null(),
            )
        )

    @staticmethod
    def _column_rename_map(names: list[str]) -> dict[str, str]:
        rename: dict[str, str] = {}
        for original in names:
            normalized = " ".join(original.lstrip("\ufeff").strip().split()).casefold()
            alias = NORMALIZED_COLUMN_ALIASES.get(normalized)
            if alias is not None:
                rename[original] = alias
        return rename

    @staticmethod
    def _detect_separator(path: Path) -> str:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            sample = handle.read(16_384)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            return dialect.delimiter
        except csv.Error:
            first_line = next((line for line in sample.splitlines() if line.strip()), "")
            counts = {delimiter: first_line.count(delimiter) for delimiter in ",;\t|"}
            return max(counts, key=lambda delimiter: counts[delimiter]) if any(counts.values()) else ","

    @staticmethod
    def _timestamp_expression(column: str) -> pl.Expr:
        text = pl.col(column).cast(pl.String, strict=False).str.strip_chars()
        numeric = text.cast(pl.Float64, strict=False).cast(pl.Int64, strict=False)
        numeric_seconds = (
            pl.when(numeric >= 10_000_000_000_000_000)
            .then(numeric // 1_000_000_000)
            .when(numeric >= 10_000_000_000_000)
            .then(numeric // 1_000_000)
            .when(numeric >= 10_000_000_000)
            .then(numeric // 1_000)
            .otherwise(numeric)
        )
        date_formats = (
            "%Y-%m-%d %H:%M:%S%.f",
            "%Y-%m-%dT%H:%M:%S%.fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S%.f",
            "%Y/%m/%d %H:%M:%S",
            "%d.%m.%Y %H:%M:%S%.f",
            "%d.%m.%Y %H:%M:%S",
            "%Y-%m-%d",
        )
        parsed_seconds = [
            text.str.strptime(pl.Datetime("us"), format=date_format, strict=False)
            .dt.epoch("s")
            for date_format in date_formats
        ]
        offset_seconds = [
            text.str.strptime(
                pl.Datetime("us", time_zone="UTC"),
                format=date_format,
                strict=False,
            ).dt.epoch("s")
            for date_format in (
                "%Y-%m-%dT%H:%M:%S%.f%#z",
                "%Y-%m-%dT%H:%M:%S%#z",
            )
        ]
        return pl.coalesce([numeric_seconds, *parsed_seconds, *offset_seconds])

    @staticmethod
    def _resample(frame: pl.LazyFrame, timeframe_seconds: int) -> pl.LazyFrame:
        every = f"{timeframe_seconds}s"
        return (
            frame.with_columns(pl.from_epoch("timestamp", time_unit="s").alias("datetime"))
            .group_by_dynamic("datetime", every=every, period=every, label="left", closed="left")
            .agg(
                pl.col("timestamp").first().alias("timestamp"),
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
                pl.col("quote_asset_volume").sum().alias("quote_asset_volume"),
                pl.col("number_of_trades").sum().alias("number_of_trades"),
                pl.col("taker_buy_base_volume").sum().alias("taker_buy_base_volume"),
                pl.col("taker_buy_quote_volume").sum().alias("taker_buy_quote_volume"),
            )
            .drop("datetime")
            .filter(pl.col("open").is_not_null() & pl.col("close").is_not_null())
        )

    def _resolve_expected_files(self) -> list[Path]:
        candidates: dict[str, Path] = {}
        roots = (self.data_dir, self.raw_dir)
        for root in roots:
            if not root.exists():
                continue
            for name in EXPECTED_FILES:
                direct = root / name
                if direct.exists():
                    candidates[name] = direct.resolve()
            for path in root.rglob("*.csv"):
                if path.name in EXPECTED_FILES:
                    candidates[path.name] = path.resolve()
        return [candidates[name] for name in EXPECTED_FILES if name in candidates]

    @staticmethod
    def _source_signature(paths: list[Path]) -> str:
        digest = hashlib.sha256()
        for path in paths:
            stat = path.stat()
            digest.update(str(path.resolve()).encode())
            digest.update(str(stat.st_size).encode())
            digest.update(str(stat.st_mtime_ns).encode())
        return digest.hexdigest()

    def _metadata_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.metadata.json"

    def _read_metadata(self, key: str) -> dict[str, Any] | None:
        path = self._metadata_path(key)
        if not path.exists():
            return None
        try:
            return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            return None

    @staticmethod
    def _stats_are_valid(stats: dict[str, Any]) -> bool:
        row_count = stats.get("row_count")
        first = stats.get("first_timestamp")
        last = stats.get("last_timestamp")
        return (
            isinstance(row_count, int)
            and row_count >= 2
            and isinstance(first, int)
            and isinstance(last, int)
            and first < last
        )

    @classmethod
    def _metadata_is_valid(cls, metadata: dict[str, Any] | None) -> bool:
        return metadata is not None and cls._stats_are_valid(metadata)


def stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=os.fspath)
    return hashlib.sha256(encoded.encode()).hexdigest()
