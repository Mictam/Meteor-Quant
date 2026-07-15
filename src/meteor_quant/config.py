from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = (
    SOURCE_PROJECT_ROOT
    if (SOURCE_PROJECT_ROOT / "pyproject.toml").is_file()
    else Path.cwd().resolve()
)
SOURCE_FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
PACKAGED_FRONTEND_DIST = PACKAGE_ROOT / "static"
DEFAULT_FRONTEND_DIST = (
    SOURCE_FRONTEND_DIST if SOURCE_FRONTEND_DIST.exists() else PACKAGED_FRONTEND_DIST
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="METEOR_", env_file=".env", extra="ignore")

    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    user_strategy_dir: Path = PROJECT_ROOT / "user_strategies"
    frontend_dist: Path = DEFAULT_FRONTEND_DIST
    rust_engine_path: Path | None = None
    symbol: str = "BTC/USD"
    binance_symbol: str = "BTCUSDT"
    kraken_rest_pair: str = "XBTUSD"
    kraken_ws_url: str = "wss://ws.kraken.com/v2"
    kraken_rest_url: str = "https://api.kraken.com"
    kraken_trade_bootstrap_max_pages: int = 120
    kraken_trade_bootstrap_page_delay_seconds: float = 0.15

    @property
    def database_path(self) -> Path:
        return self.data_dir / "meteor_quant.sqlite3"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def result_dir(self) -> Path:
        return self.data_dir / "results"


settings = Settings()
