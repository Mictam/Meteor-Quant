from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from meteor_quant import __version__
from meteor_quant.api import PaperStartRequest, create_app
from meteor_quant.cli import build_parser
from meteor_quant.config import Settings
from meteor_quant.strategies.registry import StrategyRegistry


def test_public_identity_and_cli() -> None:
    assert __version__ == "1.0.0"
    assert build_parser().prog == "meteor"


def test_packaged_frontend_is_available() -> None:
    index = Settings().frontend_dist / "index.html"
    assert index.is_file(), index


def test_showcase_registry_excludes_experimental_pullback(tmp_path: Path) -> None:
    registry = StrategyRegistry(tmp_path / "plugins")
    keys = {item["key"] for item in registry.list_metadata()["strategies"]}
    assert "btc_trend_pullback" not in keys
    assert {"sma_cross", "rsi_mean_reversion", "wavetrend"} <= keys


def test_paper_defaults_are_explicit_and_costed() -> None:
    request = PaperStartRequest(strategy_key="sma_cross")
    assert request.fee_bps == 10.0
    assert request.slippage_bps == 1.5


@pytest.mark.asyncio
async def test_health_declares_paper_only(tmp_path: Path) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("ok", encoding="utf-8")
    settings = Settings(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        user_strategy_dir=tmp_path / "plugins",
        frontend_dist=frontend,
    )
    transport = httpx.ASGITransport(app=create_app(settings))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["paper_only"] is True
    assert payload["version"] == "1.0.0"
