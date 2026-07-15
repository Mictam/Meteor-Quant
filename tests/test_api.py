from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from meteor_quant.api import create_app
from meteor_quant.config import Settings


@pytest.mark.asyncio
async def test_dataset_and_backtest_api(data_dir: Path, tmp_path: Path) -> None:
    root = tmp_path
    settings = Settings(
        project_root=root,
        data_dir=data_dir,
        user_strategy_dir=Path(__file__).resolve().parents[1] / "user_strategies",
        frontend_dist=root / "missing-frontend",
    )
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/api/health")
        assert health.status_code == 200
        capabilities = await client.get("/api/marketlm/capabilities")
        assert capabilities.status_code == 200
        hybrid = await client.get("/api/markethybrid/capabilities")
        assert hybrid.status_code == 200
        assert "quality_4060ti" in hybrid.json()["model_presets"]
        timesfm = await client.get("/api/timesfm/capabilities")
        assert timesfm.status_code == 200
        assert timesfm.json()["model_id"] == "google/timesfm-2.5-200m-pytorch"
        assert any(item["key"] == "wavetrend" for item in capabilities.json()["indicators"])
        datasets = await client.get("/api/datasets")
        assert datasets.json()["datasets"][0]["prepared"] is False
        prepared = await client.post("/api/datasets/prepare", json={})
        assert prepared.status_code == 200
        response = await client.post(
            "/api/backtests",
            json={
                "strategy_key": "sma_cross",
                "parameters": {"fast": 10, "slow": 30},
                "timeframe_seconds": 5,
                "engine": "python",
                "max_equity_points": 300,
            },
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["metrics"]["bar_count"] == 240
        chart = await client.get(f"/api/backtests/{payload['id']}/chart?max_points=200")
        assert chart.status_code == 200
        assert chart.json()["source_row_count"] == 240

@pytest.mark.asyncio
async def test_paper_api_accepts_subminute_timeframes(
    data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(
        project_root=tmp_path,
        data_dir=data_dir,
        user_strategy_dir=Path(__file__).resolve().parents[1] / "user_strategies",
        frontend_dist=tmp_path / "missing-frontend",
    )
    app = create_app(settings)
    captured: list[object] = []

    async def fake_replace(session: object) -> None:
        captured.append(session)

    monkeypatch.setattr(app.state.services.sessions, "replace", fake_replace)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/api/health")
        assert health.json()["paper_timeframes_seconds"][:4] == [1, 5, 15, 30]
        for interval in (1, 5, 15, 30):
            response = await client.post(
                "/api/paper/start",
                json={
                    "strategy_key": "sma_cross",
                    "parameters": {"fast": 5, "slow": 20},
                    "timeframe_seconds": interval,
                    "bootstrap_bars": 100,
                },
            )
            assert response.status_code == 200, response.text
        unsupported = await client.post(
            "/api/paper/start",
            json={"strategy_key": "sma_cross", "timeframe_seconds": 2},
        )
        assert unsupported.status_code == 422
    assert len(captured) == 4
