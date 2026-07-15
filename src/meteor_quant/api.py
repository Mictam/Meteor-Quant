from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from meteor_quant import __version__
from meteor_quant.config import Settings
from meteor_quant.config import settings as default_settings
from meteor_quant.datasets import DatasetCatalog
from meteor_quant.engine import BacktestConfig, HybridBacktestService
from meteor_quant.events import EventHub
from meteor_quant.live import PaperSession, PaperSessionConfig, PaperSessionManager
from meteor_quant.market import KrakenMarketStream, KrakenRestClient
from meteor_quant.markethybrid.jobs import MarketHybridJobManager
from meteor_quant.markethybrid.schemas import (
    MarketHybridRegisterRequest,
    MarketHybridRunRequest,
)
from meteor_quant.marketlm.jobs import MarketLMJobManager
from meteor_quant.marketlm.schemas import (
    MarketLMRegisterRequest,
    MarketLMRunRequest,
)
from meteor_quant.storage import Database
from meteor_quant.strategies.registry import StrategyRegistry
from meteor_quant.timesfm.runtime import timesfm_capabilities

KRAKEN_INTERVALS_MINUTES = {1, 5, 15, 30, 60, 240, 1440, 10080, 21600}
KRAKEN_PAPER_TIMEFRAMES_SECONDS = {1, 5, 15, 30} | {
    minutes * 60 for minutes in KRAKEN_INTERVALS_MINUTES
}


class DatasetPrepareRequest(BaseModel):
    dataset_key: str = "btcusdt_1s"
    force: bool = False


class BacktestRequest(BaseModel):
    dataset_key: str = "btcusdt_1s"
    strategy_key: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    timeframe_seconds: int = Field(default=60, ge=1, le=86_400)
    start_timestamp: int | None = None
    end_timestamp: int | None = None
    initial_equity: float = Field(default=10_000.0, gt=0)
    fee_bps: float = Field(default=10.0, ge=0)
    slippage_bps: float = Field(default=1.0, ge=0)
    spread_bps: float = Field(default=1.0, ge=0)
    minimum_order_notional: float = Field(default=10.0, ge=0)
    allow_short: bool = False
    max_leverage: float = Field(default=1.0, gt=0, le=10)
    max_equity_points: int = Field(default=10_000, ge=100, le=100_000)
    max_fills_returned: int = Field(default=100_000, ge=100, le=1_000_000)
    engine: Literal["auto", "rust", "python"] = "auto"
    force_signals: bool = False


class PaperStartRequest(BaseModel):
    strategy_key: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    timeframe_seconds: int = Field(default=60, ge=1)
    bootstrap_bars: int = Field(default=500, ge=2, le=20_000)
    initial_equity: float = Field(default=10_000.0, gt=0)
    fee_bps: float = Field(default=10.0, ge=0)
    slippage_bps: float = Field(default=1.5, ge=0)
    minimum_order_notional: float = Field(default=10.0, ge=0)
    allow_short: bool = False
    max_leverage: float = Field(default=1.0, gt=0, le=10)


class AppServices:
    def __init__(self, settings: Settings) -> None:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        settings.cache_dir.mkdir(parents=True, exist_ok=True)
        settings.result_dir.mkdir(parents=True, exist_ok=True)
        self.settings = settings
        self.catalog = DatasetCatalog(settings.data_dir)
        self.marketlm = MarketLMJobManager(settings.project_root, settings.data_dir)
        self.markethybrid = MarketHybridJobManager(settings.project_root, settings.data_dir)
        self.registry = StrategyRegistry(
            settings.user_strategy_dir,
            self.marketlm.registered_dir,
            self.markethybrid.registered_dir,
        )
        self.backtests = HybridBacktestService(
            self.catalog,
            settings.cache_dir,
            settings.result_dir,
            settings.rust_engine_path,
        )
        self.database = Database(settings.database_path)
        self.rest = KrakenRestClient(settings.kraken_rest_url)
        self.hub = EventHub()
        self.sessions = PaperSessionManager()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    services: AppServices = app.state.services
    await services.sessions.stop()


def create_app(application_settings: Settings | None = None) -> FastAPI:
    resolved = application_settings or default_settings
    app = FastAPI(title="Meteor Quant", version=__version__, lifespan=lifespan)
    services = AppServices(resolved)
    app.state.services = services

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "version": __version__,
            "paper_only": True,
            "backtest_engines": ["rust-arrow", "python-arrow"],
            "paper_session": services.sessions.state(),
            "paper_timeframes_seconds": sorted(KRAKEN_PAPER_TIMEFRAMES_SECONDS),
            "marketlm": {
                "torch_available": services.marketlm.torch_available,
                "registered_models": len(services.marketlm.list_registered()),
            },
            "markethybrid": {
                "torch_available": services.markethybrid.torch_available,
                "registered_models": len(services.markethybrid.list_registered()),
            },
            "timesfm": timesfm_capabilities(),
        }

    @app.get("/api/paper/capabilities")
    async def paper_capabilities() -> dict[str, Any]:
        return {
            "timeframes_seconds": sorted(KRAKEN_PAPER_TIMEFRAMES_SECONDS),
            "default_bootstrap_bars": 500,
            "maximum_bootstrap_bars": 20_000,
            "trade_bootstrap_max_pages": resolved.kraken_trade_bootstrap_max_pages,
            "subminute_source": "Kraken REST recent trades + WebSocket v2 trades",
        }

    @app.get("/api/datasets")
    async def datasets() -> dict[str, Any]:
        try:
            return {"datasets": [item.to_dict() for item in services.catalog.list_datasets()]}
        except Exception as exc:
            raise HTTPException(
                500, f"dataset discovery failed: {type(exc).__name__}: {exc}"
            ) from exc

    @app.post("/api/datasets/prepare")
    async def prepare_dataset(request: DatasetPrepareRequest) -> dict[str, Any]:
        try:
            descriptor = await asyncio.to_thread(
                services.catalog.prepare,
                request.dataset_key,
                request.force,
            )
            return descriptor.to_dict()
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                500, f"dataset preparation failed: {type(exc).__name__}: {exc}"
            ) from exc

    @app.get("/api/strategies")
    async def strategies() -> dict[str, Any]:
        return services.registry.list_metadata()

    @app.post("/api/strategies/reload")
    async def reload_strategies() -> dict[str, Any]:
        services.registry.reload()
        return services.registry.list_metadata()

    @app.get("/api/marketlm/capabilities")
    async def marketlm_capabilities() -> dict[str, Any]:
        return services.marketlm.capabilities()

    @app.get("/api/timesfm/capabilities")
    async def get_timesfm_capabilities() -> dict[str, Any]:
        return timesfm_capabilities()

    @app.get("/api/markethybrid/capabilities")
    async def markethybrid_capabilities() -> dict[str, Any]:
        return services.markethybrid.capabilities()

    @app.get("/api/markethybrid/runs")
    async def markethybrid_runs() -> dict[str, Any]:
        return {
            "runs": services.markethybrid.list_runs(),
            "models": services.markethybrid.list_registered(),
        }

    @app.get("/api/markethybrid/runs/{run_id}")
    async def markethybrid_run(run_id: str) -> dict[str, Any]:
        try:
            return services.markethybrid.get(run_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/markethybrid/runs")
    async def start_markethybrid_run(request: MarketHybridRunRequest) -> dict[str, Any]:
        try:
            active_marketlm = [
                item
                for item in services.marketlm.list_runs()
                if item.get("state") in {"queued", "preparing", "training", "stopping"}
            ]
            if active_marketlm:
                raise RuntimeError(
                    f"MarketLM job {active_marketlm[0]['run_id']} is already active; "
                    "finish or stop it before starting MarketHybrid"
                )
            return services.markethybrid.start(request)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                500,
                f"could not start MarketHybrid worker: {type(exc).__name__}: {exc}",
            ) from exc

    @app.post("/api/markethybrid/runs/{run_id}/stop")
    async def stop_markethybrid_run(run_id: str) -> dict[str, Any]:
        try:
            return services.markethybrid.stop(run_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/markethybrid/runs/{run_id}/register")
    async def register_markethybrid_run(
        run_id: str,
        request: MarketHybridRegisterRequest,
    ) -> dict[str, Any]:
        try:
            registered = services.markethybrid.register(run_id, request)
            services.registry.reload()
            return registered
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.delete("/api/markethybrid/models/{model_id}")
    async def unregister_markethybrid_model(model_id: str) -> dict[str, Any]:
        try:
            services.markethybrid.unregister(model_id)
            services.registry.reload()
            return {"ok": True, "model_id": model_id}
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/marketlm/runs")
    async def marketlm_runs() -> dict[str, Any]:
        return {
            "runs": services.marketlm.list_runs(),
            "models": services.marketlm.list_registered(),
        }

    @app.get("/api/marketlm/runs/{run_id}")
    async def marketlm_run(run_id: str) -> dict[str, Any]:
        try:
            return services.marketlm.get(run_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/marketlm/runs")
    async def start_marketlm_run(request: MarketLMRunRequest) -> dict[str, Any]:
        try:
            active_hybrid = [
                item
                for item in services.markethybrid.list_runs()
                if item.get("state") in {"queued", "preparing", "training", "stopping"}
            ]
            if active_hybrid:
                raise RuntimeError(
                    f"MarketHybrid job {active_hybrid[0]['run_id']} is already active; "
                    "finish or stop it before starting MarketLM"
                )
            return services.marketlm.start(request)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                500, f"could not start MarketLM worker: {type(exc).__name__}: {exc}"
            ) from exc

    @app.post("/api/marketlm/runs/{run_id}/stop")
    async def stop_marketlm_run(run_id: str) -> dict[str, Any]:
        try:
            return services.marketlm.stop(run_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/marketlm/runs/{run_id}/register")
    async def register_marketlm_run(
        run_id: str,
        request: MarketLMRegisterRequest,
    ) -> dict[str, Any]:
        try:
            registered = services.marketlm.register(run_id, request)
            services.registry.reload()
            return registered
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.delete("/api/marketlm/models/{model_id}")
    async def unregister_marketlm_model(model_id: str) -> dict[str, Any]:
        try:
            services.marketlm.unregister(model_id)
            services.registry.reload()
            return {"ok": True, "model_id": model_id}
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/backtests")
    async def run_backtest(request: BacktestRequest) -> dict[str, Any]:
        if (
            request.start_timestamp is not None
            and request.end_timestamp is not None
            and request.start_timestamp >= request.end_timestamp
        ):
            raise HTTPException(422, "start_timestamp must be smaller than end_timestamp")
        try:
            strategy = services.registry.create(request.strategy_key, request.parameters)
            required_timeframe = strategy.required_timeframe_seconds
            if required_timeframe is not None and required_timeframe != request.timeframe_seconds:
                raise ValueError(
                    f"strategy {strategy.name} requires {required_timeframe}-second bars; "
                    f"the backtest requested {request.timeframe_seconds}-second bars"
                )
            prepared = await asyncio.to_thread(
                services.backtests.prepare_signals,
                dataset_key=request.dataset_key,
                strategy=strategy,
                timeframe_seconds=request.timeframe_seconds,
                start_timestamp=request.start_timestamp,
                end_timestamp=request.end_timestamp,
                force=request.force_signals,
            )
            result = await asyncio.to_thread(
                services.backtests.run,
                prepared=prepared,
                strategy_name=strategy.name,
                config=BacktestConfig(
                    initial_equity=request.initial_equity,
                    fee_bps=request.fee_bps,
                    slippage_bps=request.slippage_bps,
                    spread_bps=request.spread_bps,
                    minimum_order_notional=request.minimum_order_notional,
                    allow_short=request.allow_short,
                    max_leverage=request.max_leverage,
                    max_equity_points=request.max_equity_points,
                    max_fills_returned=request.max_fills_returned,
                ),
                engine=request.engine,
            )
            return result
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(500, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(500, f"backtest failed: {type(exc).__name__}: {exc}") from exc

    @app.get("/api/backtests/{result_id}")
    async def get_backtest(result_id: str) -> dict[str, Any]:
        try:
            return services.backtests.load_result(result_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/backtests/{result_id}/chart")
    async def backtest_chart(result_id: str, max_points: int = 5_000) -> dict[str, Any]:
        try:
            result = services.backtests.load_result(result_id)
            return await asyncio.to_thread(
                services.backtests.chart_payload,
                result["signal_cache_key"],
                max_points,
            )
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(500, f"chart loading failed: {type(exc).__name__}: {exc}") from exc

    @app.post("/api/paper/start")
    async def start_paper(request: PaperStartRequest) -> dict[str, Any]:
        if request.timeframe_seconds not in KRAKEN_PAPER_TIMEFRAMES_SECONDS:
            raise HTTPException(
                422,
                "Kraken paper timeframe must be one of "
                f"{sorted(KRAKEN_PAPER_TIMEFRAMES_SECONDS)} seconds",
            )
        try:
            strategy = services.registry.create(request.strategy_key, request.parameters)
            required_timeframe = strategy.required_timeframe_seconds
            if required_timeframe is not None and required_timeframe != request.timeframe_seconds:
                raise ValueError(
                    f"strategy {strategy.name} requires {required_timeframe}-second bars; "
                    f"the paper session requested {request.timeframe_seconds}-second bars"
                )
            if request.timeframe_seconds >= 60 and strategy.minimum_bars > 719:
                raise ValueError(
                    f"strategy {strategy.name} requires {strategy.minimum_bars:,} historical bars, "
                    "but Kraken's public OHLC bootstrap provides at most 719 completed bars. "
                    "Use a sub-minute model with recent-trade bootstrap, backtest it, or train a "
                    "whole-minute model with a shorter context."
                )
            session = PaperSession(
                config=PaperSessionConfig(
                    strategy_key=request.strategy_key,
                    parameters=request.parameters,
                    symbol=resolved.symbol,
                    rest_pair=resolved.kraken_rest_pair,
                    timeframe_seconds=request.timeframe_seconds,
                    initial_equity=request.initial_equity,
                    fee_bps=request.fee_bps,
                    slippage_bps=request.slippage_bps,
                    minimum_order_notional=request.minimum_order_notional,
                    allow_short=request.allow_short,
                    max_leverage=request.max_leverage,
                    history_size=max(5_000, strategy.minimum_bars + 16),
                    bootstrap_bars=max(request.bootstrap_bars, strategy.minimum_bars),
                    bootstrap_max_trade_pages=resolved.kraken_trade_bootstrap_max_pages,
                    bootstrap_trade_page_delay_seconds=(
                        resolved.kraken_trade_bootstrap_page_delay_seconds
                    ),
                ),
                strategy=strategy,
                rest_client=services.rest,
                stream=KrakenMarketStream(
                    resolved.kraken_ws_url,
                    resolved.symbol,
                    trade_snapshot=request.timeframe_seconds < 60,
                ),
                database=services.database,
                hub=services.hub,
            )
            await services.sessions.replace(session)
            return session.state()
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                502, f"could not start Kraken paper session: {type(exc).__name__}: {exc}"
            ) from exc

    @app.post("/api/paper/stop")
    async def stop_paper() -> dict[str, Any]:
        await services.sessions.stop()
        return services.sessions.state()

    @app.get("/api/paper/state")
    async def paper_state() -> dict[str, Any]:
        return services.sessions.state()

    @app.get("/api/paper/history")
    async def paper_history(timeframe_seconds: int = 60, limit: int = 1_000) -> dict[str, Any]:
        bars = services.database.load_bars(
            symbol=resolved.symbol,
            timeframe_seconds=timeframe_seconds,
            limit=min(max(limit, 2), 20_000),
        )
        return {"bars": [bar.to_dict() for bar in bars], "count": len(bars)}

    @app.websocket("/ws/events")
    async def websocket_events(websocket: WebSocket) -> None:
        await services.hub.connect(websocket)
        await websocket.send_json({"type": "session", "payload": services.sessions.state()})
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            await services.hub.disconnect(websocket)
        except Exception:
            await services.hub.disconnect(websocket)

    frontend = resolved.frontend_dist
    if frontend.exists():
        assets = frontend / "assets"
        if assets.exists():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa(full_path: str) -> FileResponse:
            requested = frontend / full_path
            if (
                full_path
                and requested.is_file()
                and requested.resolve().is_relative_to(frontend.resolve())
            ):
                return FileResponse(requested)
            return FileResponse(frontend / "index.html")

    return app
