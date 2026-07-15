from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from meteor_quant.api import create_app
from meteor_quant.config import Settings
from meteor_quant.datasets import DatasetCatalog
from meteor_quant.markethybrid.jobs import MarketHybridJobManager
from meteor_quant.markethybrid.schemas import (
    MarketHybridRegisterRequest,
    MarketHybridRunRequest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meteor", description="Meteor Quant research and paper-trading platform")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="start the API and React dashboard")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    prepare = subparsers.add_parser(
        "prepare-data", help="normalize the two Binance CSV files to Parquet"
    )
    prepare.add_argument("--data-dir", type=Path, default=Path("data"))
    prepare.add_argument("--force", action="store_true")

    start_hybrid = subparsers.add_parser(
        "start-markethybrid", help="start a MarketHybrid prepare or training worker"
    )
    start_hybrid.add_argument("--config", type=Path, required=True)
    start_hybrid.add_argument("--data-dir", type=Path, default=Path("data"))

    hybrid_status = subparsers.add_parser(
        "markethybrid-status", help="show a MarketHybrid run status"
    )
    hybrid_status.add_argument("--run-id", required=True)
    hybrid_status.add_argument("--data-dir", type=Path, default=Path("data"))

    stop_hybrid = subparsers.add_parser(
        "stop-markethybrid", help="request a clean MarketHybrid worker stop"
    )
    stop_hybrid.add_argument("--run-id", required=True)
    stop_hybrid.add_argument("--data-dir", type=Path, default=Path("data"))

    register_hybrid = subparsers.add_parser(
        "register-markethybrid", help="register a completed MarketHybrid checkpoint"
    )
    register_hybrid.add_argument("--run-id", required=True)
    register_hybrid.add_argument("--data-dir", type=Path, default=Path("data"))
    register_hybrid.add_argument(
        "--checkpoint",
        choices=("best", "best_hybrid", "best_loss", "final"),
        default="best",
    )
    register_hybrid.add_argument("--primary-horizon", type=int)
    register_hybrid.add_argument("--display-name")
    register_hybrid.add_argument("--description")

    subparsers.add_parser(
        "print-markethybrid-default-config",
        help="print the fully resolved optimized MarketHybrid configuration",
    )

    validate_hybrid = subparsers.add_parser(
        "validate-markethybrid-config",
        help="validate and print a resolved MarketHybrid configuration",
    )
    validate_hybrid.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        root = Path.cwd().resolve()
        default_paths = Settings()
        source_frontend = root / "frontend" / "dist"
        frontend_dist = source_frontend if source_frontend.exists() else default_paths.frontend_dist
        settings = Settings(
            project_root=root,
            data_dir=root / "data",
            user_strategy_dir=root / "user_strategies",
            frontend_dist=frontend_dist,
        )
        if args.reload:
            os.environ.setdefault("METEOR_PROJECT_ROOT", str(root))
            os.environ.setdefault("METEOR_DATA_DIR", str(root / "data"))
            os.environ.setdefault("METEOR_USER_STRATEGY_DIR", str(root / "user_strategies"))
            os.environ.setdefault("METEOR_FRONTEND_DIST", str(frontend_dist))
            uvicorn.run(
                "meteor_quant.api:create_app",
                host=args.host,
                port=args.port,
                reload=True,
                factory=True,
            )
        else:
            uvicorn.run(create_app(settings), host=args.host, port=args.port)
        return
    if args.command == "prepare-data":
        descriptor = DatasetCatalog(args.data_dir).prepare(force=args.force)
        print(json.dumps(descriptor.to_dict(), indent=2))
        return
    if args.command == "print-markethybrid-default-config":
        print(json.dumps(MarketHybridRunRequest().model_dump(mode="json"), indent=2))
        return
    if args.command == "validate-markethybrid-config":
        request = MarketHybridRunRequest.model_validate_json(
            args.config.read_text(encoding="utf-8-sig")
        )
        print(json.dumps(request.model_dump(mode="json"), indent=2))
        return
    if args.command in {
        "start-markethybrid",
        "markethybrid-status",
        "stop-markethybrid",
        "register-markethybrid",
    }:
        project_root = Path.cwd().resolve()
        data_dir = args.data_dir.resolve()
        manager = MarketHybridJobManager(project_root, data_dir)
        if args.command == "start-markethybrid":
            request = MarketHybridRunRequest.model_validate_json(
                args.config.read_text(encoding="utf-8-sig")
            )
            result = manager.start(request)
        elif args.command == "markethybrid-status":
            result = manager.get(args.run_id)
        elif args.command == "stop-markethybrid":
            result = manager.stop(args.run_id)
        else:
            result = manager.register(
                args.run_id,
                MarketHybridRegisterRequest(
                    display_name=args.display_name,
                    description=args.description,
                    checkpoint=args.checkpoint,
                    primary_horizon_seconds=args.primary_horizon,
                ),
            )
        print(json.dumps(result, indent=2))
        return
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
