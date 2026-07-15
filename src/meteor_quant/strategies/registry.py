from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from meteor_quant.strategies.builtin import BUILTIN_STRATEGIES
from meteor_quant.strategies.sdk import StrategyPlugin


class StrategyRegistry:
    def __init__(
        self,
        plugin_dir: Path,
        registered_marketlm_dir: Path | None = None,
        registered_markethybrid_dir: Path | None = None,
    ) -> None:
        self.plugin_dir = plugin_dir
        self.registered_marketlm_dir = registered_marketlm_dir
        self.registered_markethybrid_dir = registered_markethybrid_dir
        self._classes: dict[str, type[StrategyPlugin]] = {}
        self._sources: dict[str, str] = {}
        self.errors: list[dict[str, str]] = []
        self.reload()

    def reload(self) -> None:
        self._classes.clear()
        self._sources.clear()
        self.errors.clear()
        for builtin_class in BUILTIN_STRATEGIES:
            self._register(builtin_class, "builtin")
        try:
            from meteor_quant.timesfm.strategy import TimesFMStrategy

            self._register(TimesFMStrategy, "timesfm")
        except Exception as exc:
            self.errors.append({"file": "TimesFM 2.5", "error": f"{type(exc).__name__}: {exc}"})
        if (
            self.registered_marketlm_dir is not None
            and self.registered_marketlm_dir.exists()
            and any(self.registered_marketlm_dir.glob("*.json"))
        ):
            try:
                from meteor_quant.marketlm.strategy import registered_marketlm_strategies

                for marketlm_class in registered_marketlm_strategies(self.registered_marketlm_dir):
                    self._register(marketlm_class, "marketlm")
            except Exception as exc:
                self.errors.append(
                    {
                        "file": str(self.registered_marketlm_dir),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        if (
            self.registered_markethybrid_dir is not None
            and self.registered_markethybrid_dir.exists()
            and any(self.registered_markethybrid_dir.glob("*.json"))
        ):
            try:
                from meteor_quant.markethybrid.strategy import (
                    registered_markethybrid_strategies,
                )

                for hybrid_class in registered_markethybrid_strategies(
                    self.registered_markethybrid_dir
                ):
                    self._register(hybrid_class, "markethybrid")
            except Exception as exc:
                self.errors.append(
                    {
                        "file": str(self.registered_markethybrid_dir),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.plugin_dir.glob("*.py")):
            if path.name.startswith("_"):
                continue
            try:
                module = self._load_module(path)
                exported = getattr(module, "STRATEGY", None)
                classes = exported if isinstance(exported, (list, tuple)) else [exported]
                if classes == [None]:
                    classes = [
                        value
                        for value in module.__dict__.values()
                        if isinstance(value, type)
                        and issubclass(value, StrategyPlugin)
                        and value is not StrategyPlugin
                        and value.__module__ == module.__name__
                    ]
                if not classes:
                    raise ValueError("no StrategyPlugin subclass or STRATEGY export found")
                for candidate in classes:
                    if not isinstance(candidate, type) or not issubclass(candidate, StrategyPlugin):
                        raise TypeError("STRATEGY must contain StrategyPlugin classes")
                    self._register(candidate, str(path))
            except Exception as exc:
                self.errors.append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})

    def create(self, key: str, parameters: dict[str, Any] | None = None) -> StrategyPlugin:
        try:
            return self._classes[key](parameters)
        except KeyError as exc:
            raise KeyError(f"unknown strategy: {key}") from exc

    def list_metadata(self) -> dict[str, Any]:
        return {
            "strategies": [
                self._classes[key].metadata(self._sources[key]) for key in sorted(self._classes)
            ],
            "errors": list(self.errors),
        }

    def _register(self, strategy_class: type[StrategyPlugin], source: str) -> None:
        key = strategy_class.key.strip()
        if not key:
            raise ValueError("strategy key cannot be empty")
        if key in self._classes:
            raise ValueError(f"duplicate strategy key: {key}")
        self._classes[key] = strategy_class
        self._sources[key] = source

    @staticmethod
    def _load_module(path: Path) -> ModuleType:
        module_name = f"meteor_user_strategy_{path.stem}_{abs(hash(path.resolve()))}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot create module spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
