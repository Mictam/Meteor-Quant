# Verification

Meteor Quant 1.0.0 was verified as a clean standalone distribution from the repository root.

## Completed checks

| Area | Result |
|---|---|
| Core Python test suite | 34 passed; 2 optional ML modules skipped when PyTorch was intentionally absent |
| MarketLM CPU suite | 8 passed |
| MarketHybrid CPU suite | 8 passed |
| Total unique Python tests | 50 passed |
| Ruff | Passed |
| Strict MyPy | Passed for `src/meteor_quant` |
| React/TypeScript production build | Passed |
| Python wheel build | Passed |
| Python source distribution build | Passed |
| Installed-wheel server smoke | Passed |
| Packaged dashboard smoke | Passed |
| Paper-only health contract | Passed |
| Public npm lockfile check | Passed; no private registry URLs |
| Secret-pattern scan | Passed |
| Showcase-scope regression | Passed; experimental BTC pullback strategy absent |

The installed-wheel smoke test started the server, loaded the packaged React dashboard, returned a healthy API response with `paper_only: true`, and loaded the baseline strategy registry without requiring PyTorch.

## Optional native engine

The Rust source is included under `rust/meteor-engine`, and CI is configured to compile and test it on a public GitHub runner. The packaging container used for this release did not have a Rust toolchain and could not download one because outbound package resolution was unavailable, so the native binary was not rebuilt locally during final packaging. The independently implemented Python/PyArrow execution engine remains the tested portable fallback.

Before publishing a tagged public release, run:

```bash
cargo test --manifest-path rust/meteor-engine/Cargo.toml
cargo build --release --manifest-path rust/meteor-engine/Cargo.toml
```

or let the included GitHub Actions workflow perform the native-engine job.

## Reproduction commands

```bash
python -m pytest
python -m ruff check src tests scripts
python -m mypy src/meteor_quant
cd frontend && npm ci && npm run build
python -m build
```

Optional neural-model tests require the relevant PyTorch extra. Native-engine checks require Rust stable.
