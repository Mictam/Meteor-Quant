# Contributing

## Development setup

```bash
./install.sh --dev --frontend
```

Windows:

```powershell
.\install.ps1 --dev --frontend
```

## Quality gates

Before opening a pull request, run:

```bash
python -m pytest
python -m ruff check src tests scripts
python -m mypy src/meteor_quant
cd frontend && npm run build
```

When Rust code changes:

```bash
cargo test --manifest-path rust/meteor-engine/Cargo.toml
cargo clippy --manifest-path rust/meteor-engine/Cargo.toml -- -D warnings
```

## Design rules

- Preserve next-bar causality.
- Do not calculate authoritative P&L in the browser or strategy layer.
- Keep the Rust and Python execution engines behaviorally aligned.
- Add tests for every execution, caching, or schema change.
- Do not add private-exchange credentials or live-order submission.
- Keep generated data, checkpoints, and local results out of Git.

## Pull requests

A pull request should include:

- a focused problem statement;
- architecture and safety implications;
- tests added or updated;
- exact verification commands;
- migration notes for persisted data or checkpoints.
