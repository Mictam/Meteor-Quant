.PHONY: install install-dev install-ml frontend rust run test lint typecheck check build clean

install:
	python scripts/bootstrap.py

install-dev:
	python scripts/bootstrap.py --dev

install-ml:
	python scripts/bootstrap.py --ml

frontend:
	python scripts/bootstrap.py --frontend

rust:
	python scripts/bootstrap.py --rust

run:
	.venv/bin/python -m meteor_quant.cli serve --host 127.0.0.1 --port 8000

test:
	.venv/bin/python -m pytest

lint:
	.venv/bin/python -m ruff check src tests scripts

typecheck:
	.venv/bin/python -m mypy src/meteor_quant

check: lint typecheck test
	cd frontend && npm run build

build:
	.venv/bin/python -m build

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist frontend/node_modules rust/meteor-engine/target
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
