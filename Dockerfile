FROM node:22-bookworm-slim AS frontend
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM rust:1-bookworm AS rust-engine
WORKDIR /build/rust
COPY rust/meteor-engine/ ./
RUN cargo build --release

FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    METEOR_RUST_ENGINE_PATH=/app/bin/meteor-engine \
    METEOR_FRONTEND_DIST=/app/src/meteor_quant/static
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
COPY --from=frontend /build/frontend/dist ./src/meteor_quant/static
RUN python -m pip install --no-cache-dir .
COPY --from=rust-engine /build/rust/target/release/meteor-engine ./bin/meteor-engine
COPY configs/ ./configs/
COPY user_strategies/ ./user_strategies/
RUN mkdir -p data/cache data/results
EXPOSE 8000
CMD ["python", "-m", "meteor_quant.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
