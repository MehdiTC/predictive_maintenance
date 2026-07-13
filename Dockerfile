# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.11.24 AS uv

FROM python:3.12-slim-bookworm AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/turbine-guard

COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /build

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

FROM python:3.12-slim-bookworm AS runtime

ENV PATH=/opt/turbine-guard/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TURBINE_GUARD_ENVIRONMENT=production \
    TURBINE_GUARD_LOG_LEVEL=INFO \
    TURBINE_GUARD_DATA_DIR=/var/lib/turbine-guard \
    TURBINE_GUARD_API_HOST=0.0.0.0 \
    TURBINE_GUARD_API_PORT=8000

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 turbineguard \
    && useradd --uid 10001 --gid turbineguard --create-home --shell /usr/sbin/nologin turbineguard \
    && mkdir -p /app /var/lib/turbine-guard /var/lib/mlflow /mlartifacts \
    && chown -R turbineguard:turbineguard /app /var/lib/turbine-guard /var/lib/mlflow /mlartifacts

WORKDIR /app

COPY --from=builder /opt/turbine-guard /opt/turbine-guard
COPY --from=builder --chown=turbineguard:turbineguard /build/src /app/src
COPY --chown=turbineguard:turbineguard alembic.ini ./alembic.ini
COPY --chown=turbineguard:turbineguard alembic/ ./alembic/
COPY --chown=turbineguard:turbineguard scripts/ ./scripts/

USER 10001:10001

EXPOSE 8000

CMD ["python", "scripts/run_api.py"]
