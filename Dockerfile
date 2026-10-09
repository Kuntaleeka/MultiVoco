# Runs as a non-root user and listens on $PORT, which the host sets (Render: 10000).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN useradd --create-home --uid 1000 user
USER user
WORKDIR /home/user/app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/home/user/app/.venv/bin:$PATH"

# Dependencies first, so code changes do not reinstall them.
COPY --chown=user pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev

COPY --chown=user app ./app
COPY --chown=user web ./web

ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port $PORT --proxy-headers --forwarded-allow-ips '*'"]
