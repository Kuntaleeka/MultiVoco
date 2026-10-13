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

# Piper voices, about 60 MB each. Baked in so the service needs no download at start.
RUN python -m piper.download_voices en_US-lessac-medium hi_IN-priyamvada-medium \
    --download-dir voices

COPY --chown=user app ./app
COPY --chown=user web ./web
COPY --chown=user scripts ./scripts

ENV PORT=8000
EXPOSE 8000
# The seed step creates the tables and the synthetic customers if the database is
# empty. With the default SQLite file that is every start, since the disk is not kept.
CMD ["sh", "-c", "python -m scripts.seed && exec uvicorn app.main:app --host 0.0.0.0 --port $PORT --proxy-headers --forwarded-allow-ips '*'"]
