FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV PYTHONOPTIMIZE=1
ENV UV_LINK_MODE=copy

ENV PYTHONPATH="/app/src:$PYTHONPATH"

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

COPY src/ ./src/

ENV PATH="/app/.venv/bin:$PATH"

RUN addgroup --system app && \
adduser --system --ingroup app app && \
chown -R app:app /app && \
mkdir -p /home/app && \
chown -R app:app /home/app && \
mkdir -p /home/app/.streamlit && \
mkdir -p /home/app/.streamlit/data && \
mkdir -p /home/app/.streamlit/cache && \
chown -R app:app /home/app/.streamlit

ENV HOME=/home/app

USER app

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "./src/app.py", "--server.address=0.0.0.0"]