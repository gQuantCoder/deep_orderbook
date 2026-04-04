FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

COPY . /deep_orderbook
WORKDIR /deep_orderbook

RUN uv sync --no-dev

CMD ["uv", "run", "deepbook-record"]
