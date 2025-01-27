import collections
from pathlib import Path
from deep_orderbook import marketdata as md
from typing import AsyncGenerator, TypeVar
import polars as pl

# defien a templte type for Messages return by __aiter__
Message = TypeVar("Message", bound=md.Message)


class EndFeed:
    pass


class BaseFeed:
    def __init__(self) -> None:
        self.markets: list[str] = []
        self.msg_history: list[str] = []
        self.depth_managers: dict[str, md.DepthCachePlus] = {}
        self.trade_tapes: dict[str, list[md.Trade]] = collections.defaultdict(list)

    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc_value, traceback):
        pass

    async def __aiter__(self) -> AsyncGenerator[md.Message, None]:
        raise NotImplementedError
        yield None

    async def close_queue(self):
        raise NotImplementedError

    @classmethod
    async def polarize(
        cls,
        *,
        jsonl_path: Path | None = None,
        json_str: str | None = None,
        explode=None,
        df: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        raise NotImplementedError
