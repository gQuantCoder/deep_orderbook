import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Protocol, cast

import polars as pl
from polars.functions.col import col
from rich import print
from tqdm.auto import tqdm

from deep_orderbook.config import ReplayConfig
from deep_orderbook.marketdata import MulitSymbolOneSecondEnds
from deep_orderbook.utils import logger

OnMessageCallback = Callable[[datetime, "pl.DataFrame | EndReplay"], Awaitable[None]]


class EndReplay:
    pass


class ReplayerProtocol(Protocol):
    on_message: OnMessageCallback | None

    async def open_async(self) -> None: ...
    async def subscribe_async(self, product_ids: list[str], channels: list[str]) -> None: ...
    async def unsubscribe_all_async(self) -> None: ...
    async def close_async(self) -> None: ...


class ParquetReplayer:
    def __init__(
        self,
        config: ReplayConfig | None = None,
        directory: str = "",
        date_regexp: str = "",
    ) -> None:
        self.config = config or ReplayConfig()
        self.directory = Path(directory) if directory else self.config.data_dir
        self.date_regexp = date_regexp or self.config.date_regexp
        self.on_message: OnMessageCallback | None = None
        self.current_file = None  # Track current file being processed
        self.parquet_files = []

    async def open_async(self) -> None:
        self.parquet_files = self.config.file_list()
        logger.info(f"Found {len(self.parquet_files)} parquet files in {self.directory}")
        if not self.parquet_files:
            print(f"No parquet files found in {self.directory} matching {self.date_regexp}")
            raise FileNotFoundError(f"No parquet files found in {self.directory} matching {self.date_regexp}")

    def skip_n_files(self, n: int) -> None:
        self.parquet_files = self.parquet_files[n:]
        if self.parquet_files:
            self.current_file = self.parquet_files[0]

    async def close_async(self) -> None:
        pass

    async def subscribe_async(
        self,
        product_ids: list[str],
        channels: list[str],
    ) -> None:
        # weirdly, the subscription name is not necessarily the same as the channel name
        channel_names = ["l2_data"] if "level2" in channels else []
        channel_names += ["market_trades"] if "market_trades" in channels else []

        # Process each parquet file individually
        self.feed_task = asyncio.create_task(self.feed_(product_ids, channel_names))

    async def feed_(
        self,
        product_ids: list[str],
        channel_names: list[str],
    ) -> None:
        await asyncio.sleep(0.01)
        if not self.on_message:
            raise ValueError("on_message handler not set for ParquetReplayer.")
        last_t_win: datetime | None = None
        for parquet_file in self.parquet_files:
            self.current_file = parquet_file  # Update current file
            logger.info(f"Reading {parquet_file}")
            df = pl.read_parquet(parquet_file)

            # should work, but doesn't seem to
            df = df.set_sorted("timestamp")
            # # # so we sort it manually...
            df = df.sort("timestamp")

            # filter on product_ids and channels
            if product_ids:
                df = df.filter(col("product_id").is_in(product_ids))
            if channel_names:
                df = df.filter(col("channel").is_in(channel_names))

            grouped = df.group_by_dynamic("timestamp", every=self.config.every, label="right")
            # grouped.explain(streaming=True)
            with tqdm(grouped, leave=False, desc="grouped") as windows:
                for (t_win_raw,), df_s in windows:
                    t_win = cast(datetime, t_win_raw)
                    last_t_win = t_win
                    windows.set_description(
                        f"replay: {t_win!s:25.22}, num trades: {len(df_s.filter(col('channel') == 'market_trades')):>3}"
                    )
                    if t_win.time() < self.config.skip_until_time:
                        continue
                    await self.on_message(t_win, df_s)
            logger.info(f"Finished {parquet_file}")
        if last_t_win is not None:
            await self.on_message(last_t_win, EndReplay())

    async def unsubscribe_all_async(self) -> None:
        self.feed_task.cancel()


async def iter_sec(config: ReplayConfig, on_message: OnMessageCallback) -> AsyncGenerator[MulitSymbolOneSecondEnds, None]:
    from deep_orderbook.feeds.coinbase_feed import CoinbaseFeed

    async with CoinbaseFeed(
        config=config,
        replayer=ParquetReplayer(config=config),
    ) as feed:
        async for onesec in feed.one_second_iterator():
            yield onesec


# =============================================================================
# CLI for testing: this makes local testing and debugging fast and easy
# =============================================================================
if __name__ == "__main__":  # pragma: no cover

    async def main():
        import pyinstrument

        from deep_orderbook.feeds.coinbase_feed import CoinbaseFeed

        config = ReplayConfig(
            markets=["ETH-USD"],  # ['BTC-USD', 'ETH-USD', 'ETH-BTC'],
            date_regexp="2026-04*",
            max_samples=250,
            every="100ms",
            data_dir=Path("../crypto"),
            # skip_until_time="05:30",
        )
        with pyinstrument.Profiler() as profiler:
            async with CoinbaseFeed(
                config=config,
                replayer=ParquetReplayer(config=config),
            ) as feed:
                async for onesec in feed.one_second_iterator():
                    print(f"{onesec}")
        profiler.open_in_browser(timeline=False)

    asyncio.run(main())
