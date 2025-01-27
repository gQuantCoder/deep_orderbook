from pathlib import Path
import asyncio
from tqdm.auto import tqdm
from rich import print
import polars as pl

from deep_orderbook.utils import logger

from deep_orderbook.config import ReplayConfig


class EndReplay:
    pass


class ParquetReplayer:
    def __init__(
        self,
        config: ReplayConfig = ReplayConfig(),
        directory: str = '',
        date_regexp: str = '',
    ) -> None:
        self.config = config
        self.directory = Path(directory) if directory else config.data_dir
        self.date_regexp = date_regexp or config.date_regexp
        self.on_message = None

    async def open_async(self) -> None:
        self.parquet_files = self.config.file_list()
        logger.info(
            f"Found {len(self.parquet_files)} parquet files in {self.directory}"
        )
        if not self.parquet_files:
            raise FileNotFoundError(
                f"No parquet files found in {self.directory} matching {self.date_regexp}"
            )

    async def close_async(self) -> None:
        pass

    async def subscribe_async(
        self,
        product_ids: list[str],
        channels: list[str],
    ) -> None:
        # weirdly, the subscription name is not necessarily the same as the channel name
        channel_names = ['l2_data'] if 'level2' in channels else []
        channel_names += ['market_trades'] if 'market_trades' in channels else []

        # Process each parquet file individually
        self.feed_task = asyncio.create_task(self.feed_(product_ids, channel_names))

    async def feed_(
        self,
        product_ids: list[str],
        channel_names: list[str],
    ) -> None:
        await asyncio.sleep(0.01)
        for parquet_file in self.parquet_files:
            if not self.on_message:
                raise ValueError("on_message handler not set for ParquetReplayer.")

            logger.info(f"Reading {parquet_file}")
            df = pl.read_parquet(parquet_file)

            # should work, but doesn't seem to
            df = df.set_sorted('timestamp')
            # # # so we sort it manually...
            df = df.sort('timestamp')

            # filter on product_ids and channels
            if product_ids:
                df = df.filter(pl.col('product_id').is_in(product_ids))
            if channel_names:
                df = df.filter(pl.col('channel').is_in(channel_names))

            grouped = df.group_by_dynamic(
                "timestamp", every=self.config.every, label='right'
            )
            # grouped.explain(streaming=True)
            with tqdm(grouped, leave=False, desc="grouped") as windows:
                for (t_win,), df_s in windows:
                    windows.set_description(
                        f"replay: {t_win!s:25.22}, num trades: {len(df_s.filter(pl.col('channel') == 'market_trades')):>3}"
                    )
                    if t_win.time() < self.config.skip_until_time:
                        continue
                    await self.on_message(t_win, df_s)
            logger.info(f"Finished {parquet_file}")
        await self.on_message(t_win, EndReplay())

    async def unsubscribe_all_async(self) -> None:
        self.feed_task.cancel()
        pass


async def iter_sec(config: ReplayConfig):
    from deep_orderbook.feeds.coinbase_feed import CoinbaseFeed

    async with CoinbaseFeed(
        config=config,
        replayer=ParquetReplayer(config=config),
    ) as feed:
        async for onesec in feed.one_second_iterator():
            yield onesec


async def main():
    from deep_orderbook.feeds.coinbase_feed import CoinbaseFeed
    import pyinstrument

    config = ReplayConfig(
        markets=['BTC-USD', 'ETH-USD', 'ETH-BTC'],
        date_regexp='2024-09',
        max_samples=250,
        every='100ms',
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


if __name__ == '__main__':
    asyncio.run(main())
