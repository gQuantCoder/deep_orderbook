import asyncio
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import aiofiles
from tqdm.auto import tqdm
import polars as pl
from deep_orderbook.feeds.base_feed import BaseFeed
from deep_orderbook.utils import logger


class Writer:
    def __init__(self, *, feed: BaseFeed, directory: str = "data") -> None:
        self.feed = feed
        self.directory = Path(directory)
        self.files: dict[str, aiofiles.threadpool.AiofilesContextManager] = {}
        timestamp = (
            datetime.now(tz=timezone.utc).isoformat().replace(":", "-").split('.')[0]
        )
        path = self.directory / "L2"
        path.mkdir(parents=True, exist_ok=True)
        self.update_filename = path / f"{timestamp}_update.jsonl"
        self.trade_filename = path / f"{timestamp}_trades.jsonl"
        self.output_parquet = path / f"../{timestamp}.parquet"

    async def __aenter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.files = {
            "update": await aiofiles.open(self.update_filename, mode="a"),
            "trades": await aiofiles.open(self.trade_filename, mode="a"),
        }
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.feed.close_queue()
        logger.info("Closing file")
        for file in self.files.values():
            await file.close()
        asyncio.create_task(self.post_process_file())

    async def post_process_file(self) -> None:
        try:
            with pl.StringCache():
                df_books = await self.feed.polarize(
                    jsonl_path=self.update_filename, explode=['updates']
                )
                df_trades = await self.feed.polarize(
                    jsonl_path=self.trade_filename, explode=['trades']
                )
                df_all = df_books.merge_sorted(
                    df_trades, key="sequence_num"
                )
                df_all.write_parquet(self.output_parquet)
                logger.info(f"Saved parquet file: {self.output_parquet}")
        except Exception as e:
            logger.error(f"Error post processing file: {e}")

    async def start_recording(self):
        asyncio.create_task(self._write_messages(self.feed))

    async def _write_messages(self, feed: BaseFeed):
        with tqdm(desc="Recording messages") as pbar:
            async for msg in feed:
                if msg.is_subscription():
                    continue

                if msg.is_book_update():
                    file = self.files['update']
                elif msg.is_trade_update():
                    file = self.files['trades']
                await file.write(msg.model_dump_json() + "\n")
                pbar.update()

    async def sleep_until_midnight(self):
        now = datetime.now()
        tomorrow = now.replace(hour=23, minute=59, second=59, microsecond=0)
        num_seconds = max(2, (tomorrow - now).total_seconds())
        logger.warning(f"Sleeping for {num_seconds} seconds")
        await asyncio.sleep(num_seconds)

    async def sleep_until_next_hour(self):
        now = datetime.now()
        next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        num_seconds = max(2, (next_hour - now).total_seconds())
        logger.warning(f"Sleeping for {num_seconds} seconds")
        await asyncio.sleep(num_seconds)


async def main():
    from deep_orderbook.feeds.coinbase_feed import CoinbaseFeed

    MARKETS = ["BTC-USD", "ETH-USD", "ETH-BTC"]

    while True:
        async with CoinbaseFeed(markets=MARKETS, feed_msg_queue=True) as feed:
            async with Writer(feed=feed) as recorder:
                await recorder.start_recording()
                await recorder.sleep_until_next_hour()


if __name__ == '__main__':
    asyncio.run(main())
