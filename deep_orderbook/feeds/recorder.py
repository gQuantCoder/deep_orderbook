import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import aiofiles
from aiofiles.threadpool.text import AsyncTextIOWrapper
from tqdm.auto import tqdm
import polars as pl
from deep_orderbook.feeds.base_feed import BaseFeed
from deep_orderbook.utils import logger


class Writer:
    def __init__(
        self, *, feed: BaseFeed, directory: str = "data/L2", save_path='data'
    ) -> None:
        self.feed = feed
        self.directory = Path(directory)
        self.save_path = Path(save_path)
        self.files: dict[str, AsyncTextIOWrapper] = {}
        timestamp = (
            datetime.now(tz=timezone.utc).isoformat().replace(":", "-").split('.')[0]
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        self.update_filename = self.directory / f"{timestamp}_update.jsonl"
        self.trade_filename = self.directory / f"{timestamp}_trades.jsonl"
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.output_parquet = self.save_path / f"{timestamp}.parquet"

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
        await self.post_process_file()

    async def post_process_file(self) -> None:
        try:
            with pl.StringCache():
                df_books = await self.feed.polarize(
                    jsonl_path=self.update_filename, explode=['updates']
                )
                df_trades = await self.feed.polarize(
                    jsonl_path=self.trade_filename, explode=['trades']
                )
                df_all = df_books.merge_sorted(df_trades, key="timestamp")
                df_all.write_parquet(self.output_parquet)
                logger.info(f"Saved parquet file: {self.output_parquet}")
                logger.info(
                    f"Removing jsonl files: {self.update_filename}, {self.trade_filename}"
                )
                self.update_filename.unlink()
                self.trade_filename.unlink()
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
            async with Writer(
                feed=feed, directory='data/L2', save_path='/media/photoDS216/crypto'
            ) as recorder:
                await recorder.start_recording()
                await recorder.sleep_until_next_hour()


if __name__ == '__main__':
    asyncio.run(main())
