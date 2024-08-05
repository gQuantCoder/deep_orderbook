import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
import aiofiles
from tqdm.auto import tqdm
from deep_orderbook.feeds.base_feed import BaseFeed
from deep_orderbook.utils import logger


class Writer:
    def __init__(self, *, feed: BaseFeed, directory: str = "data") -> None:
        self.feed = feed
        self.directory = Path(directory)
        timestamp = (
            datetime.now(tz=timezone.utc).isoformat().replace(":", "-").split('.')[0]
        )
        self.file_path = self.directory / f"{timestamp}_all.jsonl"
        self.file_all: aiofiles.threadpool.text.AsyncTextIOWrapper = None  # type: ignore[assignment]
        self.directory.mkdir(parents=True, exist_ok=True)

    async def __aenter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.file_all = await aiofiles.open(self.file_path, mode="a")
        logger.info(f"Recording to {self.file_path}")
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.feed.close_queue()
        logger.info("Closing file")
        await self.file_all.close()
        asyncio.create_task(self.post_process_file())

    async def post_process_file(self):
        try:
            output_path = self.file_path.with_suffix('.parquet')
            df = await self.feed.polarize(jsonl_path=self.file_path)
            df.write_parquet(output_path)
            logger.info(f"Saved parquet file: {output_path}")
        except Exception as e:
            logger.error(f"Error post processing file: {e}")

    async def start_recording(self):
        asyncio.create_task(self._write_messages())

    async def _write_messages(self,):
        with tqdm(desc="Recording messages") as pbar:
            async for msg in self.feed:
                if msg.is_subscription():
                    continue
                await self.file_all.write(msg.model_dump_json() + "\n")
                pbar.update()

    async def sleep_until_midnight(self):
        now = datetime.now()
        tomorrow = now.replace(day=now.day + 1, hour=0, minute=0, second=0, microsecond=0)
        await asyncio.sleep((tomorrow - now).total_seconds())

    async def sleep_until_next_hour(self):
        await asyncio.sleep(2)
        now = datetime.now()
        next_hour = now.replace(minute=0, second=0, microsecond=0)
        next_hour = next_hour.replace(hour=next_hour.hour + 1)
        num_seconds = (next_hour - now).total_seconds()
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
