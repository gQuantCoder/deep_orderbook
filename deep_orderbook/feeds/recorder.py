import asyncio
import aiofiles
from pathlib import Path
from typing import Dict
from datetime import datetime, timedelta
from deep_orderbook.feeds.base_feed import BaseFeed
from deep_orderbook.feeds.coinbase_feed import SubscriptionsEvent
from deep_orderbook.utils import logger


class Writer:
    def __init__(self, feed: BaseFeed, directory: str = "data") -> None:
        self.feed = feed
        self.directory = Path(directory)
        self.files: Dict[str, aiofiles.threadpool.binary.AsyncBufferedIOBase] = {}

    async def __aenter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().isoformat().replace(":", "-").split('.')[0]
        for market in self.feed.markets:
            market_path = self.directory / "L2" / market
            market_path.mkdir(parents=True, exist_ok=True)
            filename = f"{timestamp}.jsonl"
            market_file = await aiofiles.open(market_path / filename, mode="a")
            self.files[market] = market_file
        self.taskloop = asyncio.create_task(self._write_messages())
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        for market_file in self.files.values():
            await market_file.close()

    async def _write_messages(self):
        async for msg in self.feed:
            if msg is None:
                break
            if isinstance(msg.events[0], SubscriptionsEvent):
                continue
            market = msg.symbol
            if market in self.files:
                line = msg.model_dump_json() + "\n"
                await self.files[market].write(line)
            else:
                raise ValueError(f"Received message for unknown market: {market}")


async def main():
    from deep_orderbook.feeds.coinbase_feed import CoinbaseFeed

    MARKETS = ["BTC-USD", "ETH-USD", "ETH-BTC"]

    while True:
        async with CoinbaseFeed(markets=MARKETS, feed_msg_queue=True) as feed:
            async with Writer(feed=feed):
                seconds_to_midnight = (
                    datetime.combine(
                        datetime.now().date() + timedelta(days=1), datetime.min.time()
                    )
                    - datetime.now()
                ).total_seconds()
                logger.warning(f"Sleeping until midnight: {seconds_to_midnight}")
                await asyncio.sleep(seconds_to_midnight)


if __name__ == '__main__':
    asyncio.run(main())
