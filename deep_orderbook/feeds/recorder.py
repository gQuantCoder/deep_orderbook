import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
import aiofiles
from deep_orderbook.feeds.base_feed import BaseFeed
from deep_orderbook.utils import logger


class Writer:
    def __init__(self, *, markets: list[str], directory: str = "data") -> None:
        self.markets = markets
        self.directory = Path(directory)
        self.files: dict[
            str,
            dict[
                str,
                aiofiles.threadpool.AiofilesContextManager,
            ],
        ] = {}

    async def __aenter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        timestamp = (
            datetime.now(tz=timezone.utc).isoformat().replace(":", "-").split('.')[0]
        )
        for market in self.markets:
            market_path = self.directory / "L2" / market
            market_path.mkdir(parents=True, exist_ok=True)
            update_filename = f"{timestamp}_update.jsonl"
            trade_filename = f"{timestamp}_trades.jsonl"
            self.files[market] = {
                "update": await aiofiles.open(market_path / update_filename, mode="a"),
                "trades": await aiofiles.open(market_path / trade_filename, mode="a"),
            }
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        for market_files in self.files.values():
            for file in market_files.values():
                await file.close()

    async def start_recording(self, feed: BaseFeed):
        asyncio.create_task(self._write_messages(feed))

    async def _write_messages(self, feed: BaseFeed):
        async for msg in feed:
            if msg.is_subscription():
                continue
            market = msg.symbol
            if market in self.files:
                if msg.is_book_update():
                    file = self.files[market]['update']
                if msg.is_trade_update():
                    file = self.files[market]['trades']
                await file.write(msg.model_dump_json() + "\n")
            else:
                logger.error(f"Received message for unknown market: {market}")

    async def sleep_unitl_midnight(self):
        now = datetime.now()
        tomorrow = now.replace(day=now.day + 1, hour=0, minute=0, second=0, microsecond=0)
        await asyncio.sleep((tomorrow - now).total_seconds())


async def main():
    from deep_orderbook.feeds.coinbase_feed import CoinbaseFeed

    MARKETS = ["BTC-USD", "ETH-USD", "ETH-BTC"]

    while True:
        async with Writer(markets=MARKETS) as recorder:
            async with CoinbaseFeed(markets=MARKETS, feed_msg_queue=True) as feed:
                await recorder.start_recording(feed)
                await recorder.sleep_unitl_midnight()


if __name__ == '__main__':
    asyncio.run(main())
