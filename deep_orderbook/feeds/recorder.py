import asyncio
import aiofiles
import copy
import os
import time
import datetime

from deep_orderbook.feeds.base_feed import BaseFeed
from deep_orderbook.utils import logger


class Writer:
    PRINT_EVENTS = False
    PRINT_MESSAGE = False

    def __init__(self, feed: BaseFeed, data_folder: str) -> None:
        self.feed = feed
        self.data_folder = data_folder
        self.msg_history: list[str] = []
        self.run_timer = False
        self.lock = asyncio.Lock()
        self.tradelock = asyncio.Lock()

        self.L2folder = f"{data_folder}/L2"
        for symbol in feed.markets:
            os.makedirs(f"{self.L2folder}/{symbol}", exist_ok=True)

    async def start(self):
        asyncio.create_task(self.run_writer(save_period_minutes=1))
        await asyncio.Event().wait()  # Keeps the writer running indefinitely

    async def save_updates_since(self, prev_ts=None):
        prev_ts = prev_ts or self.prev_th
        upds = (
            datetime.datetime.fromtimestamp(prev_ts, datetime.timezone.utc)
            .isoformat()
            .replace(":", "-")
        )
        for symbol in self.feed.markets:
            async with self.lock:
                depth_cache = self.feed.depth_managers[symbol]
                depths, trades = depth_cache.dump(and_reset_trades=True)

            async with aiofiles.open(
                f"{self.L2folder}/{symbol}/{upds}_update.json", "w"
            ) as fp:
                await fp.write(depths)
            async with aiofiles.open(
                f"{self.L2folder}/{symbol}/{upds}_trades.json", "w"
            ) as fp:
                await fp.write(trades)
        logger.info(f"Saved updates since {upds}")

    async def save_snapshot(self, cur_ts, max_levels=1000):
        snap = (
            datetime.datetime.fromtimestamp(cur_ts, datetime.timezone.utc)
            .isoformat()
            .replace(":", "-")
        )
        if cur_ts:
            L2s_coro = [
                self.feed.client.get_order_book(symbol=pair, limit=max_levels)
                for pair in self.feed.markets
            ]
            L2s = await asyncio.gather(*L2s_coro)
            for symbol, L2 in zip(self.feed.markets, L2s):
                async with aiofiles.open(
                    f"{self.L2folder}/{symbol}/{snap}_snapshot.json", "w"
                ) as fp:
                    await fp.write(json.dumps(L2))
        logger.info("Saved snapshot")

    async def run_writer(self, save_period_minutes=60):
        save_period_seconds = save_period_minutes * 60

        PERIOD_L2 = 10
        self.th = 0
        self.prev_th = 0

        logger.debug(
            "Starting run_writer with save_period_minutes=%d", save_period_minutes
        )

        try:
            while True:
                t = time.time()
                logger.debug("Current time: %f", t)

                await asyncio.sleep(PERIOD_L2 - t % PERIOD_L2)
                logger.debug(
                    "Woke up after sleeping for PERIOD_L2=%d seconds", PERIOD_L2
                )

                t_ini = int(time.time())
                self.new_th = int(t_ini // save_period_seconds) * save_period_seconds

                logger.debug("Calculated t_ini=%d, new_th=%d", t_ini, self.new_th)

                if self.new_th > self.th:
                    logger.debug(
                        "new_th %d is greater than th %d, saving updates and snapshots",
                        self.new_th,
                        self.th,
                    )
                    if self.th == 0:
                        logger.debug(
                            "First run, saving updates and snapshots with t_ini=%d",
                            t_ini,
                        )
                        await self.save_updates_since(t_ini)
                        await self.save_snapshot(t_ini)
                        self.prev_th = t_ini
                    else:
                        logger.debug(
                            "Saving updates and snapshots with new_th=%d", self.new_th
                        )
                        await self.save_updates_since()
                        await self.save_snapshot(self.new_th)
                        self.prev_th = self.new_th
                    self.th = self.new_th
        except asyncio.CancelledError:
            logger.debug("run_writer cancelled, saving remaining updates")
            await self.save_updates_since()
        except Exception as e:
            logger.error("Exception occurred: %s", e)
            raise e


async def main():
    from deep_orderbook.feeds.coinbase_feed import CoinbaseFeed

    MARKETS = ["ETH-USD"]
    data_folder = "data"

    async with CoinbaseFeed(markets=MARKETS, record_history=True) as feed:
        writer = Writer(feed=feed, data_folder=data_folder)
        await writer.start()


if __name__ == '__main__':
    asyncio.run(main())
