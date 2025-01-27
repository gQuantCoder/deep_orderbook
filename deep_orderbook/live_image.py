from deep_orderbook.feeds.coinbase_feed import CoinbaseFeed as Feed
from deep_orderbook.shaper import BookShaper

import asyncio
import aioitertools
import functools


MARKETS = ["BTC-USD", "ETH-USD", "ETH-BTC"]
LENGTH = 512


class ImageStream:
    def __init__(self, markets):
        self.frame = None
        self.markets = markets or MARKETS

    async def setup(self):
        receiver = Feed(markets=self.markets)
        receiver.PRINT_MESSAGE = True
        await receiver.__aenter__()

        multi_replay = receiver.multi_generator()

        _ = await multi_replay.__anext__()

        genarr = BookShaper.gen_array_async(
            market_replay=multi_replay, markets=self.markets
        )
        _ = await aioitertools.next(genarr)

        self.genacc = aioitertools.accumulate(
            genarr, functools.partial(BookShaper.build, max_length=LENGTH)
        )
        _ = await aioitertools.next(self.genacc)

    async def run(self):
        async for toshow in BookShaper.images(
            accumulated_arrays=self.genacc, every=1, LENGTH=LENGTH
        ):
            self.frame = toshow.copy()

    async def __aenter__(self):
        await self.setup()
        asyncio.create_task(self.run())
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def read(self):
        if self.frame is not None:
            return self.frame * 255


async def main():
    async with ImageStream(markets=MARKETS) as stream:
        while True:
            frame = await stream.read()
            if frame is not None:
                print(frame.shape)
                # print(frame[384 // 2 - 3 : 384 // 2 + 3, -1:, :])
            await asyncio.sleep(1)
    print("done")


if __name__ == "__main__":
    asyncio.run(main())
