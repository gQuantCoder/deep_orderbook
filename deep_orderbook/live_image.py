from deep_orderbook.feeds.coinbase_feed import CoinbaseFeed as Receiver
from deep_orderbook.shapper import BookShapper

import asyncio
import pandas as pd
import numpy as np
from pylab import rcParams
import aioitertools
import functools


MARKETS = ["ETHBTC", "BTCUSDT", "ETHUSDT"]
MARKETS = ["ETH-BTC", "BTC-USD", "ETH-USD"]
LENGTH = 512


class ImageStream:
    def __init__(self, markets):
        self.frame = None
        self.markets = markets or MARKETS

    async def setup(self):
        
        async with Receiver(markets=self.markets) as receiver:

            multi_replay = receiver.multi_generator(self.markets)

            _ = await multi_replay.__anext__()

            genarr = BookShapper.gen_array_async(
                market_replay=multi_replay, markets=self.markets
            )
            _ = await aioitertools.next(genarr)

            self.genacc = aioitertools.accumulate(
                genarr, functools.partial(BookShapper.build, max_length=LENGTH)
            )
            _ = await aioitertools.next(self.genacc)

    async def run(self):
        async for toshow in BookShapper.images(
            accumulated_arrays=self.genacc, every=1, LENGTH=LENGTH
        ):
            self.frame = toshow.copy()

    async def __aenter__(self):
        await self.setup()
        await self.run()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def read(self):
        return self.frame * 255 if self.frame else self.frame

async def main():
    async with ImageStream(markets=MARKETS) as stream:
        while True:
            frame = stream.read()
            if frame is not None:
                print(frame)
                print(frame.shape)
                break
    print("done")


if __name__ == "__main__":
    asyncio.run(main())
