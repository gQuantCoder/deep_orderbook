from deep_orderbook.recorder import Receiver, Writer
from deep_orderbook.shapper import BookShapper

import asyncio
import pandas as pd
import numpy as np
from pylab import rcParams
import aioitertools
import functools


MARKETS = ["ETHBTC", "BTCUSDT", "ETHUSDT", "BNBBTC", "BNBETH", "BNBUSDT"]
LENGTH = 512

class ImageStream:
    def __init__(self, markets):
        self.frame = None
        self.markets = markets or MARKETS
    
    async def setup(self):
        self.receiver = await Receiver.create(markets=self.markets, print_level=1)

        shappers = {pair: await BookShapper.create() for pair in self.markets}
        multi_replay = self.receiver.multi_generator(shappers)

        _ = await multi_replay.__anext__()

        genarr = BookShapper.gen_array_async(market_replay=multi_replay, markets=self.markets)
        _ = await aioitertools.next(genarr)

        self.genacc = aioitertools.accumulate(genarr, functools.partial(BookShapper.build, max_length=LENGTH))
        _ = await aioitertools.next(self.genacc)

    async def run(self):
        async for toshow in BookShapper.images(accumulated_arrays=self.genacc, every=1, LENGTH=LENGTH):
            self.frame = toshow.copy()

    def start(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        asyncio.get_event_loop().run_until_complete(self.setup())
        asyncio.get_event_loop().run_until_complete(self.run())

    def stop(self):
        pass

    def read(self):
        return self.frame if self.frame is None else self.frame * 255