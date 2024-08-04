import asyncio
import numpy as np
from deep_orderbook.replayer import Replayer
from deep_orderbook.shaper import BookShaper
from deep_orderbook.utils import logger

async def test_raw_replay():
    from aioitertools import enumerate, next as anext

    MARKETS = ["ETHBTC", "BTCUSDT", "ETHUSDT"]

    file_replayer = Replayer('../crypto-trading/data/L2', date_regexp='2020')
    areplay = file_replayer.replayL2_async(pair='ETHBTC', shaper=BookShaper())
    a = await anext(areplay)
    logger.debug(a)
    batptr = await anext(areplay)

    for i in range(10):
        batptr = await anext(areplay)
    logger.debug(f"bids:\n{batptr['bids'].head()}")
    logger.debug(f"asks:\n{batptr['asks'].head()}")
    logger.debug(f"prices:\n{batptr['price']}")
    logger.debug(f"trades:\n{batptr['trades']}")

    multi_replay = file_replayer.multireplayL2_async(pairs=MARKETS)
    d = await anext(multi_replay)
    logger.debug(d)

    genarr = BookShaper.gen_array_async(market_replay=multi_replay, markets=MARKETS)
    _ = await anext(genarr)

    genacc = BookShaper.accumulate_array(genarr, markets=MARKETS)
    _ = await anext(genacc)

    num_to_output = 100
    async for bb in multi_replay:
        num_to_output -= 1
        logger.debug(bb)
        if num_to_output < 0:
            break


async def test_zipped_replay():
    file_replayer = Replayer('../data/crypto', date_regexp='20')
    shaper = BookShaper()
    s = file_replayer.zipped()
    logger.debug(s)

    single_pair = 'ETHUSDT'
    logger.debug(f"replaying a single market: {single_pair}")
    areplay = file_replayer.replayL2_async(pair=single_pair, shaper=shaper)
    logger.debug(areplay)
    num_to_output = 100
    async for bb in areplay:
        num_to_output -= 1
        logger.debug(bb)
        if num_to_output < 0:
            break

    multi_pairs = ['ETHUSDT', 'BTCUSDT', 'ETHBTC']
    logger.info(f"synchronizing and replaying multilple markets: {multi_pairs}")
    file_gen = file_replayer.multireplayL2_async(pairs=multi_pairs)
    num_to_output = 100
    async for bb in file_gen:
        num_to_output -= 1
        # logger.debug(bb)
        if num_to_output < 0:
            break


async def main():
    await test_zipped_replay()
    await test_raw_replay()


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
