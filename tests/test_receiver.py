import asyncio
import pytest

from deep_orderbook.feeds.coinbase_feed import CoinbaseFeed as Receiver
from deep_orderbook.shaper import BookShaper


MARKETS = ["BTC-USD", "ETH-USD", "ETH-BTC", ]

async def test_make_receiver():
    receiver = Receiver(markets=MARKETS)
    await receiver.__aenter__()
    await asyncio.sleep(5)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
