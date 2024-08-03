import collections
from deep_orderbook import marketdata as md


class BaseFeed:
    def __init__(self) -> None:
        self.markets: list[str] = []
        self.msg_history: list[str] = []
        self.depth_managers: dict[str, md.DepthCachePlus] = {}
        self.trade_tapes: dict[str, list[md.Trade]] = collections.defaultdict(list)

    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc_value, traceback):
        pass
