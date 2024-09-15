from datetime import datetime
from typing import Literal
from pydantic import AliasChoices, BaseModel, Field, ValidationInfo, field_validator
from operator import itemgetter
from deep_orderbook.utils import logger


class Message(BaseModel):
    @property
    def symbol(self) -> str:
        raise NotImplementedError

    def is_book_update(self):
        raise NotImplementedError

    def is_trade_update(self):
        raise NotImplementedError

    def is_subscription(self):
        raise NotImplementedError


class Trade(BaseModel):
    # trade_id: str = Field(alias='trade_id')
    product_id: str = Field(alias='product_id')
    price: float = Field(alias='price')
    size: float = Field(alias='size')
    side: str
    # time: datetime

    def to_binance_format(self, time: int = 0):
        return TradeUpdate(
            e='aggTrade',
            E=0,
            s='NA',
            a=0,
            p=self.price,
            q=self.size,
            m=self.side == 'BUY',
            f=0,
            l=1,
            T=time or int(datetime.now().timestamp()),
        )


class OrderLevel(BaseModel):
    price: float = Field(validation_alias=AliasChoices('price_level', 'price'))
    size: float = Field(validation_alias=AliasChoices('new_quantity', 'size'))


class OneSecondEnds(BaseModel):
    bids: list[OrderLevel]
    asks: list[OrderLevel]
    trades: list[Trade]


class MulitSymbolOneSecondEnds(BaseModel):
    time: datetime
    symbols: dict[str, OneSecondEnds] = {}

    def __str__(self):
        """returns a simplfied verion of the object.
        for each symbol, BBO and number of trades"""
        to_ret = f"One second: {self.time} \n"
        for symbol, sec in self.symbols.items():
            to_ret += f"{symbol}: BBO: {sec.bids[0]}-{sec.asks[0]}, num trades: {len(sec.trades)}\n"
        return to_ret

    @classmethod
    async def make_one_second(cls, time: datetime, depth_managers: dict[str, 'DepthCachePlus']):
        oneSec = MulitSymbolOneSecondEnds(time=time)
        for symbol, depth in depth_managers.items():
            oneSec.symbols[symbol] = depth.make_one_sec()
        return oneSec


class BookUpdate(BaseModel):
    bids: list[OrderLevel]
    asks: list[OrderLevel]

    @field_validator('bids', 'asks', mode='before')
    @classmethod
    def from_lists(cls, values: list, info: ValidationInfo):
        if not values:
            return []
        if isinstance(values[0], list):
            if len(values[0]) == 2:
                return [{'price_level': v[0], 'new_quantity': v[1]} for v in values]
            else:
                raise ValueError(
                    f"PriceLevel: {values} is not a valid list of length 2"
                )
        return values


class BookSnaphsot(BookUpdate):
    lastUpdateId: int
    pass


class BinanceUpdate(BaseModel):
    e: Literal['depthUpdate', 'aggTrade']
    E: int = Field(alias='E')
    s: str = Field(alias='s')


class BinanceBookUpdate(BinanceUpdate, BookUpdate):
    """
    {
        "e": "depthUpdate", # Event type
        "E": 123456789,     # Event time
        "s": "BNBBTC",      # Symbol
        "U": 157,           # First update ID in event
        "u": 160,           # Final update ID in event
        "b": [              # Bids to be updated
            [
                "0.0024",   # price level to be updated
                "10",       # quantity
                []          # ignore
            ]
        ],
        "a": [              # Asks to be updated
            [
                "0.0026",   # price level to be updated
                "100",      # quantity
                []          # ignore
            ]
        ]
    }
    """

    first_id: int = Field(alias='U')
    final_id: int = Field(alias='u')
    bids: list[OrderLevel] = Field(alias='b')
    asks: list[OrderLevel] = Field(alias='a')


class TradeUpdate(BinanceUpdate):
    """
    {
        "e": "aggTrade",                # event type
        "E": 1499405254326,             # event time
        "s": "ETHBTC",                  # symbol
        "a": 70232,                             # aggregated tradeid
        "p": "0.10281118",              # price
        "q": "8.15632997",              # quantity
        "f": 77489,                             # first breakdown trade id
        "l": 77489,                             # last breakdown trade id
        "T": 1499405254324,             # trade time
        "m": false,                             # whether buyer is a maker
        "M": true                               # can be ignored
    }
    """

    price: float = Field(alias='p')
    size: float = Field(alias='q')
    is_buyer_maker: bool = Field(alias='m')
    trade_id: int = Field(alias='a')
    first_trade_id: int = Field(alias='f')
    last_trade_id: int = Field(alias='l')
    T: int = Field(alias='T')


class TradeBunch(BaseModel):
    trades: list[Trade] = []

    def add_trade(self, trade: Trade):
        self.trades.append(trade)

    def clear_trades(self):
        self.trades = []


class DepthCachePlus(BaseModel):
    # symbol: str
    _bids: dict[float, float] = {}
    _asks: dict[float, float] = {}
    trade_bunch: TradeBunch = TradeBunch()

    def add_trade(self, trade: Trade):
        self.trade_bunch.add_trade(trade)

    def get_bids(self):
        lst = [(price, quantity) for price, quantity in self._bids.items() if quantity]
        return sorted(lst, key=itemgetter(0), reverse=True)

    def get_asks(self):
        lst = [(price, quantity) for price, quantity in self._asks.items() if quantity]
        return sorted(lst, key=itemgetter(0), reverse=False)

    def add_bid(self, bid: OrderLevel):
        self._bids[bid.price] = bid.size

    def add_ask(self, ask: OrderLevel):
        self._asks[ask.price] = ask.size

    def update(self, updates: BookUpdate):
        for bid in updates.bids:
            self.add_bid(bid)
        for ask in updates.asks:
            self.add_ask(ask)

    def reset(self, snapshot: BookUpdate | None = None):
        self._bids = {}
        self._asks = {}
        if snapshot:
            for bid in snapshot.bids:
                self.add_bid(bid)
            for ask in snapshot.asks:
                self.add_ask(ask)

    def get_bids_asks(self):
        bids = self.get_bids()
        asks = self.get_asks()
        if bids and asks and bids[0][0] >= asks[0][0]:
            logger.warning(
                f"cleaning the crossed BBO \nBIDS: {bids[:5]}\nASKS: {asks[:5]}"
            )
            for p in list(self._bids.keys()):
                if p >= asks[0][0]:
                    del self._bids[p]
            for p in list(self._asks.keys()):
                if p <= bids[0][0]:
                    del self._asks[p]
            bids = self.get_bids()
            asks = self.get_asks()
            logger.debug(f"result: \nBIDS: {bids[:5]}\nASKS: {asks[:5]}")

            assert bids[0][0] < asks[0][0]
        return bids, asks

    def dump(self, and_reset_trades=False) -> tuple[str, str]:
        depths: str = self.model_dump_json(include={'_bids', '_asks'})
        trades: str = self.trade_bunch.model_dump_json(include={'trades'})
        if and_reset_trades:
            self.trade_bunch.clear_trades()
        return depths, trades

    def make_one_sec(self):
        bids, asks = self.get_bids_asks()
        # if not bids or not asks:
        #     logger.warning(f"no bids or asks for {symbol}")
        #     break
        return OneSecondEnds(
            bids=[OrderLevel(price=price, size=size) for price, size in bids],
            asks=[OrderLevel(price=price, size=size) for price, size in asks],
            trades=self.trade_bunch.trades,
        )
