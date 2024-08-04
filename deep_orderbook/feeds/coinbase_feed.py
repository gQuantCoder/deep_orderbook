import asyncio
import collections
import time
from typing import AsyncGenerator, Literal
from datetime import datetime
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from coinbase.websocket import WSClient
from deep_orderbook import marketdata as md
from deep_orderbook.shaper import BookShaper
from deep_orderbook.feeds.base_feed import BaseFeed
from deep_orderbook.utils import logger


class Settings(BaseSettings):
    api_key: str = ''
    api_secret: str = ''
    model_config = SettingsConfigDict(
        env_file='credentials/coinbase.txt', env_file_encoding='utf-8'
    )


class SubscriptionsEvent(BaseModel):
    class Subscription(BaseModel):
        level2: list[str] | None = None
        market_trades: list[str] | None = None

    subscriptions: Subscription


class CoinbasePriceLevel(md.OrderLevel):
    side: Literal['bid', 'offer']
    # event_time: datetime = Field(alias='event_time')


class L2Event(BaseModel):
    type: Literal['snapshot', 'update']
    product_id: str = Field(alias='product_id')
    updates: list[CoinbasePriceLevel]

    def book_update(self) -> md.BookUpdate:
        update = md.BookUpdate(
            bids=[x for x in self.updates if x.side == 'bid'],
            asks=[x for x in self.updates if x.side == 'offer'],
        )
        return update

    @property
    def pair(self):
        return self.product_id


class TradeEvent(BaseModel):
    type: Literal['snapshot', 'update']
    trades: list[md.Trade]

    @property
    def product_id(self):
        # check it is the same symbol in all trades
        assert (
            len(set([trade.product_id for trade in self.trades])) == 1
        ), f"Not the same symbol in  trades: {self.model_dump_json()}"
        return self.trades[0].product_id


class CoinbaseMessage(md.Message):
    channel: Literal['l2_data', 'market_trades', 'subscriptions']
    timestamp: datetime
    sequence_num: int = Field(alias='sequence_num')
    events: list[L2Event] | list[TradeEvent] | list[SubscriptionsEvent]

    @property
    def symbol(self):
        # check it is the same symbol in all events
        assert (
            len(set([event.product_id for event in self.events])) == 1
        ), f"Not the same symbol in events: {self.model_dump_json()}"
        return self.events[0].product_id
    
    def is_book_update(self):
        return self.channel == 'l2_data'
    
    def is_trade_update(self):
        return self.channel == 'market_trades'
    
    def is_subscription(self):
        return self.channel == 'subscriptions'

class CoinbaseFeed(BaseFeed):
    PRINT_EVENTS = False
    PRINT_MESSAGE = False

    def __init__(self, markets: list[str], feed_msg_queue=False) -> None:
        settings = Settings()
        self.feed_msg_queue = feed_msg_queue
        # print(settings.api_key, settings.api_secret)
        self._client = WSClient(
            api_key=settings.api_key,
            api_secret=settings.api_secret,
            on_message=self._on_message,
            on_open=self.on_open,
            on_close=self.on_close,
        )
        self.markets = markets
        self.msg_history: list[str] = []
        self.depth_managers: dict[str, md.DepthCachePlus] = {
            s: md.DepthCachePlus() for s in self.markets
        }
        self.trade_tapes: dict[str, list[md.Trade]] = collections.defaultdict(list)
        self.run_timer = False
        self.queue: asyncio.Queue[CoinbaseMessage] = asyncio.Queue()

    async def __aenter__(self):
        await self._client.open_async()
        # Subscribe to the necessary channels, adjust according to your requirements
        await self._client.subscribe_async(
            product_ids=self.markets,
            channels=[
                'level2',
                'market_trades',
                # 'ticker',
                # 'hearbeats',
            ],
        )
        asyncio.create_task(self.start_timer())
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.run_timer = False
        await self._client.unsubscribe_all_async()
        await self._client.close_async()
        self.queue.put_nowait(None)

    async def start_timer(self):
        self.run_timer = True
        tall = time.time()
        tall = tall // 1 + 1
        while self.run_timer:
            twake = tall
            timesleep = twake - time.time()
            if timesleep > 0:
                await asyncio.sleep(timesleep)
            else:
                logger.warning(f"time sleep is negative: {timesleep}")
            self.cut_trade_tape()
            tall += 1

    def cut_trade_tape(self):
        # logger.debug("cutting trade tapes")
        for symbol, trade_man in self.trade_tapes.items():
            self.depth_managers[symbol].trade_bunch.trades = [t for t in trade_man]
        self.trade_tapes = collections.defaultdict(list)

    async def __aiter__(self) -> AsyncGenerator[CoinbaseMessage, None]:
        while True:
            msg = await self.queue.get()
            # logger.debug(f"yielding message from queue: {msg}")
            yield msg

    def _on_message(self, msg):
        if isinstance(msg, str):
            try:
                message = CoinbaseMessage.model_validate_json(msg)
            except ValidationError as e:
                logger.error(f"Failed to validate message: {msg}")
                logger.error(e)
                return
        else:
            message = msg

        if self.feed_msg_queue:
            self.queue.put_nowait(message)
    
        self.process_message(message)

    def process_message(self, message: CoinbaseMessage):
        assert len(message.events) == 1

        match message.channel:
            case 'subscriptions':
                for event in message.events:
                    assert isinstance(event, SubscriptionsEvent) and event.subscriptions
                    logger.info("Successfully subscribed to the following channels:")
                    for subscription in event.subscriptions:
                        logger.info(f"{subscription=}")
                return
            case 'l2_data':
                for event in message.events:
                    assert isinstance(event, L2Event)
                    if self.PRINT_MESSAGE:
                        logger.debug(
                            f"{message.channel}       {event.type} {event.product_id} {message.timestamp}: {len(event.updates)}"
                        )
                    if self.PRINT_EVENTS:
                        logger.debug(event)
                    match event.type:
                        case 'snapshot':
                            self.depth_managers[event.product_id].reset(
                                event.book_update()
                            )
                        case 'update':
                            self.depth_managers[event.product_id].update(
                                event.book_update()
                            )
            case 'market_trades':
                assert len(message.events) == 1
                assert isinstance(message.events[0], TradeEvent) and (
                    len(set([trade.product_id for trade in message.events[0].trades]))
                    == 1
                )
                for event in message.events:
                    assert isinstance(event, TradeEvent)
                    if self.PRINT_MESSAGE:
                        logger.debug(
                            f"{message.channel} {event.type} {message.timestamp}: {len(event.trades)}"
                        )
                    if self.PRINT_EVENTS:
                        logger.debug(event)
                    match event.type:
                        case 'snapshot':
                            pass
                        case 'update':
                            for trade in event.trades:
                                self.trade_tapes[trade.product_id].append(trade)
            case _:
                logger.error(f"Unhandled channel: {message.channel}")

    def on_open(self):
        logger.info("Connection opened!")

    def on_close(self):
        logger.info("Connection closed!")

    async def multi_generator(self, *, markets: list[str] | None = None):
        """this function is used to run the replay of the market data in parallel for all the symbols."""
        markets = markets or self.markets
        symbol_shapers = {pair: BookShaper() for pair in markets}

        tall = time.time()
        tall = tall // 1 + 1
        while True:
            twake = tall
            timesleep = twake - time.time()
            if timesleep > 0:
                await asyncio.sleep(timesleep)
            else:
                logger.warning(f"time sleep is negative: {timesleep}")

            oneSec = {}
            for symbol, shaper in symbol_shapers.items():
                bids, asks = self.depth_managers[symbol].get_bids_asks()
                if not bids or not asks:
                    logger.warning(f"no bids or asks for {symbol}")
                    break
                await shaper.update_ema(bids, asks, twake)
                list_trades = self.depth_managers[symbol].trade_bunch.trades
                list_trades = [t.to_binance_format() for t in list_trades]
                await shaper.on_trades_bunch(list_trades, force_t_avail=twake)
                oneSec[symbol] = await shaper.make_frames_async(
                    t_avail=twake,
                    bids=bids,
                    asks=asks,
                )
            else:
                yield oneSec
            tall += 1


async def main():
    import pyinstrument

    async with CoinbaseFeed(
        markets=['ETH-USD', 'BTC-USD'], feed_msg_queue=True
    ) as feed:
        feed.PRINT_MESSAGE = True
        await asyncio.sleep(10)

    with pyinstrument.Profiler() as profiler:
        async for msg in feed:
            feed.process_message(msg)

    # profiler.print(show_all=True)
    profiler.open_in_browser(timeline=False)


if __name__ == '__main__':
    asyncio.run(main())
