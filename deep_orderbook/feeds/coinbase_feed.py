import asyncio
import collections
import time
from typing import Literal
from coinbase.websocket import WSClient
from rich import print
from pydantic import BaseModel, Field
from datetime import datetime

import deep_orderbook.marketdata as md
from deep_orderbook.shapper import BookShapper


# read creds from .env using pydantic
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    api_key: str = ''
    api_secret: str = ''
    model_config = SettingsConfigDict(
        env_file='credentials/coinbase.txt', env_file_encoding='utf-8'
    )


class Subscriptions(BaseModel):
    subscriptions: dict[str, list[str]]


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


class TradeEvent(BaseModel):
    type: Literal['snapshot', 'update']
    trades: list[md.Trade]


class Message(BaseModel):
    channel: Literal['l2_data', 'market_trades', 'subscriptions']
    timestamp: datetime
    sequence_num: int = Field(alias='sequence_num')
    events: list[L2Event] | list[TradeEvent] | list[Subscriptions]


class CoinbaseFeed:
    PRINT_EVENTS = False
    PRINT_MESSAGE = False

    def __init__(self, markets: list[str], record_history=False) -> None:
        settings = Settings()
        self.RECORD_HISTORY = record_history
        # print(settings.api_key, settings.api_secret)
        self.client = WSClient(
            api_key=settings.api_key,
            api_secret=settings.api_secret,
            on_message=(
                self.on_message if not self.RECORD_HISTORY else self.recorded_on_message
            ),
            on_open=self.on_open,
            on_close=self.on_close,
        )
        self.markets = markets
        self.msg_history: list[str] = []
        self.depth_managers: dict[str, md.DepthCachePlus] = {
            s: md.DepthCachePlus() for s in self.markets
        }
        self.trade_managers: dict[str, list[md.Trade]] = collections.defaultdict(list)
        self.run_timer = False

    async def __aenter__(self):
        await self.client.open_async()
        # Subscribe to the necessary channels, adjust according to your requirements
        await self.client.subscribe_async(
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
        await self.client.unsubscribe_all_async()
        await self.client.close_async()

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
                print(f"time sleep is negative: {timesleep}")
            self.cut_trade_tape()
            tall += 1

    def cut_trade_tape(self):
        # print('cut')
        for symbol, trade_man in self.trade_managers.items():
            self.depth_managers[symbol].trades = [t for t in trade_man]
        self.trade_managers = collections.defaultdict(list)

    def recorded_on_message(self, msg):
        self.msg_history.append(msg)
        self.on_message(msg)

    def on_message(self, msg):
        message = Message.model_validate_json(msg)
        assert len(message.events) == 1

        # Assuming your aggregation logic here (simplified for demonstration)
        match message.channel:
            case 'subscriptions':
                for event in message.events:
                    if not event.subscriptions:
                        print("ERROR: Failed to subscribe to the requested channels")
                        # exit(1)
                    print("Successfully subscribed to the following channels:")
                    for channel, products in event.subscriptions.items():
                        print(f"{channel}: {products}")
                return  # Return early as there's no further processing needed for these messages
            case 'l2_data':
                for event in message.events:
                    if self.PRINT_MESSAGE:
                        print(
                            f"{message.channel}       {event.type} {event.product_id} {message.timestamp}: {len(event.updates)}"
                        )
                    if self.PRINT_EVENTS:
                        print(event)
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
                # asserts there is only one product per update
                assert len(message.events) == 1
                assert (
                    len(set([trade.product_id for trade in message.events[0].trades]))
                    == 1
                )
                for event in message.events:
                    if self.PRINT_MESSAGE:
                        print(
                            f"{message.channel} {event.type} {message.timestamp}: {len(event.trades)}"
                        )
                    if self.PRINT_EVENTS:
                        print(event)
                    match event.type:
                        case 'snapshot':
                            pass
                        case 'update':
                            for trade in event.trades:
                                self.trade_managers[trade.product_id].append(trade)
            case _:
                print(f"Unhandled channel: {channel}")
                print(msg[:256])
                # Handle other message types as needed
                pass

    def on_open(self):
        print("Connection opened!")

    def on_close(self):
        print("Connection closed!")

    async def multi_generator(self, markets: list[str]):
        """this function is a generator of generators, each one for a different symbol.
        It is used to run the replay of the market data in parallel for all the symbols.
        """
        symbol_shappers = {pair: BookShapper() for pair in markets}

        tall = time.time()
        tall = tall // 1 + 1
        while True:
            twake = tall
            timesleep = twake - time.time()
            if timesleep > 0:
                await asyncio.sleep(timesleep)
            else:
                print(f"time sleep is negative: {timesleep}")

            oneSec = {}
            for symbol, shapper in symbol_shappers.items():
                bids, asks = self.depth_managers[symbol].get_bids_asks()
                if not bids or not asks:
                    print(f"no bids or asks for {symbol}")
                    break
                await shapper.update_ema(bids, asks, twake)
                list_trades = self.depth_managers[symbol].trades
                list_trades = [t.to_binanace_format() for t in list_trades]
                await shapper.on_trades_bunch(list_trades, force_t_avail=twake)
                oneSec[symbol] = await shapper.make_frames_async(
                    t_avail=twake, bids=bids, asks=asks
                )
            else:
                yield oneSec
            tall += 1


async def main():
    import pyinstrument

    async with CoinbaseFeed(
        markets=['ETH-USD', 'BTC-USD'], record_history=True
    ) as coinbase:
        coinbase.PRINT_MESSAGE = True
        await asyncio.sleep(10)

    with pyinstrument.Profiler() as profiler:
        for msg in coinbase.msg_history:
            coinbase.on_message(msg)

    # profiler.print(show_all=True)
    profiler.open_in_browser(timeline=False)


if __name__ == '__main__':
    asyncio.run(main())