import asyncio
from typing import Literal
from coinbase.websocket import WSClient
from rich import print
from pydantic import BaseModel, Field
from datetime import datetime

from deep_orderbook.marketdata import PriceLevel, Trade


***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***





class Subscriptions(BaseModel):
    subscriptions: dict[str, list[str]]


class L2Event(BaseModel):
    type: Literal["snapshot", "update"]
    product_id: str
    updates: list[PriceLevel]


class TradeEvent(BaseModel):
    type: Literal["snapshot", "update"]
    trades: list[Trade]


class Message(BaseModel):
    channel: Literal["l2_data", "market_trades", "subscriptions"]
    timestamp: datetime
    sequence_num: int = Field(alias="sequence_num")
    events: list[L2Event] | list[TradeEvent] | list[Subscriptions]


class CoinbaseFeed:
    RECORD_HISTORY = True
    PRINT_EVENTS = False

    def __init__(self, markets: list[str]) -> None:
        self.client = WSClient(
            api_key=api_key,
            api_secret=api_secret,
            on_message=(
                self.on_message if not self.RECORD_HISTORY else self.recorded_on_message
            ),
            on_open=self.on_open,
            on_close=self.on_close,
        )
        self.markets = markets
        self.msg_history: list[str] = []

    async def __aenter__(self):
        await self.client.open_async()
        # Subscribe to the necessary channels, adjust according to your requirements
        await self.client.subscribe_async(
            product_ids=["ETH-USD"],
            channels=[
                "level2",
                "market_trades",
                # 'ticker',
                # 'hearbeats',
            ],
        )
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.client.unsubscribe_all_async()
        await self.client.close_async()

    def recorded_on_message(self, msg):
        self.msg_history.append(msg)
        self.on_message(msg)

    def on_message(self, msg):
        message = Message.model_validate_json(msg)

        # Assuming your aggregation logic here (simplified for demonstration)
        match message.channel:
            case "subscriptions":
                for event in message.events:
                    print("Successfully subscribed to the following channels:")
                    for channel, products in event.subscriptions.items():
                        print(f"{channel}: {products}")
                return  # Return early as there's no further processing needed for these messages
            case "l2_data":
                for event in message.events:
                    print(
                        f"{message.channel}       {event.type} {message.timestamp}: {len(event.updates)}"
                    )
                    if self.PRINT_EVENTS:
                        print(event)
                    match event.type:
                        case "snapshot":
                            pass
                        case "update":
                            pass
            case "market_trades":
                for event in message.events:
                    print(
                        f"{message.channel} {event.type} {message.timestamp}: {len(event.trades)}"
                    )
                    if self.PRINT_EVENTS:
                        print(event)
                    match event.type:
                        case "snapshot":
                            pass
                        case "update":
                            pass
            case _:
                print(f"Unhandled channel: {channel}")
                print(msg[:256])
                # Handle other message types as needed
                pass

    def on_open(self):
        print("Connection opened!")

    def on_close(self):
        print("Connection closed!")


async def main():
    import pyinstrument

    try:
        async with CoinbaseFeed(markets=["ETH-USD"]) as coinbase:
            await asyncio.sleep(5)
    except Exception as e:
        print(f"Exception: {e}")

    with pyinstrument.Profiler(interval=0.00001) as profiler:
        for msg in coinbase.msg_history:
            coinbase.on_message(msg)

    # profiler.print(show_all=True)
    profiler.open_in_browser(timeline=False)


if __name__ == "__main__":
    asyncio.run(main())
