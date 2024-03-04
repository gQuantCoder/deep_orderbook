import json
import asyncio
from coinbase.websocket import WSClient
from rich import print
from pydantic import BaseModel, Field
from datetime import datetime


***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***

# Initialize your aggregation structures outside of the on_message to ensure they're in scope
order_book_aggregate = {"bids": {}, "asks": {}}
trades_aggregate = []


class Trade(BaseModel):
    # trade_id: str = Field(alias="trade_id")
    product_id: str = Field(alias="product_id")
    price: str
    size: str
    side: str
    # time: datetime


class L2Data(BaseModel):
    side: str
    # event_time: datetime = Field(alias="event_time")
    price_level: str
    new_quantity: str


class Subscriptions(BaseModel):
    subscriptions: dict[str, list[str]]


class L2Event(BaseModel):
    type: str
    product_id: str
    updates: list[L2Data]


class TradeEvent(BaseModel):
    type: str
    trades: list[Trade]


class Message(BaseModel):
    channel: str
    timestamp: datetime
    sequence_num: int
    events: list[L2Event | TradeEvent | Subscriptions]


def on_message(msg):
    global order_book_aggregate, trades_aggregate
    # print(msg[:256])
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
                    f"{message.channel} {event.type} {message.timestamp}: {len(event.updates)}"
                )
                print(event)
                match event.type:
                    case "snapshot":
                        pass
                    case "update":
                        for update in event.updates:
                            # Update the order book aggregate
                            if update.side == "bid":
                                order_book_aggregate["bids"][
                                    update.price_level
                                ] = update.new_quantity
                            elif update.side == "ask":
                                order_book_aggregate["asks"][
                                    update.price_level
                                ] = update.new_quantity
        case "market_trades":
            for event in message.events:
                print(
                    f"{message.channel} {event.type} {message.timestamp}: {len(event.trades)}"
                )
                print(event)
                match event.type:
                    case "snapshot":
                        pass
                    case "update":
                        for trade in event.trades:
                            trades_aggregate.append(trade)
        case _:
            print(f"Unhandled channel: {channel}")
            print(msg[:256])
            # Handle other message types as needed
            pass


def on_open():
    print("Connection opened!")


def on_close():
    print("Connection closed!")


# Create the WSClient instance
client = WSClient(
    api_key=api_key,
    api_secret=api_secret,
    on_message=on_message,
    on_open=on_open,
    on_close=on_close,
)


async def main():
    try:
        client.open()
        # Subscribe to the necessary channels, adjust according to your requirements
        client.subscribe(
            product_ids=["ETH-USD"],
            channels=[
                "level2",
                "market_trades",
                # "ticker",
                # "hearbeats",
            ],
        )

        # Here, you should implement logic to run for a certain period or handle reconnection/exceptions as needed.
        await asyncio.sleep(30)  # Run for 30 seconds for demonstration

    finally:
        client.close()
        # Here, you can print or process the aggregated data
        print(
            json.dumps(order_book_aggregate, indent=2)[:500]
        )  # Example of handling order_book_aggregate


if __name__ == "__main__":
    asyncio.run(main())
