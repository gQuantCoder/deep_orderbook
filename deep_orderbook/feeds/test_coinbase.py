import pytest
import deep_orderbook.feeds.coinbase_feed as cb
from rich import print


L2_message = """
{
    "channel": "l2_data",
    "client_id": "",
    "timestamp": "2024-03-06T19:22:19.684768592Z",
    "sequence_num": 0,
    "events": [
        {
            "type": "snapshot",
            "product_id": "ETH-USD",
            "updates": [
                {
                    "side": "bid",
                    "event_time": "2024-03-06T19:22:19.189498202Z",
                    "price_level": "3855.34",
                    "new_quantity": "0.2"
                },
                {
                    "side": "bid",
                    "event_time": "2024-03-06T19:22:19.189498202Z",
                    "price_level": "3855.3",
                    "new_quantity": "0.30000003"
                },
                {
                    "side": "bid",
                    "event_time": "2024-03-06T19:22:19.189498202Z",
                    "price_level": "3855.28",
                    "new_quantity": "0.18674766"
                },
                {
                    "side": "bid",
                    "event_time": "2024-03-06T19:22:19.189498202Z",
                    "price_level": "3855.27",
                    "new_quantity": "1.73234"
                },
                {
                    "side": "bid",
                    "event_time": "2024-03-06T19:22:19.189498202Z",
                    "price_level": "3855.21",
                    "new_quantity": "1.32649862"
                },
                {
                    "side": "bid",
                    "event_time": "2024-03-06T19:22:19.189498202Z",
                    "price_level": "3855.2",
                    "new_quantity": "1.93301464"
                },
                {
                    "side": "offer",
                    "event_time": "2024-03-06T19:22:19.189498202Z",
                    "price_level": "20650",
                    "new_quantity": "0.205988"
                },
                {
                    "side": "offer",
                    "event_time": "2024-03-06T19:22:19.189498202Z",
                    "price_level": "20750",
                    "new_quantity": "1"
                },
                {
                    "side": "offer",
                    "event_time": "2024-03-06T19:22:19.189498202Z",
                    "price_level": "20833.33",
                    "new_quantity": "1.20173537"
                },
                {
                    "side": "offer",
                    "event_time": "2024-03-06T19:22:19.189498202Z",
                    "price_level": "20875",
                    "new_quantity": "1"
                },
                {
                    "side": "offer",
                    "event_time": "2024-03-06T19:22:19.189498202Z",
                    "price_level": "20900",
                    "new_quantity": "12"
                }
            ]
        }
    ]
}
"""


def test_coinbase_message_L2(msg=L2_message):
    obj = cb.Message.model_validate_json(msg)
    print(f"{obj}")


async def main():
    test_coinbase_message_L2(L2_message)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
