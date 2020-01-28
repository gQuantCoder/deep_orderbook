import asyncio
import json

from binance import AsyncClient, DepthCacheManager, BinanceSocketManager


async def main():
    # initialise the client
    client = await AsyncClient.create()

    # run some simple requests
    print(json.dumps(await client.get_exchange_info(), indent=2))

    print(json.dumps(await client.get_symbol_ticker(symbol="BTCUSDT"), indent=2))


    # initialise socket manager
    bsm = BinanceSocketManager(client, asyncio.get_event_loop())

    # setup async callback handler for socket messages
    async def handle_evt(msg):
        pair = msg['s']
        print(f'{pair} : {msg}')

    # create listener, can use the `ethkey` value to close the socket later
    trxkey = await bsm.start_trade_socket('TRXBTC', handle_evt)


    # setup an async callback for the Depth Cache
    async def process_depth(depth_cache):
        print(f"symbol {depth_cache.symbol} updated:{depth_cache.update_time}")
        print("Top 5 asks:")
        print(depth_cache.get_asks()[:5])
        print("Top 5 bids:")
        print(depth_cache.get_bids()[:5])

    # create the Depth Cache
    dcm1 = await DepthCacheManager.create(client, asyncio.get_event_loop(), 'TRXBTC', process_depth)

    while True:
        print("doing a sleep")
        await asyncio.sleep(20)


if __name__ == "__main__":

    asyncio.get_event_loop().run_until_complete(main())
    