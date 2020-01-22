
import time
import datetime
import json
import os, sys
import collections
import copy
import threading
import tqdm
import asyncio
import aiofiles
from binance import AsyncClient, DepthCacheManager # Import the Binance Client

# Import the Binance Socket Manager
from binance.websockets import BinanceSocketManager
from binance.exceptions import BinanceAPIException
from binance.depthcache import DepthCache

# https://github.com/binance-exchange/binance-official-api-docs/blob/master/web-socket-streams.md#how-to-manage-a-local-order-book-correctly

class MessageDepthCacheManager(DepthCacheManager):
    _default_refresh = 60 * 30  # 30 minutes
    @classmethod
    async def create(cls, client, loop, symbol, coro=None, refresh_interval=_default_refresh, bm=None, limit=500, msg_coro=None):
        self = MessageDepthCacheManager()
        self._client = client
        self._loop = loop
        self._symbol = symbol
        self._limit = limit
        self._coro = coro
        self._msg_coro = msg_coro
        self._last_update_id = None
        self._depth_message_buffer = []
        self._bm = bm
        self._depth_cache = DepthCache(self._symbol)
        self._refresh_interval = refresh_interval
        self.trades = list()

        await self._start_socket()
        await self._init_cache()

        return self

    async def _depth_event(self, msg):
        await super()._depth_event(msg)
        await self._msg_coro(msg)

class Receiver:

    @classmethod
    async def create(cls, **kwargs):
        # print(cls)
        self = cls()
        await self.setup(**kwargs)
        return self

    async def setup(self, markets):
        self.markets = markets
        # Instantiate a Client
        self.client = await AsyncClient.create()
        #print(json.dumps(await self.client.get_exchange_info(), indent=2))
        for m in self.markets:
            print(json.dumps(await self.client.get_symbol_ticker(symbol=m), indent=2))

        # Instantiate a BinanceSocketManager, passing in the client that you instantiated
        self.bm = BinanceSocketManager(self.client, loop=asyncio.get_event_loop())
        self.nummsg = collections.defaultdict(int)
        self.conn_keys = []
        self.depth_managers = {}
        self.trade_managers = collections.defaultdict(list)

        await self.stoprestart()

    async def on_depth_msg(self, msg):
        symbol = msg['s']
        # print('no bid' if not depth_cache.get_bids() else '', 'no ask' if not depth_cache.get_asks() else '')
        #self.nummsg[symbol] += 1
        #print(', '.join([f'{s}: {self.nummsg[s]:06}' for s in self.markets]), end='\r')

    async def on_depth(self, depth_cache):
#        print(f"symbol {depth_cache.symbol} updated:{depth_cache.update_time}")
#        print("Top 5 asks:", len(depth_cache.get_asks()))
#        print(depth_cache.get_asks()[:5])
#        print("Top 5 bids:", len(depth_cache.get_bids()))
#        print(depth_cache.get_bids()[:5])
        symbol = depth_cache.symbol
        self.depth_managers[symbol].trades = copy.deepcopy(self.trade_managers[symbol])
        self.trade_managers[symbol] = list()

    async def on_aggtrades(self, msg):
        symbol = msg["s"]
        self.trade_managers[symbol].append(copy.deepcopy(msg))
        self.nummsg[symbol] += 1
        print(', '.join([f'{s}: {self.nummsg[s]:06}' for s in self.markets]), end='\r')

    async def stoprestart(self, dorestart=True):
        # stop the socket manager
        for conn_key in self.conn_keys:
            print(f"stopping socket {conn_key}\n")
            #await self.bm.stop_socket(conn_key)
        conn_keys = []
        self.depth_managers = {}
#        await self.bm.close()

        if dorestart:
            for symbol in self.markets:
                #os.makedirs(f"{DATA_FOLDER}/L2/{symbol}", exist_ok=True)
#                key = await self.bm.start_depth_socket(symbol, self.on_depth)  # , depth='20')
#                print("start", key)
#                conn_keys.append(key)
                key = await self.bm.start_aggtrade_socket(symbol, self.on_aggtrades)
                print("start", key)
                conn_keys.append(key)
                # create the Depth Cache
            for symbol in self.markets:
                depthmanager = await MessageDepthCacheManager.create(self.client, asyncio.get_event_loop(), 
                                                                                symbol, 
                                                                                self.on_depth,
                                                                                bm=self.bm,
                                                                                limit=1000,
                                                                                msg_coro=self.on_depth_msg
                                                                                )
                self.depth_managers[symbol] = depthmanager

            #await self.bm.start()

class Writer(Receiver):
    async def setup(self, markets, data_folder):
        self.store = collections.defaultdict(list)
        self.tradestore = collections.defaultdict(list)
        self.L2folder = f"{data_folder}/L2"
        
        for symbol in markets:
            os.makedirs(f"{self.L2folder}/{symbol}", exist_ok=True)

        await super().setup(markets)

    async def on_depth_msg(self, msg):
        await super().on_depth_msg(msg)
        symbol = msg['s']
        if True:#with self.lock:
            self.store[symbol].append(msg)

    async def on_depth(self, depth_cache):
        await super().on_depth(depth_cache)
        symbol = depth_cache.symbol
        self.tradestore[symbol] += self.depth_managers[symbol].trades

#    async def on_aggtrades(self, msg):
#        await super().on_aggtrades(msg)
#        symbol = msg["s"]
#        if True:#with self.tradelock:
#            self.tradestore[symbol].append(copy.deepcopy(msg))

    async def save_snapshot(self, cur_ts, prev_ts):
        progressbar = self.markets
        for symbol in progressbar:
            snap = datetime.datetime.utcfromtimestamp(cur_ts).isoformat().replace(":", "-")  # .replace('-',"_")
            upds = datetime.datetime.utcfromtimestamp(prev_ts).isoformat().replace(":", "-")  # .replace('-',"_")
            if True:#with self.lock:
                tosave = copy.deepcopy(self.store[symbol])
                self.store[symbol] = list()
                self.nummsg[symbol] = 0
            if True:#with self.tradelock:
                tradetosave = copy.deepcopy(self.tradestore[symbol])
                self.tradestore[symbol] = list()

            async with aiofiles.open(f"{self.L2folder}/{symbol}/{upds}_update.json", "w") as fp:
                await fp.write(json.dumps(tosave))
            async with aiofiles.open(f"{self.L2folder}/{symbol}/{upds}_trades.json", "w") as fp:
                await fp.write(json.dumps(tradetosave))

        if cur_ts:
            progressbar = self.markets
            for symbol in progressbar:
                print(symbol)
                # time.sleep(1)
                L2 = await self.client.get_order_book(symbol=symbol, limit=1000)
                async with aiofiles.open(f"{self.L2folder}/{symbol}/{snap}_snapshot.json", "w") as fp:
                    await fp.write(json.dumps(L2))
        print("\nsaved_snapshot \n")

    async def run_writer(self, SAVE_PERIOD=60*60, stoptime=datetime.time(23, 58, 30)):
        stopat = datetime.datetime.combine(
            (datetime.datetime.now() + datetime.timedelta(hours=1)).date(), stoptime
        )
        print("\nstopping at", stopat, "in", stopat - datetime.datetime.now())

        PERIOD_L2 = 10
        th = 0
        prev_th = 0

        try:
            for s in range(999999999):
                t = time.time()
                # print("t", t)
                await asyncio.sleep(PERIOD_L2 - t % PERIOD_L2)
                ti = int(time.time())
                # print("ti", ti)
                new_th = int(ti // SAVE_PERIOD) * SAVE_PERIOD

                #if not self.bm.is_alive():
                #    break

                if datetime.datetime.now() >= stopat:
                    # saves for next period that will be overwritten by next script
                    await self.save_snapshot(0, prev_th)
                    print("\nexiting\n", "s =", s, "\n")
                    break

                if new_th > th:
                    print("\n", ti, datetime.datetime.fromtimestamp(ti))
                    if th == 0:
                        await self.save_snapshot(ti, ti)
                        prev_th = ti
                    else:
                        await self.save_snapshot(new_th, prev_th)
                        prev_th = new_th
                    th = new_th
        except Exception as e:
            print(e.__class__, e)
            os._exit(1)

        print("\nexited writting loop\n")

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    MARKETS = ["BNBUSDT", "BTCUSDT", "ETHUSDT", "BNBBTC", "ETHBTC", "BNBETH"]
    rec = loop.run_until_complete(Receiver.create(markets=MARKETS))#['ETHBTC', 'BTCUSDT']))
    while True:
        loop.run_until_complete(asyncio.sleep(10))
