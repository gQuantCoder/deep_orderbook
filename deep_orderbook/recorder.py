
import time
import datetime
import json
import os, sys
import collections
import copy
import threading
import tqdm
import asyncio
from binance.client import Client  # Import the Binance Client

# Import the Binance Socket Manager
from binance.websockets import BinanceSocketManager
from binance.exceptions import BinanceAPIException

# https://github.com/binance-exchange/binance-official-api-docs/blob/master/web-socket-streams.md#how-to-manage-a-local-order-book-correctly


class Receiver:
    def __init__(self, markets):
        self.markets = markets

        # Instantiate a Client
        self.client = Client(api_key='PUBLIC', api_secret='SECRET')

        # Instantiate a BinanceSocketManager, passing in the client that you instantiated
        self.bm = BinanceSocketManager(self.client)
        self.nummsg = collections.defaultdict(int)
        self.conn_keys = []

    def on_depth(self, msg):
        symbol = msg["s"]
#        print('no bid' if not msg['b'] else '', 'no ask' if not msg['a'] else '')
        self.nummsg[symbol] += 1
        print(', '.join([f'{s}: {self.nummsg[s]:06}' for s in self.markets]), end='\r')


    def on_aggtrades(self, msg):
        symbol = msg["s"]
        self.nummsg[symbol] += 1

    def stoprestart(self, dorestart=True):
        # stop the socket manager
        for conn_key in self.conn_keys:
            print(f"stopping socket {conn_key}\n")
            self.bm.stop_socket(conn_key)
        conn_keys = []
        self.bm.close()

        if dorestart:
            for symbol in self.markets:
                #os.makedirs(f"{DATA_FOLDER}/L2/{symbol}", exist_ok=True)
                key = self.bm.start_depth_socket(symbol, self.on_depth)  # , depth='20')
                print("start", key)
                conn_keys.append(key)
                key = self.bm.start_aggtrade_socket(symbol, self.on_aggtrades)
                print("start", key)
                conn_keys.append(key)
            self.bm.start()

class Writer(Receiver):
    def __init__(self, markets, data_folder):
        super().__init__(markets)
        self.store = collections.defaultdict(list)
        self.lock = threading.Lock()
        self.tradestore = collections.defaultdict(list)
        self.tradelock = threading.Lock()

        self.L2folder = f"{data_folder}/L2"
        
        for symbol in self.markets:
            os.makedirs(f"{self.L2folder}/{symbol}", exist_ok=True)

    def on_depth(self, msg):
        super().on_depth(msg)
        symbol = msg["s"]
        with self.lock:
            self.store[symbol].append(msg)

    def on_aggtrades(self, msg):
        super().on_aggtrades(msg)
        symbol = msg["s"]
        with self.tradelock:
            self.tradestore[symbol].append(msg)

    def save_snapshot(self, cur_ts, prev_ts):
        progressbar = self.markets
        for symbol in progressbar:
            snap = datetime.datetime.utcfromtimestamp(cur_ts).isoformat().replace(":", "-")  # .replace('-',"_")
            upds = datetime.datetime.utcfromtimestamp(prev_ts).isoformat().replace(":", "-")  # .replace('-',"_")
            with self.lock:
                tosave = copy.deepcopy(self.store[symbol])
                self.store[symbol] = list()
                self.nummsg[symbol] = 0
            with self.tradelock:
                tradetosave = copy.deepcopy(self.tradestore[symbol])
                self.tradestore[symbol] = list()

            with open(f"{self.L2folder}/{symbol}/{upds}_update.json", "w") as fp:
                json.dump(tosave, fp)
            with open(f"{self.L2folder}/{symbol}/{upds}_trades.json", "w") as fp:
                json.dump(tradetosave, fp)

        if cur_ts:
            progressbar = self.markets
            for symbol in progressbar:
                print(symbol)
                # time.sleep(1)
                L2 = self.client.get_order_book(symbol=symbol, limit=1000)
                with open(f"{self.L2folder}/{symbol}/{snap}_snapshot.json", "w") as fp:
                    json.dump(L2, fp)
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

                if not self.bm.is_alive():
                    break

                if datetime.datetime.now() >= stopat:
                    # saves for next period that will be overwritten by next script
                    self.save_snapshot(0, prev_th)
                    print("\nexiting\n", "s =", s, "\n")
                    break

                if new_th > th:
                    print("\n", ti, datetime.datetime.fromtimestamp(ti))
                    if th == 0:
                        self.save_snapshot(ti, ti)
                        prev_th = ti
                    else:
                        self.save_snapshot(new_th, prev_th)
                        prev_th = new_th
                    th = new_th
        except Exception as e:
            print(e.__class__, e)
            os._exit(1)

        print("\nexited writting loop\n")
        # self.stoprestart(dorestart=False)

if __name__ == '__main__':
    rec = Receiver(['ETHBTC'])
    rec.stoprestart()
