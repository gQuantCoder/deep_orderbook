
import time
import datetime
import json
import os, sys
import collections
import copy
import threading
import tqdm
from binance.client import Client  # Import the Binance Client

# Import the Binance Socket Manager
from binance.websockets import BinanceSocketManager
from binance.exceptions import BinanceAPIException


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

if __name__ == '__main__':
    rec = Receiver(['ETHBTC'])
    rec.stoprestart()
