import glob
import json
import sys
import time, datetime
import pandas as pd
import numpy as np
import asyncio
import aiofiles
from tqdm.auto import tqdm
from deep_orderbook.shapper import BookShapper

MARKETS = ["ETHBTC", "BTCUSDT", "ETHUSDT", "BNBBTC", "BNBETH", "BNBUSDT"]


class Replayer:
    def __init__(self, data_folder, date_regexp='2020'):
        self.data_folder = data_folder
        self.date_regexp = date_regexp

    def snapshots(self, pair):
        return sorted(glob.glob(f'{self.data_folder}/{pair}/{self.date_regexp}*snapshot.json'))

    def updates(self, pair):
        BTs = sorted(glob.glob(f'{self.data_folder}/{pair}/{self.date_regexp}*update.json'))
        return BTs

    def trades_file(self, pair):
        BTs = sorted(glob.glob(f'{self.data_folder}/{pair}/{self.date_regexp}*trades.json'))
        return BTs

    def training_file(self, pair):
        BTs = sorted(glob.glob(f'{self.data_folder}/training/{self.date_regexp}*{pair}*time2level.npy'))
        for fn_ts in BTs:
            fn_bs = fn_ts.replace('time2level', 'bs')
            fn_ps = fn_ts.replace('time2level', 'ps')
            yield (fn_bs, fn_ps, fn_ts)

    def training_samples(self, pair):
        for (fn_bs, fn_ps, fn_ts) in self.training_file(pair):
            arr_books = np.load(fn_bs)
            arr_prices = np.load(fn_ps)
            arr_time2level = np.load(fn_ts)
            yield arr_books, arr_prices, arr_time2level

    @staticmethod
    def tradesframe(file):
        ts = pd.DataFrame(json.load(open(file)))
        if ts.empty:
            return ts
        ts = ts.drop(['M', 's', 'e', 'a'], axis=1).astype(np.float64)
        ts['t'] = 1 + ts['E'] // 1000
        ts['delay'] = ts['E'] - ts['T']
        ts['num'] = ts['l'] - ts['f'] + 1
        ts['up'] = 1 - 2*ts['m']
        ts.drop(['E', 'T', 'f', 'l', 'm'], axis=1, inplace=True)
        ts.set_index(['t'], inplace=True)
        return ts

    @staticmethod
    def sample(of_file):
        return json.load(open(of_file))[0]

    def replayL2(self, pair, shapper):
        snapshotupdates = {}
        files = tqdm(self.snapshots(pair))
        for snap_file in files:
            snap = json.load(open(snap_file))
            lastUpdateId = snap['lastUpdateId']
            snapshotupdates[lastUpdateId] = snap_file
        snapshotupdates = iter(sorted(snapshotupdates.items()))
        next_snap,snapshot_file = next(snapshotupdates)

        snapshot = json.load(open(snapshot_file))
        lastUpdateId = snapshot['lastUpdateId']
        shapper.on_snaphsot(snapshot)


        file_updates = tqdm(self.updates(pair))
        for fupdate in file_updates:
                ftrades = fupdate.replace('update', 'trades')
                file_updates.set_description(fupdate.replace(self.data_folder, ''))
                alltrdf = self.tradesframe(ftrades)
                js = json.load(open(fupdate))
                allupdates = tqdm(js, leave=False)
                # prev_ts = None
                for book_upd in allupdates:
                    if book_upd['e'] != 'depthUpdate':
                        continue
                    U = book_upd['U']
                    u = book_upd['u']
                    E = book_upd['E']
                    ts = 1 + E // 1000

                    if next_snap and u >= next_snap:
                        snapshot = json.load(open(snapshot_file))
                        lastUpdateId = snapshot['lastUpdateId']
                        shapper.on_snaphsot(snapshot)
                        try:
                            next_snap,snapshot_file = next(snapshotupdates)
                        except StopIteration as e:
                            next_snap = None

                    px = shapper.on_depth_msg(book_upd)

                    if not alltrdf.empty:
                        trdf = alltrdf.iloc[alltrdf.index==ts]
                        trdf = trdf.set_index(['p'])
                        trdf.loc[px] = 0
                        trdf.sort_index(inplace=True)
                        prev_px = px
                    else:
                        trdf = trdf.loc[[prev_px]]
                    oneSec = shapper.on_trades(trdf)

                    allupdates.set_description(f"ts={datetime.datetime.fromtimestamp(ts)}, E={E}, trades={len(trdf):02}, px={px:16.12f}")
                    # assert not prev_ts or ts == 1 + prev_ts
                    # prev_ts = ts

                    yield oneSec
    async def replayL2_async(self, pair, shapper):
        yield pair
        snapshotupdates = {}
        files = tqdm(self.snapshots(pair))
        for snap_file in files:
            snap = json.load(open(snap_file))
            lastUpdateId = snap['lastUpdateId']
            snapshotupdates[lastUpdateId] = snap_file
        snapshotupdates = iter(sorted(snapshotupdates.items()))
        next_snap,snapshot_file = next(snapshotupdates)

        snapshot = json.load(open(snapshot_file))
        lastUpdateId = snapshot['lastUpdateId']
        await shapper.on_snaphsot_async(snapshot)


        file_updates = tqdm(self.updates(pair))
        for fupdate in file_updates:
                trades_file = fupdate.replace('update', 'trades')
                file_updates.set_description(fupdate.replace(self.data_folder, ''))
                async with aiofiles.open(trades_file, 'r') as fp:
                    list_trades = json.loads(await fp.read())
                    await shapper.on_trades_bunch(list_trades)
                js = json.load(open(fupdate))
                allupdates = tqdm(js, leave=False)
                # prev_ts = None
                for book_upd in allupdates:
                    if book_upd['e'] != 'depthUpdate':
                        continue
                    U = book_upd['U']
                    u = book_upd['u']
                    eventTime = book_upd['E']
                    ts = 1 + eventTime // 1000

                    if next_snap and u >= next_snap:
                        snapshot = json.load(open(snapshot_file))
                        lastUpdateId = snapshot['lastUpdateId']
                        await shapper.on_snaphsot_async(snapshot)
                        try:
                            next_snap,snapshot_file = next(snapshotupdates)
                        except StopIteration as e:
                            next_snap = None

                    px = await shapper.on_depth_msg_async(book_upd)

                    t_avail = shapper.secondAvail(book_upd)
                    oneSec = await shapper.make_frames_async(t_avail)

                    allupdates.set_description(f"ts={datetime.datetime.fromtimestamp(ts)}, tr={len(shapper.trdf):02}")#", px={px:16.12f}")
                    # assert not prev_ts or ts == 1 + prev_ts
                    # prev_ts = ts

                    yield oneSec

    @staticmethod
    def multireplayL2(replayers):
        pairs = range(len(replayers))
        gens = {pair: replayers[pair] for pair in pairs}
        nexs = {pair: next(gens[pair]) for pair in pairs}
        def secs(pair): return nexs[pair][2]['time']
        tall = max([secs(p) for p in pairs])
        curs = {pair: nexs[pair] for pair in pairs}
        print('tall', tall)
        while True:
            for pair in pairs:
                while secs(pair) < tall:
                    curs[pair] = nexs[pair]
                    try:
                        nexs[pair] = next(gens[pair])
                    except StopIteration:
                        return # break
            yield curs
            tall += 1

    @staticmethod
    async def multireplayL2_async(replayers):
        pairs = [await replayer.__anext__() for replayer in replayers]
        gens = {pairs[i]: replayers[i] for i in range(len(pairs))}
        nexs = {pair: await gens[pair].__anext__() for pair in pairs}
        def secs(pair): return nexs[pair][2]['time']
        tall = max([secs(p) for p in pairs])
        curs = {pair: nexs[pair] for pair in pairs}
        print('tall', tall)
        while True:
            for pair in pairs:
                while secs(pair) < tall:
                    curs[pair] = nexs[pair]
                    try:
                        nexs[pair] = await gens[pair].__anext__()
                    except StopAsyncIteration:
                        return # break
            yield curs
            tall += 1


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    shapper = loop.run_until_complete(BookShapper.create())
    file_replayer = Replayer('../crypto-trading/data/L2')
    replay = file_replayer.replayL2('ETHBTC', shapper)
    while True:
        b, a, tp, tr = next(replay)

        print(f"bids:\n{b.head()}")
        print(f"asks:\n{a.head()}")
        print(f"prices:\n{tp}")
        print(f"trades:\n{tr}")
        loop.run_until_complete(asyncio.sleep(1))
