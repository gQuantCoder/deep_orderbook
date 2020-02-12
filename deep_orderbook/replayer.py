import glob
import json
import sys
import time, datetime
import pandas as pd
import numpy as np
import asyncio
import aiofiles
import aioitertools
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
        BTs = sorted(glob.glob(f'{self.data_folder}/{self.date_regexp}*{pair}*ps.npy'))
        for fn_ps in BTs:
            fn_bs = fn_ps.replace('ps.npy', 'bs.npy')
            fn_ts = fn_ps.replace('ps.npy', 'time2level.npy')
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

                    allupdates.set_description(f"ts={datetime.datetime.utcfromtimestamp(ts)}, E={E}, trades={len(trdf):02}, px={px:16.12f}")
                    # assert not prev_ts or ts == 1 + prev_ts
                    # prev_ts = ts

                    yield oneSec
    async def replayL2_async(self, pair, shapper):
        yield pair
        snapshotupdates = {}
        files = tqdm(self.snapshots(pair))
        for snap_file in files:
            try:
                snap = json.load(open(snap_file))
            except:
                continue
            lastUpdateId = snap['lastUpdateId']
            snapshotupdates[lastUpdateId] = snap_file
        snapshotupdates = iter(sorted(snapshotupdates.items()))
        next_snap,snapshot_file = next(snapshotupdates)

        snapshot = json.load(open(snapshot_file))
#        print("\nsnapshot_file", snapshot_file)
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
#                print("\nfupdate", fupdate)
                allupdates = tqdm(js, leave=False)
                # prev_ts = None
                for book_upd in allupdates:
                    if book_upd['e'] != 'depthUpdate':
                        print("not update:", book_upd['e'])
                        continue
                    U = book_upd['U']
                    u = book_upd['u']
                    eventTime = book_upd['E']
                    ts = 1 + eventTime // 1000

                    if next_snap and u >= next_snap:
                        snapshot = json.load(open(snapshot_file))
#                        print("\nsnapshot_file", snapshot_file)

                        lastUpdateId = snapshot['lastUpdateId']
                        await shapper.on_snaphsot_async(snapshot)
                        try:
                            next_snap,snapshot_file = next(snapshotupdates)
                        except StopIteration as e:
                            next_snap = None

                    px = await shapper.on_depth_msg_async(book_upd)

                    t_avail = shapper.secondAvail(book_upd)
                    oneSec = await shapper.make_frames_async(t_avail)
                    BBO = oneSec['bids'].index[0], oneSec['asks'].index[0]

                    allupdates.set_description(f"ts={datetime.datetime.utcfromtimestamp(ts)}, tr={len(shapper.trdf):02}, BBO:{BBO}")#", px={px:16.12f}")
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
        def next_sec(pair): return nexs[pair]['time']
        tall = max([next_sec(p) for p in pairs])
        curs = {pair: nexs[pair] for pair in pairs}
        print('tall', tall)
        while True:
            for pair in pairs:
                while next_sec(pair) < tall:
                    curs[pair] = nexs[pair]
                    try:
                        nexs[pair] = await gens[pair].__anext__()
                    except StopAsyncIteration:
                        return # break
            yield curs
            next_overall_sec = min([next_sec(pair) for pair in pairs])
            jump = next_overall_sec - tall
            if jump > 60:
                print(f"\njumping {datetime.timedelta(seconds=jump)} seconds to have an update from one of the symbols")
                tall += jump
            else:
                tall += 1


async def main():
    markets = ['ETHBTC']
    file_replayer = Replayer('../crypto-trading/data/L2')
    replay = file_replayer.replayL2('ETHBTC', await BookShapper.create())
    for d in replay:
        pass
        break
    
    replay = file_replayer.replayL2_async('ETHBTC', await BookShapper.create())
    async for d in replay:
        pass
        break

    replayers = [file_replayer.replayL2_async(pair, await BookShapper.create()) for pair in markets]
    multi_replay = file_replayer.multireplayL2_async(replayers)
    async for d in multi_replay:
        pass

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    shapper = loop.run_until_complete(main())
