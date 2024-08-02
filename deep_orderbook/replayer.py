import glob
import json
import datetime
import numpy as np
import asyncio
import itertools
from tqdm.auto import tqdm
import zipfile
import deep_orderbook.marketdata as md

from deep_orderbook.shaper import BookShaper

MARKETS = ["ETHBTC", "BTCUSDT", "ETHUSDT", "BNBBTC", "BNBETH", "BNBUSDT"]


class Replayer:
    def __init__(self, data_folder, date_regexp=''):
        self.data_folder = data_folder
        self.date_regexp = date_regexp
        self.dates = self.zipped_dates()
        if self.dates:
            print("using zipped file generator.")
            self.file_generator = self.book_updates_trades_and_snapshots_zip
        else:
            self.dates = self.raw_files_dates()
            if self.dates:
                print("using raw file generator.")
                self.file_generator = self.book_updates_trades_and_snapshots_raw

        if self.dates:
            print(
                f"found {len(self.dates)} dates: [{self.dates[0]} .. {self.dates[-1]}]"
            )
        else:
            print(
                f"the data folder doen't seem to contain any raw or zipped files: {self.data_folder}"
            )

    def raw_files(self):
        zs = sorted(glob.glob(f'{self.data_folder}/*/{self.date_regexp}*.json'))
        yield from zs

    def raw_files_dates(self):
        dates = itertools.groupby(
            self.raw_files(), lambda fn: fn.split('/')[-1].split('T')[0]
        )
        return [d[0] for d in dates]

    def zipped(self):
        zs = sorted(glob.glob(f'{self.data_folder}/{self.date_regexp}*.zip'))
        yield from zs

    def zipped_dates(self):
        dates = itertools.groupby(
            self.zipped(), lambda fn: fn.split('/')[-1].split('.')[0]
        )
        return [d[0] for d in dates]

    @staticmethod
    def loadjson(filename, open_fc=open):
        try:
            return json.load(open_fc(filename))
        except json.JSONDecodeError as e:
            print(f"Error loading {filename}: {e}")
            return []

    def snapshots(self, pair):
        return sorted(
            glob.glob(f'{self.data_folder}/{pair}/{self.date_regexp}*snapshot.json')
        )

    def updates_files(self, pair):
        Bs = sorted(
            glob.glob(f'{self.data_folder}/{pair}/{self.date_regexp}*update.json')
        )
        return Bs

    def trades_file(self, pair):
        Ts = sorted(
            glob.glob(f'{self.data_folder}/{pair}/{self.date_regexp}*trades.json')
        )
        return Ts

    def book_updates_and_trades(self, pair):
        Bs = self.updates_files(pair)
        Ts = self.trades_file(pair)
        return [
            (b, b.replace('update', 'trades'))
            for b in Bs
            if b.replace('update', 'trades') in Ts
        ]

    async def book_updates_trades_and_snapshots_raw(
        self, pair, file_generator=None, open_fc=None
    ):
        file_generator = file_generator or self.raw_files()
        open_fc = open_fc or open
        fns_pair = filter(lambda fn: pair in fn, file_generator)
        fns_pair_time = itertools.groupby(fns_pair, lambda fn: fn.split('/')[-1][:19])
        for js_group in tqdm(fns_pair_time, leave=False):
            ts, gr = js_group
            files = list(gr)
            if len(files) == 3:
                snapshot_file, trades_file, updates_file = [
                    self.loadjson(fn, open_fc) for fn in files
                ]
                yield updates_file, trades_file, snapshot_file

    async def book_updates_trades_and_snapshots_zip(self, pair):
        zs_gen = self.zipped()
        with tqdm(zs_gen, leave=False) as zs_tqdm:
            for z in zs_tqdm:
                zs_tqdm.set_description(f"zip: {z}")
                with zipfile.ZipFile(z) as myzip:
                    fns = sorted(zipfile.ZipFile(z).namelist())
                    async for fff in self.book_updates_trades_and_snapshots_raw(
                        pair, fns, open_fc=myzip.open
                    ):
                        yield fff

    def training_files(self, pair, side_bips, side_width):
        BTs = sorted(
            glob.glob(
                f'{self.data_folder}/sidepix{side_width:03}/{self.date_regexp}*{pair}*ps.npy'
            )
        )
        for fn_ps in BTs:
            fn_bs = fn_ps.replace('ps.npy', 'bs.npy')
            fn_ts = fn_ps.replace('ps.npy', f'time2level-bip{side_bips:02}.npy')
            yield (fn_bs, fn_ps, fn_ts)

    def training_samples(self, pair):
        for fn_bs, fn_ps, fn_ts in self.training_file(pair):
            arr_books = np.load(fn_bs)
            arr_prices = np.load(fn_ps)
            arr_time2level = np.load(fn_ts)
            yield arr_books, arr_prices, arr_time2level

    # @staticmethod
    # def tradesframe(file):
    #     ts = pd.DataFrame(self.loadjson(file))
    #     if ts.empty:
    #         return ts
    #     ts = ts.drop(['M', 's', 'e', 'a'], axis=1).astype(np.float64)
    #     ts['t'] = 1 + ts['E'] // 1000
    #     ts['delay'] = ts['E'] - ts['T']
    #     ts['num'] = ts['l'] - ts['f'] + 1
    #     ts['up'] = 1 - 2*ts['m']
    #     ts.drop(['E', 'T', 'f', 'l', 'm'], axis=1, inplace=True)
    #     ts.set_index(['t'], inplace=True)
    #     return ts

    # @staticmethod
    # def sample(of_file):
    #     return self.loadjson(of_file)[0]

    async def replayL2_async(self, pair: str, shaper: BookShaper):
        yield pair
        file_updates_tqdm = self.file_generator(pair)
        with tqdm(
            file_updates_tqdm, desc="file_updates", leave=False
        ) as file_updates_tqdm:
            async for js_updates, list_trades_msgs, snapshot_msg in file_updates_tqdm:
                list_trades = [md.TradeUpdate(**trade) for trade in list_trades_msgs]
                await shaper.on_trades_bunch(list_trades)

                snapshot = md.BookSnaphsot(**snapshot_msg)
                await shaper.on_snaphsot_async(snapshot)

                with tqdm(
                    js_updates, desc="js_updates", leave=False
                ) as js_updates_tqdm:
                    for book_upd_msg in js_updates_tqdm:
                        book_upd = md.BinanceBookUpdate(**book_upd_msg)
                        if book_upd.e != 'depthUpdate':
                            print("not update:", book_upd.e)
                            continue

                        if book_upd.final_id < snapshot.lastUpdateId:
                            continue

                        eventTime = book_upd.E
                        ts = 1 + eventTime // 1000

                        px = await shaper.on_depth_msg_async(book_upd)

                        t_avail = shaper.secondAvail(book_upd)
                        oneSec = await shaper.make_frames_async(t_avail)
                        BBO = oneSec['bids'].index[0], oneSec['asks'].index[0]

                        js_updates_tqdm.set_description(
                            f"ts={datetime.datetime.fromtimestamp(ts, datetime.timezone.utc)}, tr={len(oneSec['trades']):02}, BBO:{BBO}"
                        )  # ", px={px:16.12f}")

                        yield oneSec

    async def multireplayL2_async(self, pairs: list[str], shaper: BookShaper):
        replayers = [self.replayL2_async(pair, shaper) for pair in pairs]
        pairs = [await replayer.__anext__() for replayer in replayers]
        gens = {pairs[i]: replayers[i] for i in range(len(pairs))}
        nexs = {pair: await gens[pair].__anext__() for pair in pairs}

        def next_sec(pair):
            return nexs[pair]['time']

        tall = max([next_sec(p) for p in pairs])
        curs = {pair: nexs[pair] for pair in pairs}
        print(f'\n{tall=}')
        with tqdm() as pbar:
            while True:
                pbar.set_description(f"tall={datetime.datetime.fromtimestamp(tall, datetime.timezone.utc)}")
                for pair in pairs:
                    while next_sec(pair) < tall:
                        curs[pair] = nexs[pair]
                        try:
                            nexs[pair] = await gens[pair].__anext__()
                        except StopAsyncIteration:
                            return  # break
                yield curs
                next_overall_sec = min([next_sec(pair) for pair in pairs])
                jump = next_overall_sec - tall
                if jump > 60:
                    print(
                        f"\njumping {datetime.timedelta(seconds=jump)} seconds to have an update from one of the symbols"
                    )
                    tall += jump
                else:
                    tall += 1


async def test_raw_replay():
    from aioitertools import enumerate, next as anext

    MARKETS = ["ETHBTC", "BTCUSDT", "ETHUSDT"]

    shaper = BookShaper()
    file_replayer = Replayer('../crypto-trading/data/L2', date_regexp='2020')
    areplay = file_replayer.replayL2_async('ETHBTC', shaper)
    a = await anext(areplay)
    print(a)
    batptr = await anext(areplay)

    for i in range(100):
        batptr = await anext(areplay)
    print(f"bids:\n{batptr['bids'].head()}")
    print(f"asks:\n{batptr['asks'].head()}")
    print(f"prices:\n{batptr['price']}")
    print(f"trades:\n{batptr['trades']}")

    multi_replay = file_replayer.multireplayL2_async(pairs=MARKETS, shaper=shaper)
    d = await anext(multi_replay)
    print(d)

    genarr = BookShaper.gen_array_async(market_replay=multi_replay, markets=MARKETS)
    _ = await anext(genarr)

    genacc = BookShaper.accumulate_array(genarr, markets=MARKETS)
    _ = await anext(genacc)

    every = 10
    LENGTH = 128
    x = []
    async for n, sec in enumerate(genacc):
        allim = []
        for symb, data in sec.items():
            arr = np.stack(data['bs'][-LENGTH:])
            im = arr
            im[:, :, 0] /= 10
            im += 0.5
            allim.append(im)
        allim = np.concatenate(allim, axis=1)


async def test_zipped_replay():
    file_replayer = Replayer('../data/crypto', date_regexp='20')
    shaper = BookShaper()
    s = file_replayer.zipped()
    print(s)

    single_pair = 'BTCUSDT'
    print(f"replaying a single market: {single_pair}")
    areplay = file_replayer.replayL2_async(pair=single_pair, shaper=shaper)
    print(areplay)
    num_to_output = 10
    async for bb in areplay:
        num_to_output -= 1
        print(bb)
        if num_to_output < 0:
            break

    multi_pairs = ['ETHUSDT', 'BTCUSDT', 'ETHBTC']
    print(f"synchronizing and replaying multilple markets: {multi_pairs}")
    file_gen = file_replayer.multireplayL2_async(pairs=multi_pairs, shaper=shaper)
    num_to_output = 10
    async for bb in file_gen:
        num_to_output -= 1
        # print(bb)
        if num_to_output < 0:
            break


async def main():
    await test_zipped_replay()
    await test_raw_replay()


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
