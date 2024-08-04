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
from deep_orderbook.utils import logger

class Replayer:
    def __init__(self, data_folder, date_regexp=''):
        self.data_folder = data_folder
        self.date_regexp = date_regexp
        if dates := self.zipped_dates():
            logger.info("Using zipped file generator.")
            self.file_generator = self.book_updates_trades_and_snapshots_zip
        elif dates := self.raw_files_dates():
            logger.info("Using raw file generator.")
            self.file_generator = self.book_updates_trades_and_snapshots_raw
        else:
            logger.error(
                f"The data folder doesn't seem to contain any raw or zipped files: {self.data_folder}"
            )
            raise FileNotFoundError

        self.dates = dates
        if self.dates:
            logger.info(
                f"Found {len(self.dates)} dates: [{self.dates[0]} .. {self.dates[-1]}]"
            )
        else:
            logger.info(
                f"The data folder doesn't seem to contain any raw or zipped files: {self.data_folder}"
            )

    def raw_files(self):
        zs = sorted(glob.glob(f'{self.data_folder}/*/{self.date_regexp}*.json*'))
        logger.debug(f"Raw files: {zs}")
        yield from zs

    def raw_files_dates(self):
        dates = itertools.groupby(
            self.raw_files(), lambda fn: fn.split('/')[-1].split('T')[0]
        )
        dates = [d[0] for d in dates]
        logger.debug(f"Raw file dates: {dates}")
        return dates

    def zipped(self):
        zs = sorted(glob.glob(f'{self.data_folder}/{self.date_regexp}*.zip'))
        logger.debug(f"Zipped files: {zs}")
        yield from zs

    def zipped_dates(self):
        dates = itertools.groupby(
            self.zipped(), lambda fn: fn.split('/')[-1].split('.')[0]
        )
        dates = [d[0] for d in dates]
        logger.debug(f"Zipped file dates: {dates}")
        return dates

    @staticmethod
    def loadjson(filename, open_fc=open):
        try:
            if filename.endswith('.json'):
                data = json.load(open_fc(filename))
                logger.debug(f"Loaded {len(data)} entries from {filename}")
            elif filename.endswith('.jsonl'):
                data = [json.loads(line) for line in open_fc(filename)]
                logger.debug(f"Loaded {len(data)} lines from {filename}")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Error loading {filename}: {e}")
            return []

    def snapshots(self, pair):
        snapshots = sorted(
            glob.glob(f'{self.data_folder}/{pair}/{self.date_regexp}*snapshot.json')
        )
        logger.debug(f"Snapshots for {pair}: {snapshots}")
        return snapshots

    def updates_files(self, pair):
        updates = sorted(
            glob.glob(f'{self.data_folder}/{pair}/{self.date_regexp}*update.json')
        )
        logger.debug(f"Update files for {pair}: {updates}")
        return updates

    def trades_file(self, pair):
        trades = sorted(
            glob.glob(f'{self.data_folder}/{pair}/{self.date_regexp}*trades.json')
        )
        logger.debug(f"Trade files for {pair}: {trades}")
        return trades

    def book_updates_and_trades(self, pair):
        updates = self.updates_files(pair)
        trades = self.trades_file(pair)
        book_updates_trades = [
            (b, b.replace('update', 'trades'))
            for b in updates
            if b.replace('update', 'trades') in trades
        ]
        logger.debug(f"Book updates and trades for {pair}: {book_updates_trades}")
        return book_updates_trades

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
                logger.info(f"Reading from {files=}")
                snapshot_file, trades_file, updates_file = [
                    self.loadjson(fn, open_fc) for fn in files
                ]
                yield updates_file, trades_file, snapshot_file
            elif len(files) == 2:
                logger.info(f"Reading from {files=}")
                trades_file, updates_file = [
                    self.loadjson(fn, open_fc) for fn in files
                ]
                yield updates_file, trades_file, None

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

    async def replayL2_async(self, *, pair: str, shaper: BookShaper):
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
                            logger.debug("Not update:", book_upd.e)
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

    async def multireplayL2_async(self, pairs: list[str]):
        replayers = [
            self.replayL2_async(pair=pair, shaper=BookShaper()) for pair in pairs
        ]
        pairs = [await replayer.__anext__() for replayer in replayers]
        gens = {pairs[i]: replayers[i] for i in range(len(pairs))}
        nexs = {pair: await gens[pair].__anext__() for pair in pairs}

        def next_sec(pair):
            return nexs[pair]['time']

        tall = max([next_sec(p) for p in pairs])
        curs = {pair: nexs[pair] for pair in pairs}
        logger.debug(f'\n{tall=}')
        with tqdm() as pbar:
            while True:
                pbar.set_description(
                    f"tall={datetime.datetime.fromtimestamp(tall, datetime.timezone.utc)}"
                )
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
                    logger.info(
                        f"\nJumping {datetime.timedelta(seconds=jump)} seconds to have an update from one of the symbols"
                    )
                    tall += jump
                else:
                    tall += 1


async def main():
    single_pair = 'BTCUSDT'
    file_replayer = Replayer('../data/crypto', date_regexp='20')
    areplay = file_replayer.replayL2_async(pair=single_pair, shaper=BookShaper())
    num_to_output = 100
    async for bb in areplay:
        num_to_output -= 1
        print(bb)
        if num_to_output < 0:
            break

    single_pair = 'BTC-USD'
    file_replayer = Replayer('data/L2', date_regexp='20')
    areplay = file_replayer.replayL2_async(pair=single_pair, shaper=BookShaper())
    num_to_output = 100
    async for bb in areplay:
        num_to_output -= 1
        print(bb)
        if num_to_output < 0:
            break


if __name__ == '__main__':
    asyncio.run(main())
