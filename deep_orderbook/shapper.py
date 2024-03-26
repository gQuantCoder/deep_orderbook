import datetime
import pandas as pd
import numpy as np
import asyncio

from tqdm.auto import tqdm
import deep_orderbook.marketdata as md
import aioitertools

pd.set_option('display.precision', 12)
pd.set_option('future.no_silent_downcasting', True)
import matplotlib.pyplot as plt


class BookShapper:
    PriceShape = [2, 3]

    def __init__(self):
        self.depth_cache = md.DepthCachePlus(symbol='')

        self.sec_trades = dict()
        self.bids = None
        self.asks = None
        self.tpr = None
        self.ts = None
        self.px = None
        #        self.prev_px = None
        self.emaPrice = None
        self.emaNew = 1 / 32
        self.emptyframe = pd.DataFrame(
            columns=['p', 'q', 'delay', 'num', 'up']
        ).set_index(['p'])

    async def on_snaphsot_async(self, snapshot):
        self.depth_cache.reset(snapshot)

    async def on_depth_msg_async(self, msg: md.BinanceBookUpdate):
        self.depth_cache.update(msg)
        bids, asks = self.depth_cache.get_bids_asks()
        ts = self.secondAvail(msg)
        await self.update_ema(bids, asks, ts)

    async def update_ema(self, bids, asks, ts):
        self.ts = ts
        bbp, bbs = bids[0]
        bap, bas = asks[0]
        price = (bbp * bas + bap * bbs) / (bbs + bas)
        self.px = round(price, 8)
        self.emaPrice = self.px * self.emaNew + (
            self.emaPrice if self.emaPrice is not None else self.px
        ) * (1 - self.emaNew)
        self.px += 1e-12
        return self.px

    @staticmethod
    def secondAvail(tr_dict: md.BinanceUpdate):
        return 1 + tr_dict.E // 1000

    async def on_trades_bunch(self, list_trades, force_t_avail=None):
        list_trades = [
            {k: float(v) for k, v in trs if k not in ['M', 's', 'e', 'trade_id']}
            for trs in list_trades
        ]
        if not list_trades:
            return
        if force_t_avail:
            self.sec_trades[force_t_avail] = self.trades2frame(list_trades).drop(
                ['tavail'], axis=1
            )
            return
        alltrades = self.trades2frame(list_trades)
        for i, l in alltrades.groupby('tavail'):
            # print(f'{i}\n{l}')
            self.sec_trades[i] = l.drop(['tavail'], axis=1)
        return

    @staticmethod
    def trades2frame(list_trades):
        ts = pd.DataFrame(list_trades).set_index('price')
        ts['tavail'] = ts['E'] // 1000 + 1
        ts['delay'] = ts['E'] - ts['T']
        ts['num'] = ts['last_trade_id'] - ts['first_trade_id'] + 1
        ts['up'] = 1 - 2 * ts['is_buyer_maker']
        ts = ts.drop(
            ['E', 'T', 'first_trade_id', 'last_trade_id', 'is_buyer_maker'], axis=1
        )

        # ts.loc[self.px] = 0
        ts.sort_index(inplace=True)
        #        self.prev_px = self.px
        return ts

    async def make_frames_async(self, t_avail, bids=None, asks=None):
        if bids is None or asks is None:
            bids, asks = self.depth_cache.get_bids_asks()

        oneSec = {
            'time': self.ts,
            'price': self.px,
            'bids': pd.DataFrame(bids, columns=['price', 'size']).set_index('price'),
            'asks': pd.DataFrame(asks, columns=['price', 'size']).set_index('price'),
            'trades': self.sec_trades.pop(t_avail, self.emptyframe),
            'emaPrice': self.emaPrice,
        }
        return oneSec

    FRAC_LEVELS = 0.01
    NUM_LEVEL_BINS = 128
    SPACING = np.cumsum(
        0 + np.linspace(0, NUM_LEVEL_BINS, NUM_LEVEL_BINS, endpoint=False)
    )  # * 4
    SPACING = SPACING / SPACING[-1]

    @staticmethod
    def bin_books(
        dfb: pd.DataFrame,
        dfa: pd.DataFrame,
        tr: pd.DataFrame,
        ref_price: float,
        zoom_frac=FRAC_LEVELS,
        spacing=SPACING,
    ):
        """
        This method bins books into price levels and calculates the cumulative sum of the trades at each price level.
        It then reindexes the data to include all price levels in the specified range and fills any missing values with zero.
        Finally, it applies the arcsinh transformation to the data to reduce the impact of extreme values.

        Parameters
        ----------
        dfb : DataFrame
            DataFrame of bid prices.
        dfa : DataFrame
            DataFrame of ask prices.
        tr : DataFrame
            DataFrame of trades.
        ref_price : float
            Reference price used to determine the price range.
        zoom_frac : float, optional
            Fraction of the price range to zoom in on. Default is FRAC_LEVELS.
        spacing : float, optional
            Spacing between price levels. Default is SPACING.

        Returns
        -------
        reind_b : Series
            Transformed bid data.
        reind_a : Series
            Transformed ask data.
        treind_b : Series
            Transformed trade data for bids.
        treind_a : Series
            Transformed trade data for asks.
        """

        # Define price level indices.
        b_idx = np.round(pd.Index(ref_price * (1 - spacing * zoom_frac)), 7)
        a_idx = np.round(pd.Index(ref_price * (1 + spacing * zoom_frac)), 7)
        t_idx = b_idx[::-1].append(a_idx)
        t_idx_inv = t_idx[::-1]

        # Bin bid data into price levels and apply arcsinh transformation.
        reind_b = (
            dfb.cumsum()
            .reindex(t_idx_inv, method='ffill', fill_value=0)
            .diff()
            .fillna(0)[::-1]
        )

        # Bin ask data into price levels and apply arcsinh transformation.
        reind_a = (
            dfa.cumsum().reindex(t_idx, method='ffill', fill_value=0).diff().fillna(0)
        )

        # Bin trade data into price levels for bids and apply arcsinh transformation.
        treind_b = (
            tr[tr['up'] <= 0]
            .groupby(level=0)
            .sum()[::-1]
            .cumsum()
            .reindex(t_idx_inv, method='ffill', fill_value=0)
            .diff()
            .fillna(0.0)[::-1]
        )
        if treind_b.empty:
            treind_b = reind_b * 0.0

        # Bin trade data into price levels for asks and apply arcsinh transformation.
        treind_a = (
            tr[tr['up'] >= 0]
            .groupby(level=0)
            .sum()
            .cumsum()
            .reindex(t_idx, method='ffill', fill_value=0)
            .diff()
            .fillna(0.0)
        )
        if treind_a.empty:
            treind_a = reind_a * 0.0

        treind_b = np.arcsinh(treind_b.astype(np.float32))
        treind_a = np.arcsinh(treind_a.astype(np.float32))
        reind_b = np.arcsinh(reind_b.astype(np.float32))
        reind_a = np.arcsinh(reind_a.astype(np.float32))
        return reind_b, reind_a, treind_b, treind_a

    def sampleArrays(self, replayer, numpoints=None, apply_fnct=None):
        arrs = []
        trrs = []
        pric = []
        prev_price = None
        i = 0
        spacing = np.arange(64)
        # spacing = np.square(spacing) + spacing
        spacing = spacing / spacing[-1]
        for bi, ai, tpi, tri in replayer:
            prev_price = prev_price or tpi['price']
            bib, aib, trb, tra = self.bin_books(
                bi, ai, tri, ref_price=prev_price, zoom_frac=1 / 256, spacing=spacing
            )
            prev_price = tpi['emaPrice']  # 0.5*(tpi['bid'] + tpi['ask'])
            arrs.append(np.concatenate([bib.values - aib.values]))
            trrs.append(np.concatenate([trb.values - tra.values]))
            pric.append(
                np.array(
                    [tpi['price'], tpi['emaPrice'], tpi['bid'], tpi['ask'], tpi['time']]
                )
            )
            i += 1
            if apply_fnct and i % 10 == 0:
                apply_fnct(
                    **{
                        'books': np.stack(arrs),
                        'prices': np.stack(pric),
                        'trades': np.stack(trrs),
                    }
                )
            if numpoints and i >= numpoints:
                break
        books = np.stack(arrs)
        prices = np.stack(pric)
        trades = np.stack(trrs)
        ret = {'books': books, 'prices': prices, 'trades': trades}
        return ret

    def sampleImages(self, books, prices, trades):
        # print(books.shape, prices.shape, trades.shape)
        plt.margins(0.0)
        plt.plot(prices[:, 0])
        plt.plot(prices[:, 1])
        plt.plot(prices[:, 2])
        plt.plot(prices[:, 3])
        plt.show()
        im = np.abs(books[:, :, 0]).copy()
        im[im == 0] = -1
        plt.imshow(im.T, cmap='nipy_spectral', origin='lower')
        plt.show()
        im = np.abs(trades[:, :, 2])
        # im[im == 0] = 3
        plt.imshow(im.T, cmap='nipy_spectral', origin='lower')
        plt.show()
        im0 = books[:, :, 0].T / 10
        im1 = trades[:, :, 0].T / 1
        im2 = trades[:, :, 2].T / 1
        im3 = np.stack([im0, im1, im2], -1) + 0.5
        print(im3.shape)
        plt.imshow(im3[:, :, :], origin='lower')
        plt.show()

    @staticmethod
    async def gen_array_async(
        market_replay, markets, width_per_side=64, zoom_frac=1 / 256
    ):
        # market_replay = self.multireplayL2(markets)
        prev_price = {p: None for p in markets}
        spacing = np.arange(width_per_side)
        # spacing = np.square(spacing) + spacing
        spacing = spacing / spacing[-1]
        spacing = np.arcsin(spacing) * 3 - spacing * 2
        async for second in market_replay:
            market_second = {}  # collections.defaultdict(list)
            for pair in markets:
                sec = second[pair]
                prev_price[pair] = prev_price[pair] or sec['price']
                bib, aib, trb, tra = BookShapper.bin_books(
                    sec['bids'],
                    sec['asks'],
                    sec['trades'],
                    ref_price=prev_price[pair],
                    zoom_frac=zoom_frac,
                    spacing=spacing,
                )
                prev_price[pair] = sec['emaPrice']
                arr0 = bib.values - aib.values
                arr1 = tra.values - trb.values
                utc = datetime.datetime.utcfromtimestamp(sec['time'])
                d = sec['time'] // (3600 * 24)  # int(utc.strftime('%y%m%d'))
                t = sec['time'] % (3600 * 24)  # float(utc.strftime('%H%M%S.%f'))
                lowtrade = sec['trades'].index.min()
                hightrade = sec['trades'].index.max()
                # print(lowtrade, hightrade)
                tp = np.array(
                    [
                        [lowtrade, sec['bids'].index[0], sec['asks'].index[0]],
                        [d, t, hightrade],
                    ],
                    dtype=np.float32,
                )
                # print('tp', tp)
                arr3d = np.concatenate([arr0, arr1[:, ::2]], axis=-1)
                market_second[pair] = {'ps': [tp], 'bs': [arr3d]}
            yield market_second

    @staticmethod
    def build_time_level_trade(books, prices, side_bips=32, side_width=64):
        """
        This function builds a time level trade matrix. It calculates the time
        it takes for the price to hit a certain level.
        The levels are defined by the side_bips and side_width parameters.

        Parameters:
        books (numpy array): The order book data.
        prices (numpy array): The price data.
        side_bips (int): The number of basis points to consider on each side of
        the price. Default is 32.
        side_width (int): The width of the price band to consider. Default is 64.

        Returns:
        numpy array: A matrix of time it takes for the price to hit each level.
        """

        # Define constants
        mult = 0.0001 * side_bips / side_width
        FUTURE = 120

        # Initialize time2levels matrix with default value
        time2levels = np.zeros_like(books[:, : 2 * side_width, :1]) + FUTURE

        # Calculate the price step
        pricestep = prices[0, 0, 1] * mult

        # Loop over all prices
        for i in tqdm(range(prices.shape[0])):

            # Initialize timeupdn list
            timeupdn = []

            # Loop over all price levels
            for j in range(side_width):

                # Calculate the threshold
                thresh = j * pricestep
                # Get the bid and ask prices
                [_, b, a], [d, t, _] = prices[i]
                bids = prices[i : i + FUTURE, 0, 1]
                asks = prices[i : i + FUTURE, 0, 2]

                # Calculate the waitUp and waitDn conditions
                waitUp = bids < a + thresh
                waitDn = asks > b - thresh

                # Calculate the tradeUp and tradeDn conditions
                lowtrade = prices[i : i + FUTURE, 0, 0]
                hightrade = prices[i : i + FUTURE, 1, 2]
                tradeUp = hightrade >= a + thresh
                tradeDn = lowtrade <= b - thresh

                # Update the tradeUp and tradeDn conditions
                tradeUp[0] = False
                tradeDn[0] = False

                # Update the waitUp and waitDn conditions
                waitUp &= ~tradeUp
                waitDn &= ~tradeDn

                # Calculate the timeUp and timeDn
                timeUp = np.argmin(waitUp) or FUTURE * 10
                timeDn = np.argmin(waitDn) or FUTURE * 10

                # Update the timeupdn list
                timeupdn.insert(0, [timeDn])
                timeupdn.append([timeUp])

            # Update the time2levels matrix
            time2levels[i] = timeupdn

        # Ensure that the minimum value in time2levels is greater than 0
        assert time2levels.min() > 0

        # Return the time2levels matrix as a float32 numpy array
        return time2levels.astype(np.float32)

    @staticmethod
    def build(
        total,
        element,
        reduce_func=None,
        max_length=None,
        side_bips=None,
        side_width=None,
    ):
        """
        This function is used to build and save numpy arrays from the given data.
        It also applies a reduce function if provided and handles the data for a new day.

        Parameters:
        total (dict): The total data dictionary.
        element (dict): The element data dictionary.
        reduce_func (function, optional): A function to reduce the data. Defaults to None.
        max_length (int, optional): The maximum length of the data. Defaults to None.
        side_bips (int, optional): The side bips value. Defaults to None.
        side_width (int, optional): The side width value. Defaults to None.
        Returns:
        dict: The updated total data dictionary.
        """

        # If element is None, force_save is set to True
        force_save = element is None
        element = element or total

        for market, second in element.items():
            # Convert the timestamp to date
            dt = total[market]['ps'][-1][1]
            datetotal = datetime.datetime.utcfromtimestamp(
                int(dt[0]) * 3600 * 24 + int(dt[1])
            ).date()

            de = element[market]['ps'][-1][1]
            dateeleme = datetime.datetime.utcfromtimestamp(
                int(de[0]) * 3600 * 24 + int(de[1])
            ).date()

            # Check if it's a new day
            newDay = datetotal < dateeleme

            if not (newDay or force_save):
                # If it's not a new day and force_save is False, add the second data to the total data
                for name, arrs in second.items():
                    total[market][name] += arrs
            else:
                # If it's a new day or force_save is True, save the total data to numpy files
                arrday_bs = np.stack(total[market]['bs']).astype(np.float32)
                arrday_ps = np.stack(total[market]['ps']).astype(np.float32)
                np.save(
                    f'data/sidepix{side_width:03}/{datetotal}-{market}-bs.npy',
                    arrday_bs,
                )
                np.save(
                    f'data/sidepix{side_width:03}/{datetotal}-{market}-ps.npy',
                    arrday_ps,
                )

                # If a reduce function is provided, apply it to the total data and save the result to a numpy file
                if reduce_func is not None:
                    t2l = reduce_func(
                        books=np.stack(total[market]['bs']).astype(np.float32),
                        prices=np.stack(total[market]['ps']).astype(np.float32),
                    )
                    np.save(
                        f'data/sidepix{side_width:03}/{datetotal}-{market}-time2level-bip{side_bips:02}.npy',
                        t2l,
                    )

            # If a max_length is provided, truncate the total data to the max_length
            if max_length:
                for name, arrs in second.items():
                    if len(total[market][name]) > max_length:
                        total[market][name] = total[market][name][-max_length:]

            # If it's a new day, replace the total data with the second data
            if newDay:
                for name, arrs in second.items():
                    total[market][name] = arrs

        return total

    @staticmethod
    def accumulate_array(genarr, markets):
        genacc = aioitertools.accumulate(genarr, BookShapper.build)
        return genacc

    @staticmethod
    async def images(accumulated_arrays, every=10, LENGTH=128):
        async for n, sec in aioitertools.enumerate(accumulated_arrays):
            if n % every:
                continue
            allim = []
            for symb, data in sec.items():
                # prices_dts = np.stack(data['ps'][-LENGTH:])
                books = np.stack(data['bs'][-LENGTH:])
                im = books
                im[:, :, 0] /= 10
                im += 0.5
                im = im.transpose(1, 0, 2)
                im = np.clip(im, 0, 1)
                allim.append(im[::-1])
            toshow = np.concatenate(allim, axis=0)
            yield toshow


async def main():
    shapper = BookShapper()
    print(shapper)
    print(shapper.depth_cache)


if __name__ == '__main__':
    asyncio.run(main())
