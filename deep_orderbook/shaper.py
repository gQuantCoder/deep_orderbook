import datetime
from typing import AsyncGenerator, Generator, Iterator
import pandas as pd
import numpy as np
import asyncio
import polars as pl

from tqdm.auto import tqdm
import deep_orderbook.marketdata as md
import aioitertools

pd.set_option('display.precision', 12)
pd.set_option('future.no_silent_downcasting', True)
import matplotlib.pyplot as plt


class BookShaper:
    PriceShape = [2, 3]

    def __init__(self):

        self.sec_trades = dict()
        self.ts = None
        self.px = None
        #        self.prev_px = None
        self.emaPrice = None
        self.emaNew = 1 / 32
        self.emptyframe = None

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
        list_trades = list_trades.to_pandas()
        list_trades['up'] = 1 - 2 * (list_trades['side'] == 'SELL')
        list_trades['num'] = len(list_trades)

        self.sec_trades[force_t_avail] = list_trades.drop(columns=['side'])

    async def make_frames_async(self, t_avail, bids=None, asks=None):
        if bids is None or asks is None:
            bids, asks = self.depth_cache.get_bids_asks()

        oneSec = {
            'time': self.ts,
            'price': self.px,
            'bids': pd.DataFrame(bids, columns=['price', 'size']).set_index('price'),
            'asks': pd.DataFrame(asks, columns=['price', 'size']).set_index('price'),
            'trades': self.sec_trades.pop(t_avail, self.emptyframe).set_index('price'),
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

        a_treind_b = np.arcsinh(treind_b.astype(np.float32))
        a_treind_a = np.arcsinh(treind_a.astype(np.float32))
        a_reind_b = np.arcsinh(reind_b.astype(np.float32))
        a_reind_a = np.arcsinh(reind_a.astype(np.float32))
        return a_reind_b, a_reind_a, a_treind_b, a_treind_a

    @staticmethod
    async def gen_array_async(
        market_replay: Iterator,
        markets: list[str],
        width_per_side=64,
        zoom_frac=1 / 256,
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
                tp, arr3d = await BookShaper.make_arr3d(
                    zoom_frac, prev_price, spacing, pair, sec
                )
                market_second[pair] = {'ps': [tp], 'bs': [arr3d]}
            yield market_second

    @staticmethod
    async def make_arr3d(zoom_frac, prev_price, spacing, pair, sec):
        prev_price[pair] = prev_price[pair] or sec['price']
        bib, aib, trb, tra = BookShaper.bin_books(
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
        d = sec['time'] // (
            3600 * 24
        )  # int(datetime.datetime.fromtimestamp(sec['time'], datetime.timezone.utc).strftime('%y%m%d'))
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
        return tp, arr3d

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
            datetotal = datetime.datetime.fromtimestamp(
                int(dt[0]) * 3600 * 24 + int(dt[1]), datetime.timezone.utc
            ).date()

            de = element[market]['ps'][-1][1]
            dateeleme = datetime.datetime.fromtimestamp(
                int(de[0]) * 3600 * 24 + int(de[1]), datetime.timezone.utc
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
        genacc = aioitertools.accumulate(genarr, BookShaper.build)
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
                im[:, :, 0] /= 5
                im += 0.5
                im = im.transpose(1, 0, 2)
                im = np.clip(im, 0, 1)
                allim.append(im[::-1])
            toshow = np.concatenate(allim, axis=0)
            yield toshow


class ArrayShaper:
    def __init__(
        self, zoom_frac: float = 0.004, width_per_side=64, window_length=256
    ) -> None:
        self.zoom_frac = zoom_frac
        self.num_side_lvl = width_per_side
        self.prev_price: float = None  # type: ignore[assignment]
        self.emaNew = 1 / 32
        self.emaPrice: float = None  # type: ignore[assignment]

        self._cut_scales = pl.arange(0, self.num_side_lvl, eager=True)  # ** 2
        self._cut_scales = self._cut_scales / self._cut_scales[-1]
        self.ask_bin_labels = [f"{p:03}" for p in range(self.num_side_lvl)]
        self.bid_bin_labels = [f"-{lab}" for lab in self.ask_bin_labels[::-1]]
        self.ALL_BIN_LABELS = self.bid_bin_labels + self.ask_bin_labels
        self.lev_labels = pl.Enum(self.ALL_BIN_LABELS)

        self.ask_bin_idx = pl.DataFrame(
            {'bin_idx': pl.Series(self.ask_bin_labels, dtype=self.lev_labels)}
        ).sort('bin_idx')
        self.bid_bin_idx = pl.DataFrame(
            {'bin_idx': pl.Series(self.bid_bin_labels, dtype=self.lev_labels)}
        ).sort('bin_idx')
        self.ALL_BIN_INDEX = self.bid_bin_idx.vstack(self.ask_bin_idx)

        self.total_array = np.zeros((window_length, self.num_side_lvl * 2, 3))
        self.prices_array = np.zeros((window_length, 2)) + np.nan

    def update_ema(self, price: float) -> None:
        if self.emaPrice is None:
            self.emaPrice = price
        self.prev_price = self.emaPrice
        self.emaPrice = price * self.emaNew + (self.emaPrice) * (1 - self.emaNew)

    def price_level_binning(
        self, df: pl.DataFrame, all_edges: list[float]
    ) -> pl.DataFrame:
        df_binned = df.with_columns(
            pl.col('price')
            .cut(
                breaks=all_edges,
                labels=self.ALL_BIN_LABELS,
            )
            .cast(self.lev_labels)
            .alias('bin_idx')
        )
        return self.ALL_BIN_INDEX.join(
            df_binned.group_by('bin_idx').agg(pl.col('size').sum().alias('size')),
            on='bin_idx',
            how='left',
        ).fill_null(0)

    def bin_books(
        self,
        one_sec: md.OneSecondEnds,
    ) -> pl.DataFrame:
        """
        This function bins order book and trade data into specified price levels,
        applies cumulative sums, reindexes the data, and applies the arcsinh transformation.
        """
        # print(one_sec.avg_price())
        price_range = self.prev_price * self.zoom_frac

        bid_edges: pl.Series = self.prev_price - self._cut_scales * price_range
        ask_edges: pl.Series = self.prev_price + self._cut_scales * price_range
        all_edges = bid_edges[1:].reverse().append(ask_edges).to_list()

        dfa = one_sec.asks.with_columns((-pl.col('size')).alias('size'))
        dfb = one_sec.bids
        trup = one_sec.trades.filter(pl.col('side') == 'BUY')
        trdn = one_sec.trades.filter(pl.col('side') == 'SELL').with_columns(
            (-pl.col('size')).alias('size')
        )

        dfb = self.price_level_binning(dfb, all_edges)
        dfa = self.price_level_binning(dfa, all_edges)
        df_trup = self.price_level_binning(trup, all_edges)
        df_trdn = self.price_level_binning(trdn, all_edges)

        # sum the sizes for the same bin_idx
        df_book = (
            dfb.join(dfa, on='bin_idx', suffix='_ask', how='left')
            .with_columns(pl.col('size') + pl.col('size_ask').alias('size'))
            .drop('size_ask')
        )
        df_book = df_book.join(df_trup, on='bin_idx', how='left', suffix='_trup')
        df_book = df_book.join(df_trdn, on='bin_idx', how='left', suffix='_trdn')

        # # re-add the edges in a new column "price"
        # df_book = df_book.with_columns(
        #     bid_edges.reverse().append(ask_edges).cast(pl.Float32).alias('price')
        # ).sort('price')

        return df_book

    async def make_arr3d(self, one_sec: md.OneSecondEnds) -> np.ndarray:
        self.update_ema(one_sec.avg_price())
        df_book = self.bin_books(one_sec)
        # print(df_book.reverse()[self.num_side_lvl - 5 : self.num_side_lvl + 5])

        df_3d = df_book.drop('bin_idx')
        df_3d_exp = df_3d.select(pl.all().arcsinh())

        # add a new first axis to represent time
        image_col = df_3d_exp.to_numpy().reshape((1, -1, 3))

        self.total_array = np.roll(self.total_array, -1, axis=0)
        self.total_array[-1] = image_col

        self.prices_array = np.roll(self.prices_array, -1, axis=0)
        self.prices_array[-1] = np.array([lev.price for lev in one_sec.bbos()])
        return self.total_array

    async def build_time_level_trade(self, side_bips=32, side_width=16):
        """
        Vectorized version of build_time_level_trade.
        Calculates the time it takes for the price to hit certain levels without explicit for-loops.

        Parameters:
        books (numpy array): The order book data.
        prices (numpy array): The price data with shape (T, 2) where T is the number of time steps.
        side_bips (int): The number of basis points to consider on each side of the price.
        side_width (int): The width of the price band to consider.

        Returns:
        numpy array: A matrix of time it takes for the price to hit each level.
        """
        books, prices = self.total_array, self.prices_array

        # Define constants
        mult = 0.0001 * side_bips / side_width
        FUTURE = 64

        T = prices.shape[0]
        T_effective = T - FUTURE + 1  # Effective time steps considering FUTURE window

        # Calculate the price step
        pricestep = prices[-1, 1] * mult

        # Compute thresholds for each level
        thresh = np.arange(side_width) * pricestep  # Shape: (side_width,)

        # Get bid and ask prices up to T_effective
        b = prices[:T_effective, 0]  # Bid prices, shape: (T_effective,)
        a = prices[:T_effective, 1]  # Ask prices, shape: (T_effective,)

        # Compute price levels for thresholds
        a_plus_thresh = a[:, np.newaxis] + thresh[np.newaxis, :]  # Shape: (T_effective, side_width)
        b_minus_thresh = b[:, np.newaxis] - thresh[np.newaxis, :]  # Shape: (T_effective, side_width)

        # Get future bids and asks using sliding window view
        asks_future = np.lib.stride_tricks.sliding_window_view(prices[:, 1], window_shape=FUTURE)
        bids_future = np.lib.stride_tricks.sliding_window_view(prices[:, 0], window_shape=FUTURE)

        # Limit asks_future and bids_future to T_effective
        asks_future = asks_future[:T_effective]  # Shape: (T_effective, FUTURE)
        bids_future = bids_future[:T_effective]  # Shape: (T_effective, FUTURE)

        # Expand dimensions for broadcasting
        asks_future = asks_future[:, :, np.newaxis]  # Shape: (T_effective, FUTURE, 1)
        bids_future = bids_future[:, :, np.newaxis]  # Shape: (T_effective, FUTURE, 1)
        a_plus_thresh = a_plus_thresh[:, np.newaxis, :]  # Shape: (T_effective, 1, side_width)
        b_minus_thresh = b_minus_thresh[:, np.newaxis, :]  # Shape: (T_effective, 1, side_width)

        # Compute tradeUp and tradeDn conditions
        tradeUp = asks_future >= a_plus_thresh  # Shape: (T_effective, FUTURE, side_width)
        tradeDn = bids_future <= b_minus_thresh  # Shape: (T_effective, FUTURE, side_width)

        # Exclude the current time step
        tradeUp[:, 0, :] = False
        tradeDn[:, 0, :] = False

        # Compute timeUp and timeDn by finding the first occurrence where the condition is True
        tradeUp_any = tradeUp.any(axis=1)  # Shape: (T_effective, side_width)
        timeUp = np.where(
            tradeUp_any,
            np.argmax(tradeUp, axis=1) + 1,  # Add 1 because we skipped the first time step
            FUTURE * 10
        )

        tradeDn_any = tradeDn.any(axis=1)
        timeDn = np.where(
            tradeDn_any,
            np.argmax(tradeDn, axis=1) + 1,
            FUTURE * 10
        )

        # multiply timeDn and timeUp by the distance to the reference price
        timeDn = timeDn * 1 / (thresh[np.newaxis, :] + 1)
        timeUp = timeUp * 1 / (thresh[np.newaxis, :] + 1)

        # Reverse timeDn along the side_width axis to match the original order
        timeDn_reversed = timeDn[:, ::-1]

        # Concatenate timeDn and timeUp to form the time2levels matrix
        time2levels = np.concatenate([timeDn_reversed, timeUp], axis=1)  # Shape: (T_effective, 2 * side_width)

        # Add a new axis to match the expected output shape
        time2levels = time2levels[:, :, np.newaxis]  # Shape: (T_effective, 2 * side_width, 1)

        # Initialize the full time2levels array with FUTURE * 10 for time steps beyond T_effective
        time2levels_full = np.full((T, 2 * side_width, 1), FUTURE * 10, dtype=np.float32)

        # Assign the computed values to the corresponding positions
        time2levels_full[:T_effective] = time2levels.astype(np.float32)

        # Ensure that the minimum value in time2levels is greater than 0
        assert time2levels_full.min() >= 0

        return 1 / time2levels_full


async def iter_shapes() -> AsyncGenerator[np.ndarray, None]:
    from deep_orderbook.feeds.coinbase_feed import CoinbaseFeed
    from deep_orderbook.replayer import ParquetReplayer

    shaper = ArrayShaper(zoom_frac=0.25)
    MARKETS = ["ETH-USD"]
    replayer = ParquetReplayer('data', date_regexp='2024-08-06')
    async with CoinbaseFeed(
        markets=MARKETS,
        replayer=replayer,
    ) as feed:
        async for onesec in feed.one_second_iterator(max_samples=200):
            books_array = await shaper.make_arr3d(onesec.symbols[MARKETS[0]])
            time_levels = await shaper.build_time_level_trade()
            # print(books_array.shape)
            # print(time_levels.shape)
            yield books_array, time_levels


async def main():
    import pyinstrument

    profiler = pyinstrument.Profiler()
    with profiler:
        async for shape in iter_shapes():
            pass
    profiler.open_in_browser()


if __name__ == '__main__':
    asyncio.run(main())
