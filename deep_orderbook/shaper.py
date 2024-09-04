from typing import AsyncGenerator
import numpy as np
import asyncio
import polars as pl

from deep_orderbook.config import ReplayConfig, ShaperConfig
from deep_orderbook.utils import logger

import deep_orderbook.marketdata as md


class ArrayShaper:
    def __init__(self, config: ShaperConfig) -> None:
        self.config = config
        self.prev_price: float = None  # type: ignore[assignment]
        self.emaNew = 1 / 32
        self.emaPrice: float = None  # type: ignore[assignment]

        self._cut_scales = pl.arange(0, self.config.num_side_lvl, eager=True)  # ** 2
        self._cut_scales = self._cut_scales / self._cut_scales[-1]
        self.ask_bin_labels = [f"{p:03}" for p in range(self.config.num_side_lvl)]
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

        self.total_array = np.zeros(
            (self.config.rolling_window_size, self.config.num_side_lvl * 2, 3)
        )
        self.prices_array = np.zeros((self.config.rolling_window_size, 2)) + np.nan

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
        price_range = self.prev_price * self.config.zoom_frac

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

    async def make_arr3d(self, new_books: md.OneSecondEnds) -> np.ndarray:
        self.update_ema(new_books.avg_price())
        df_book = self.bin_books(new_books)
        # print(df_book.reverse()[self.num_side_lvl - 5 : self.num_side_lvl + 5])

        df_3d = df_book.drop('bin_idx')
        df_3d_exp = df_3d.select(pl.all().arcsinh())

        # add a new first axis to represent time
        image_col = df_3d_exp.to_numpy().reshape((1, -1, 3))

        self.total_array = np.roll(self.total_array, -1, axis=0)
        self.prices_array = np.roll(self.prices_array, -1, axis=0)
        self.total_array[-1] = image_col
        self.prices_array[-1] = np.array([lev.price for lev in new_books.bbos()])

        return self.total_array

    async def build_time_level_trade(self) -> np.ndarray:
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
        FUTURE = self.config.look_ahead
        side_bips = self.config.look_ahead_side_bips
        side_width = self.config.look_ahead_side_width

        # Define constants
        mult = 0.0001 * side_bips / side_width

        num_t = prices.shape[0]
        T_eff = num_t - FUTURE + 1  # Effective time steps considering FUTURE window

        # Calculate the price step
        pricestep = prices[-1, 1] * mult

        # Compute thresholds for each level
        thresh = np.arange(side_width) * pricestep  # (side_width,)

        # Get bid and ask prices up to T_eff
        b = prices[:T_eff, 0]  # Bid prices, (T_eff,)
        a = prices[:T_eff, 1]  # Ask prices, (T_eff,)

        # Compute price levels for thresholds
        a_plus_thresh = a[:, np.newaxis] + thresh[np.newaxis, :]  # (T_eff, side_width)
        b_minus_thresh = b[:, np.newaxis] - thresh[np.newaxis, :]  # (T_eff, side_width)

        # Get future bids and asks using sliding window view
        asks_future = np.lib.stride_tricks.sliding_window_view(
            prices[:, 1], window_shape=FUTURE
        )
        bids_future = np.lib.stride_tricks.sliding_window_view(
            prices[:, 0], window_shape=FUTURE
        )

        # Limit asks_future and bids_future to T_eff
        asks_future = asks_future[:T_eff]  # (T_eff, FUTURE)
        bids_future = bids_future[:T_eff]  # (T_eff, FUTURE)

        # Expand dimensions for broadcasting
        asks_future = asks_future[:, :, np.newaxis]  # (T_eff, FUTURE, 1)
        bids_future = bids_future[:, :, np.newaxis]  # (T_eff, FUTURE, 1)
        a_plus_thresh = a_plus_thresh[:, np.newaxis, :]  # (T_eff, 1, side_width)
        b_minus_thresh = b_minus_thresh[:, np.newaxis, :]  # (T_eff, 1, side_width)

        # Compute tradeUp and tradeDn conditions
        tradeUp = asks_future >= a_plus_thresh  # (T_eff, FUTURE, side_width)
        tradeDn = bids_future <= b_minus_thresh  # (T_eff, FUTURE, side_width)

        # Exclude the current time step
        tradeUp[:, 0, :] = False
        tradeDn[:, 0, :] = False

        # Compute timeUp and timeDn by finding the first occurrence where the condition is True
        tradeUp_any = tradeUp.any(axis=1)  # (T_eff, side_width)
        timeUp = np.where(tradeUp_any, np.argmax(tradeUp, axis=1) + 1, 1e9)

        tradeDn_any = tradeDn.any(axis=1)
        timeDn = np.where(tradeDn_any, np.argmax(tradeDn, axis=1) + 1, 1e9)

        # multiply timeDn and timeUp by the distance to the reference price
        timeDn = timeDn * 1 / (thresh[np.newaxis, :] + 1)
        timeUp = timeUp * 1 / (thresh[np.newaxis, :] + 1)

        # Reverse timeDn along the side_width axis to match the original order
        timeDn_reversed = timeDn[:, ::-1]

        # Concatenate timeDn and timeUp to form the time2levels matrix
        time2levels = np.concatenate(
            [timeDn_reversed, timeUp], axis=1
        )  # (T_eff, 2 * side_width)

        # Add a new axis to match the expected output shape
        time2levels = time2levels[:, :, np.newaxis]  # (T_eff, 2 * side_width, 1)

        # Initialize the full time2levels array with 1e9 for time steps beyond T_eff
        time2levels_full = np.full(
            (num_t, 2 * side_width, 1), 1e9, dtype=np.float32
        )

        # Assign the computed values to the corresponding positions
        time2levels_full[:T_eff] = time2levels.astype(np.float32)

        # Ensure that the minimum value in time2levels is greater than 0
        assert time2levels_full.min() >= 0

        return 1 / time2levels_full


async def iter_shapes_t2l(
    replay_config: ReplayConfig,
    shaper_config: ShaperConfig,
) -> AsyncGenerator[tuple[np.ndarray, np.ndarray, np.ndarray], None]:
    from deep_orderbook.feeds.coinbase_feed import CoinbaseFeed
    from deep_orderbook.replayer import ParquetReplayer

    shaper = ArrayShaper(config=shaper_config)
    async with CoinbaseFeed(
        config=replay_config,
        replayer=ParquetReplayer(config=replay_config),
    ) as feed:
        async for onesec in feed.one_second_iterator():
            new_books = onesec.symbols[replay_config.markets[0]]
            if new_books.no_bbo():
                logger.warning('Empty books')
                continue
            books_array = await shaper.make_arr3d(new_books)
            time_levels = await shaper.build_time_level_trade()
            if shaper_config.only_full_arrays:
                # print(np.isnan(shaper.prices_array).sum())
                if np.isnan(shaper.prices_array).any():
                    continue
            yield books_array, time_levels, shaper.prices_array


async def main():
    import pyinstrument

    replay_config = ReplayConfig(date_regexp='2024-08-04', max_samples=250)
    shaper_config = ShaperConfig()

    profiler = pyinstrument.Profiler()
    with profiler:
        async for books_array, time_levels, pxar in iter_shapes_t2l(
            replay_config=replay_config, shaper_config=shaper_config
        ):
            pass
    profiler.open_in_browser()


if __name__ == '__main__':
    asyncio.run(main())
