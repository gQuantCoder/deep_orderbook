from typing import AsyncGenerator, cast, Iterator
import numpy as np
import asyncio
import random
from pathlib import Path

from deep_orderbook.config import ReplayConfig, ShaperConfig
from deep_orderbook.utils import logger
from deep_orderbook.cache_manager import ArrayCache
from deep_orderbook.feeds.coinbase_feed import CoinbaseFeed, CoinbaseMessage
from deep_orderbook.replayer import ParquetReplayer
from deep_orderbook.shaper import ArrayShaper
import deep_orderbook.marketdata as md


class SimpleShaper(ArrayShaper):
    def __init__(self, config: ShaperConfig) -> None:
        super().__init__(config)  # Initialize parent class
        # Override the array shapes for our simpler case
        self.total_array = np.zeros(
            (self.config.rolling_window_size, 2, 1)  # Time, Features, Channels
        )
        self.prices_array = np.zeros((self.config.rolling_window_size, 2)) + np.nan

    async def make_arr3d(
        self, new_books: md.OneSecondEnds
    ) -> tuple[np.ndarray, np.ndarray]:
        # Calculate book pressure
        total_bid_size = new_books.bids['size'].sum()
        total_ask_size = new_books.asks['size'].sum()
        book_pressure = (total_bid_size - total_ask_size) / (
            total_bid_size + total_ask_size + 1e-6
        )

        # Get current prices
        bbo = new_books.bbos()
        price_col = np.array([lev.price for lev in bbo])
        mid_price = (price_col[0] + price_col[1]) / 2

        # Create feature array with book pressure
        image_col = np.array([[book_pressure]], dtype=np.float32)

        # Update rolling arrays
        self.total_array = np.roll(self.total_array, -1, axis=0)
        self.prices_array = np.roll(self.prices_array, -1, axis=0)
        self.total_array[-1] = image_col
        self.prices_array[-1] = price_col

        return image_col, price_col

    async def build_time_level_trade(self) -> np.ndarray:
        """
        Compute forward returns for each time step.
        Returns array of shape (T, 2, 1) where:
        - First feature is the forward return
        - Second feature is the book pressure
        """
        prices = self.prices_array
        FUTURE = self.config.look_ahead

        # Check for NaN in input prices
        if np.isnan(prices).any():
            logger.warning(
                f"NaN found in prices array: {np.isnan(prices).sum()} NaN values"
            )
            # Replace NaN with the last valid price
            prices = np.nan_to_num(prices, nan=prices[~np.isnan(prices)].mean())

        # Calculate mid prices
        mid_prices = (prices[:, 0] + prices[:, 1]) / 2

        # Create sliding windows for future prices
        future_prices = np.lib.stride_tricks.sliding_window_view(
            np.pad(mid_prices, (0, FUTURE - 1), mode='edge'),
            window_shape=FUTURE,
        )[: len(mid_prices)]

        # Calculate forward returns
        current_prices = mid_prices[:, np.newaxis]
        forward_returns = (
            future_prices[:, -1] - current_prices[:, 0]
        ) / current_prices[:, 0]

        # Reshape to match expected dimensions (T, 1, 1)
        forward_returns = forward_returns.reshape(-1, 1, 1)
        
        # Get book pressure from total_array and ensure same shape
        book_pressure = self.total_array[:len(forward_returns), :1, :]

        # Combine forward returns and book pressure
        time2levels = np.concatenate([forward_returns, book_pressure], axis=1)

        return time2levels.astype(np.float32)


async def iter_shapes_t2l(
    replay_config: ReplayConfig,
    shaper_config: ShaperConfig,
    live: bool = False,
) -> AsyncGenerator[tuple[np.ndarray, np.ndarray, np.ndarray], None]:
    """Iterator that yields shaped arrays from market data, using cache when possible."""
    cache = ArrayCache(cache_dir=Path('cache/simple_shaper'))
    shaper = SimpleShaper(config=shaper_config)
    collector = cache.create_collector()
    replayer = ParquetReplayer(config=replay_config) if not live else None

    current_file_idx = 0
    if shaper_config.use_cache and not live and replayer is not None:
        parquet_files = replay_config.file_list()

        while current_file_idx < len(parquet_files):
            current_file = parquet_files[current_file_idx]
            cached_data = cache.load_cached(current_file, shaper_config, replay_config)

            if cached_data is not None:
                # Use cached data for this file
                logger.debug(f"Using cached data from {current_file}")
                books_array, time_levels, prices_array = cached_data
                total_length = len(books_array)

                end_indexes = list(
                    range(1, 1 + total_length, shaper_config.window_stride)
                )
                if replay_config.randomize:
                    end_indexes = random.sample(end_indexes, len(end_indexes))
                for end_idx in end_indexes:
                    start_idx = max(0, end_idx - shaper_config.rolling_window_size)
                    window_books = books_array[start_idx:end_idx]
                    window_times = time_levels[start_idx:end_idx]
                    window_prices = prices_array[start_idx:end_idx]

                    if not shaper_config.only_full_arrays or (
                        not np.isnan(window_prices).any()
                        and len(window_books) >= shaper_config.rolling_window_size
                    ):
                        yield window_books, window_times, window_prices

                current_file_idx += 1
            else:
                # Cache miss - switch to live processing from this file onwards
                logger.info(
                    f"Cache miss for {current_file}, switching to live processing"
                )
                break
        else:
            logger.debug("All files processed")
            return

    async with CoinbaseFeed(
        config=replay_config,
        replayer=(
            cast(Iterator[CoinbaseMessage], replayer) if replayer is not None else None
        ),
    ) as feed:
        if replayer is not None:
            replayer.skip_n_files(current_file_idx)
        async for onesec in feed.one_second_iterator():
            # Check if we've moved to a new file
            if (
                not live
                and replayer is not None
                and replayer.current_file != collector.current_file
            ):
                # Cache previous file's data if we have any
                if shaper_config.save_cache:
                    await collector.cache_arrays(shaper_config, shaper, replay_config)

                # Reset collector for new file
                collector.reset(replayer.current_file)

            new_books = onesec.symbols[replay_config.markets[0]]
            if new_books.no_bbo():
                continue

            image_col, price_col = await shaper.make_arr3d(new_books)

            # Add arrays and check if we should yield
            if collector.add_arrays(image_col, price_col, shaper_config.window_stride):
                # Get current window size based on only_full_arrays
                if shaper_config.only_full_arrays:
                    # Only yield if we have a full window
                    if not collector.has_full_window(shaper_config.rolling_window_size):
                        continue
                    window_size = shaper_config.rolling_window_size
                else:
                    # Yield whatever we have, up to rolling_window_size
                    window_size = min(
                        len(collector.all_books), shaper_config.rolling_window_size
                    )

                # Get window arrays
                window_books, window_prices = collector.get_window(window_size)

                # Compute time_levels for just this window
                shaper.prices_array = window_prices
                window_times = await shaper.build_time_level_trade()

                # Skip windows with NaN values if only_full_arrays is True
                if (
                    not shaper_config.only_full_arrays
                    or not np.isnan(window_prices).any()
                ):
                    yield window_books, window_times, window_prices

        # Cache the last file's data if we have any
        if shaper_config.save_cache and not live and replayer is not None:
            await collector.cache_arrays(shaper_config, shaper, replay_config)


async def main() -> None:
    import pyinstrument

    replay_config = ReplayConfig(
        date_regexp='2024-11-06*',
        markets=["ETH-USD"],
        data_dir='/media/photoDS216/crypto/',
    )
    shaper_config = ShaperConfig(
        window_stride=1,
        rolling_window_size=1024,
        look_ahead=60,
    )

    logger.warning(f"Starting with config: {replay_config}")
    logger.warning(f"Number of files to process: {replay_config.num_files()}")
    logger.warning(f"Files: {replay_config.file_list()}")

    profiler = pyinstrument.Profiler()
    count = 0
    with profiler:
        async for books_array, time_levels, pxar in iter_shapes_t2l(
            replay_config=replay_config,
            shaper_config=shaper_config,
        ):
            count += 1
            if count % 100 == 0:  # Log every 100 samples
                logger.warning(
                    f"Processed {count} samples. Shapes: "
                    f"books={books_array.shape}, "
                    f"time_levels={time_levels.shape}, "
                    f"prices={pxar.shape}"
                )
                logger.warning(
                    f"Sample values: book_pressure={books_array[-1,0,0]:.3f}, "
                    f"forward_return={time_levels[-1,0,0]:.3f}, "
                    f"bid/ask={pxar[-1]}"
                )

    logger.warning(f"Finished processing {count} samples")


if __name__ == '__main__':
    asyncio.run(main())
